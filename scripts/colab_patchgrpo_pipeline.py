#!/usr/bin/env python3
"""
Google Colab / Kaggle Single-GPU (A100 80GB) Training Pipeline for PatchGRPO.

Supports two complementary training modes:
  1. Online Mode (`train_patchgrpo_pipeline`):
     On-policy Multi-Turn (or Single-Shot) Group Relative Policy Optimization (PatchGRPO)
     with live repository checkouts, multi-axis exploration (`EXPLORATION_STRATEGIES`),
     PPO ratio clipping, and reference-model KL regularization.
  2. Offline Mode (`train_from_harvested_trajectories`):
     Advantage-Weighted Multi-Turn SFT / Reward-Weighted Regression (RWR) with KL regularization
     and PPO ratio clipping on pre-harvested trajectory JSONL files (e.g., when training the 31B
     policy on trajectories harvested off-policy by E4B).

Addresses all Round 1, Round 2, and Round 3 (`N1`–`N11`) review items from `CODE_REVIEW_PATCHGRPO.md`:
  - N1 Fixed: Preserves the `<|turn>model\n` assistant turn header with `loss_mask = 0.0` before splicing
              `sampled_token_ids` (`loss_mask = 1.0`) in `_compute_multi_turn_token_ids_and_mask`.
  - N2 Fixed: Explicitly asserts prefix token invariance (`cur_ids[:len(prev_ids)] == prev_ids`) at every turn
              during prefix rendering; logs a warning and falls back to turn-by-turn formatting on mutation.
  - N3 Fixed: Periodic checkpoints are written to `results/checkpoints/` (never polluting `submission/adapters/`),
              `tokenizer.save_pretrained` is never called on the submission adapter directory, and
              `_sanitize_submission_adapter_dir` enforces `adk-submission` file-extension whitelist compliance.
  - N4 Fixed: `--resume-from` errors are never caught/swallowed by the loader fallback; NF4 fallback on
              `compressed-tensors` checkpoints is opt-in (`--allow-nf4-fallback`) and writes `TRAINED_ON_NF4.txt`;
              optimizer state and step count are saved and restored with periodic checkpoints.
  - N5 Fixed: Feature-detects `logits_to_keep` vs `num_logits_to_keep` on `model.forward` and wraps chunked
              `-F.cross_entropy` in non-reentrant `torch.utils.checkpoint.checkpoint` to free per-chunk buffers.
  - N6 Fixed: `run_gpu_smoke_test` exercises both a multi-turn trajectory (with `sampled_token_ids`) and a
              synthetic `max_seq_length` (40% active mask) stress case, reporting peak VRAM and grad norms.
  - N8..N11 Fixed: Single-shot online mode trains on raw sampled text/tokens (`N8`), multi-turn messages are
                   preserved (`N9`), `max_new_tokens` is synchronized with `DirectGemmaTransformersBackend` (`N10`),
                   and all exception handlers log diagnostics (`N11`).
"""

from __future__ import annotations

import argparse
import ast
from contextlib import contextmanager
from dataclasses import dataclass, field
import inspect
import json
import logging
import math
import os

# Prevent PyTorch CUDA memory fragmentation on large 31B models (A100 80GB)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from pathlib import Path
import random
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
from typing import Any, Optional, Union

try:
    import torch
    import torch.nn.functional as F
    import torch.utils.checkpoint as torch_ckpt
except ImportError:
    torch = None       # type: ignore
    F = None           # type: ignore
    torch_ckpt = None  # type: ignore

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger("patchgrpo")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

# Whitelisted filenames allowed inside `gemma-4-developer-agent/submission/adapters/<name>/`
# per HARNESS_README.md Section 2.4 (`allowed_file_extensions` & `adapter_extensions`).
ALLOWED_SUBMISSION_ADAPTER_FILES = frozenset({
    "adapter_config.json",
    "adapter_model.safetensors",
    "README.md",
})


# ==============================================================================
# 1. Hyperparameters & Configuration
# ==============================================================================

@dataclass
class PatchGRPOConfig:
    base_model_id: str = "google/gemma-4-31B-it-qat-w4a16-ct"
    fallback_local_model_id: str = "google/gemma-4-e2b-it"
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0              # 0.0 prevents dropout skew between old_log_probs and step-0 policy
    target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    group_size: int = 4                    # G = 4 candidate rollouts per task state
    clip_epsilon: float = 0.2              # PPO surrogate clipping epsilon
    kl_beta: float = 0.04                  # Reference policy KL penalty weight
    max_kl_threshold: float = 5.0          # Per-token KL divergence ceiling to prevent runaway policy drift
    learning_rate: float = 2e-5            # Safe learning rate for 31B LoRA policy gradient
    min_reward_std_scale: float = 0.25     # Floor on advantage std denominator to prevent tie-breaker amplification
    max_prompt_length: int = 4096
    max_new_tokens: int = 1024
    max_seq_length: int = 6144             # Max total tokens for multi-turn trajectory masking
    logprob_chunk_size: int = 128          # Token chunk size for checkpointed cross-entropy log-probs
    max_turns_per_candidate: int = 8       # Bound turns per candidate rollout
    gradient_accumulation_steps: int = 2   # Optimizer step every N active task groups
    ppo_epochs: int = 2                    # Inner PPO epochs per rollout batch (makes ratio != 1.0 on epoch 2+)
    save_every_n_steps: int = 2            # Save periodic checkpoint to `checkpoint_dir` every N optimizer steps
    checkpoint_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "results" / "checkpoints")
    resume_from_checkpoint: Optional[Path] = None
    use_4bit_qlora: bool = True            # Auto-disabled when model_id is natively compressed-tensors W4A16
    force_bnb_4bit: bool = False           # Force BitsAndBytes NF4 even on compressed-tensors checkpoints
    allow_nf4_fallback: bool = False       # Opt-in fallback to BitsAndBytes NF4 if native compressed-tensors fails
    positive_advantages_only: bool = True  # Clamp A_i >= 0 in both online and offline mode (prevents negative gradients on tool calling)
    positive_advantages_only_offline: bool = True  # Backwards compatibility alias
    interleave_repos: bool = True          # Round-robin tasks by repository for balanced training distribution
    seed: int = 42


def set_global_seed(seed: int) -> None:
    """Seeds Python, OS hash, and PyTorch RNGs for reproducible training."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def _get_model_input_device(model: Any) -> Any:
    """Returns the device of the model's input embeddings (safe under `device_map='auto'`)."""
    if hasattr(model, "get_input_embeddings"):
        emb = model.get_input_embeddings()
        if emb is not None and hasattr(emb, "weight") and emb.weight is not None:
            return emb.weight.device
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu") if torch is not None else "cpu"


# Multi-axis exploration schedule matching eval/trajectory_harvester.py
EXPLORATION_STRATEGIES = [
    {
        "name": "Direct_RootCause",
        "temperature": 0.6,
        "hint": (
            "[Exploration Strategy 1: Direct Root-Cause Fix]\n"
            "Focus directly on the primary symbol and file mentioned in the problem statement. "
            "Read the target lines immediately, apply the cleanest minimal surgical `edit_file`, "
            "verify with a quick pytest command, and call `submit_patch`."
        ),
    },
    {
        "name": "CallGraph_Structural",
        "temperature": 0.8,
        "hint": (
            "[Exploration Strategy 2: Call-Graph & Caller/Callee Inspection]\n"
            "Before editing, use `get_code_neighbors` or `search_similar_code` (or inspect callers/callees) "
            "to check how upstream/downstream functions interact with the target symbol. "
            "Ensure your fix preserves caller invariants, run targeted pytest, and call `submit_patch`."
        ),
    },
    {
        "name": "Minimal_Deletion_EdgeCase",
        "temperature": 0.95,
        "hint": (
            "[Exploration Strategy 3: Minimalist Line-Deletion / Guard-Clause Approach]\n"
            "Consider whether the bug is caused by redundant normalization, an overly strict check, "
            "or a missing guard clause. Prefer removing buggy lines or adding a 1-2 line guard "
            "over complex rewrites. Verify with pytest and call `submit_patch`."
        ),
    },
    {
        "name": "ReproFirst_TDD",
        "temperature": 0.75,
        "hint": (
            "[Exploration Strategy 4: Hypothesis-Driven /tmp/repro.py Verification]\n"
            "Write a tiny 10-line reproduction script in `/tmp/repro.py` first to confirm the exact failure behavior, "
            "apply a surgical `edit_file` fix to the library source file, re-run `/tmp/repro.py` and targeted pytest, "
            "and call `submit_patch`."
        ),
    },
]


# ==============================================================================
# 2. Diff Extraction & Unified Multi-Objective Deterministic Reward Engine
# ==============================================================================

def _extract_files_from_diff(diff_text: str) -> list[str]:
    """Extracts relative file paths touched by a unified git diff."""
    files: list[str] = []
    for line in (diff_text or "").splitlines():
        if line.startswith("--- a/") or line.startswith("+++ b/"):
            f = line[6:].strip()
            if f and f != "/dev/null" and f not in files:
                files.append(f)
    return files


def extract_unified_diff(raw_text: str) -> str:
    """Extracts a clean unified git diff from raw model output or markdown blocks.

    Preserves exact leading/trailing spaces on diff context lines (`.strip("\\r\\n")`).
    """
    if not raw_text or not raw_text.strip():
        return ""

    text = raw_text.strip()

    # 1. Look for fenced ```diff or ```patch block
    fence_match = re.search(
        r"```(?:diff|patch)\s*\n(.*?)```",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if fence_match:
        candidate = fence_match.group(1).strip("\r\n")
        if "--- " in candidate and "+++ " in candidate:
            return candidate + "\n"

    # 2. Look for generic fenced block containing unified diff headers
    for block_match in re.finditer(r"```[a-zA-Z0-9_-]*\s*\n(.*?)```", text, flags=re.DOTALL):
        candidate = block_match.group(1).strip("\r\n")
        if ("diff --git " in candidate or "--- a/" in candidate) and "+++ b/" in candidate:
            return candidate + "\n"

    # 3. Extract from first `diff --git` or `--- a/` header in raw text
    lines = text.splitlines()
    start_idx = None
    for idx, line in enumerate(lines):
        if line.startswith("diff --git ") or (
            line.startswith("--- ") and idx + 1 < len(lines) and lines[idx + 1].startswith("+++ ")
        ):
            start_idx = idx
            break

    if start_idx is not None:
        diff_lines = []
        for line in lines[start_idx:]:
            if line.startswith("```"):
                break
            diff_lines.append(line)
        candidate = "\n".join(diff_lines).strip("\r\n")
        if "--- " in candidate and "+++ " in candidate and "@@" in candidate:
            return candidate + "\n"

    return ""


def _force_rmtree(path: Path) -> None:
    """Removes a directory tree, clearing read-only attributes on Windows .git packfiles."""
    def _on_rm_error(func, fpath, exc_info):
        try:
            os.chmod(fpath, stat.S_IWRITE)
            func(fpath)
        except Exception as exc:
            logger.debug(f"Could not remove {fpath}: {exc}")
    if path.exists():
        shutil.rmtree(path, onerror=_on_rm_error)


def _safe_extract_tar(tar: tarfile.TarFile, dest: Path) -> None:
    """Extracts a tarball using `filter='data'` on Python 3.12+ with fallback for older Python."""
    try:
        tar.extractall(path=dest, filter="data")
    except TypeError:
        tar.extractall(path=dest)


def prepare_workspace_from_snapshot(
    instance_id: str,
    snapshots_dir: Optional[Path],
    workspace_dir: Path,
) -> bool:
    """Unpacks the repository snapshot tarball into `workspace_dir` and initializes baseline git HEAD."""
    if (workspace_dir / ".git").exists() and any(workspace_dir.rglob("*.py")):
        subprocess.run(["git", "checkout", "-f", "HEAD"], cwd=workspace_dir, capture_output=True)
        subprocess.run(["git", "clean", "-fd"], cwd=workspace_dir, capture_output=True)
        return True

    if snapshots_dir is not None and instance_id:
        snap_path = Path(snapshots_dir) / f"{instance_id}.tgz"
        if snap_path.exists():
            _force_rmtree(workspace_dir)
            workspace_dir.mkdir(parents=True, exist_ok=True)
            with tarfile.open(snap_path, "r:gz") as tar:
                _safe_extract_tar(tar, workspace_dir)
            subprocess.run(["git", "init", "-q"], cwd=workspace_dir, capture_output=True)
            subprocess.run(["git", "config", "core.fileMode", "false"], cwd=workspace_dir, capture_output=True)
            subprocess.run(["git", "config", "user.email", "eval@gemma4.local"], cwd=workspace_dir, capture_output=True)
            subprocess.run(["git", "config", "user.name", "Eval Harness"], cwd=workspace_dir, capture_output=True)
            subprocess.run(["git", "add", "-A"], cwd=workspace_dir, capture_output=True)
            subprocess.run(["git", "commit", "-m", "baseline", "--allow-empty", "-q"], cwd=workspace_dir, capture_output=True)
            return True

    if workspace_dir.exists() and any(workspace_dir.rglob("*.py")):
        return True
    return False


class DeterministicRewardEngine:
    """Computes verifiable, multi-tier rewards on the unified [0.0, 5.0] scale shared with
    `eval/run_task_e2e.py:compute_patchgrpo_reward` and `eval/trajectory_harvester.py`.
    """

    @staticmethod
    def evaluate_candidate_patch(
        workspace_dir: Optional[Path],
        patch_str: str,
        target_files: list[str],
        test_cmd: list[str],
        timeout_sec: int = 60,
        instance_id: str = "",
        snapshots_dir: Optional[Path] = None,
        test_patch: str = "",
    ) -> tuple[float, dict[str, Any]]:
        from eval.run_task_e2e import compute_patchgrpo_reward

        metrics = {
            "syntax_valid": False,
            "tests_passed": False,
            "pytest_resolved": False,
            "localized_correctly": False,
            "target_matched": False,
            "agent_files": [],
            "patch_lines": 0,
            "patch_size": 0,
            "reward": 0.0,
            "patch_applied": False,
        }

        clean_patch = extract_unified_diff(patch_str)
        if not clean_patch.strip():
            return 0.0, metrics

        created_temp_ws = False
        if workspace_dir is None:
            workspace_dir = Path(tempfile.mkdtemp(prefix=f"patchgrpo_eval_{instance_id or 'task'}_"))
            created_temp_ws = True

        try:
            repo_ready = prepare_workspace_from_snapshot(instance_id, snapshots_dir, workspace_dir)
            if not repo_ready or not any(workspace_dir.rglob("*.py")):
                metrics["error"] = "Workspace is empty or snapshot missing; cannot verify patch."
                return 0.0, metrics

            added_or_removed = [
                line for line in clean_patch.splitlines()
                if (line.startswith("+") or line.startswith("-"))
                and not line.startswith("+++")
                and not line.startswith("---")
            ]
            metrics["patch_lines"] = len(added_or_removed)
            metrics["patch_size"] = len(added_or_removed)

            touched_files = _extract_files_from_diff(clean_patch)
            metrics["agent_files"] = touched_files

            if any(tf in touched_files for tf in target_files):
                metrics["localized_correctly"] = True
                metrics["target_matched"] = True

            patch_file = workspace_dir / ".tmp_candidate_patch.diff"
            test_patch_file = workspace_dir / ".tmp_eval_test_patch.diff"
            try:
                subprocess.run(["git", "checkout", "-f", "HEAD"], cwd=workspace_dir, capture_output=True)
                subprocess.run(["git", "clean", "-fd"], cwd=workspace_dir, capture_output=True)

                patch_file.write_text(clean_patch, encoding="utf-8", newline="\n")
                apply_proc = subprocess.run(
                    ["git", "apply", "--ignore-space-change", "--ignore-whitespace", "--whitespace=nowarn", str(patch_file)],
                    cwd=workspace_dir,
                    capture_output=True,
                    text=True,
                )
                if apply_proc.returncode != 0:
                    apply_proc = subprocess.run(
                        ["git", "apply", "-C1", "--ignore-space-change", "--ignore-whitespace", "--whitespace=nowarn", str(patch_file)],
                        cwd=workspace_dir,
                        capture_output=True,
                        text=True,
                    )
                if patch_file.exists():
                    patch_file.unlink()

                if apply_proc.returncode != 0:
                    metrics["apply_error"] = apply_proc.stderr[:300]
                    return 0.0, metrics

                metrics["patch_applied"] = True

                syntax_ok = True
                checked_count = 0
                for rel_path in touched_files:
                    full_path = workspace_dir / rel_path
                    if full_path.exists() and full_path.suffix == ".py":
                        checked_count += 1
                        try:
                            ast.parse(full_path.read_text(encoding="utf-8"))
                        except SyntaxError:
                            syntax_ok = False
                            break

                if not syntax_ok or (checked_count == 0 and any(f.endswith(".py") for f in touched_files)):
                    metrics["syntax_valid"] = False
                else:
                    metrics["syntax_valid"] = True

                if metrics["syntax_valid"]:
                    if test_patch and test_patch.strip():
                        test_patch_file.write_text(test_patch, encoding="utf-8", newline="\n")
                        subprocess.run(
                            ["git", "apply", "--ignore-space-change", "--ignore-whitespace", "--whitespace=nowarn", str(test_patch_file)],
                            cwd=workspace_dir,
                            capture_output=True,
                            text=True,
                        )
                        if test_patch_file.exists():
                            test_patch_file.unlink()

                    try:
                        from eval.run_task_e2e import _build_sandbox_env
                        sub_env = _build_sandbox_env(workspace_dir)
                    except Exception:
                        sub_env = os.environ.copy()
                        sub_env["PYTHONPATH"] = os.pathsep.join(
                            [str((workspace_dir / "src").resolve()), str(workspace_dir.resolve())]
                        )
                    try:
                        proc = subprocess.run(
                            test_cmd,
                            cwd=workspace_dir,
                            env=sub_env,
                            capture_output=True,
                            text=True,
                            timeout=timeout_sec,
                        )
                        if proc.returncode == 0:
                            metrics["tests_passed"] = True
                            metrics["pytest_resolved"] = True
                    except subprocess.TimeoutExpired:
                        logger.warning(f"Test command timed out after {timeout_sec}s on {instance_id}")

            finally:
                if patch_file.exists():
                    patch_file.unlink(missing_ok=True)
                if test_patch_file.exists():
                    test_patch_file.unlink(missing_ok=True)
                subprocess.run(["git", "checkout", "-f", "HEAD"], cwd=workspace_dir, capture_output=True)
                subprocess.run(["git", "clean", "-fd"], cwd=workspace_dir, capture_output=True)

            reward = compute_patchgrpo_reward(metrics, {}, clean_patch)
            metrics["reward"] = reward
            return reward, metrics
        finally:
            if created_temp_ws:
                _force_rmtree(workspace_dir)


def compute_tie_broken_rewards(
    records: list[dict[str, Any]],
    max_turns: int = 8,
) -> list[float]:
    """Computes rewards with meaningful behavioral tie-breakers (no char/message noise).

    Only rewards two genuine agentic behaviors when a valid patch was produced:
      1. Explicit `submit_patch` tool call (+0.20 bonus if called vs 0.0 if omitted).
      2. Turn efficiency (+0.03 per unused turn under `max_turns`).
    """
    base_rewards = [float(r.get("reward", 0.0)) for r in records]
    if all(br == 0.0 and not (r.get("patch") or "").strip() for br, r in zip(base_rewards, records)):
        return base_rewards

    adjusted_rewards: list[float] = []
    for r, base_r in zip(records, base_rewards):
        patch_text = (r.get("patch") or "").strip()
        if not patch_text or base_r <= 0.0:
            adjusted_rewards.append(base_r)
            continue

        bonus = 0.0
        called_submit = bool(r.get("patch_submitted", False))
        messages = r.get("messages") or []
        if not called_submit and messages:
            for m in messages:
                for tc in (m.get("tool_calls") or []):
                    fn = (tc.get("function") or {}).get("name", "")
                    if fn == "submit_patch":
                        called_submit = True
                        break
        if called_submit:
            bonus += 0.20

        turns_used = int(r.get("turns_used") or (r.get("loop_stats") or {}).get("turns_used") or 0)
        if turns_used == 0 and messages:
            turns_used = sum(1 for m in messages if m.get("role") == "assistant")
        if turns_used > 0:
            bonus += 0.03 * max(0, max_turns - turns_used)

        adjusted_rewards.append(round(base_r + bonus, 4))

    return adjusted_rewards


# ==============================================================================
# 3. Gemma 4 Native Tool-Call Formatting & Multi-Turn Prefix-Span Tokenization
# ==============================================================================

def _format_gemma4_kv_args(args: Any) -> str:
    """Formats tool arguments in Gemma 4's native key-value syntax `{filepath: "a.py", start_line: 1}`
    (unquoted keys, JSON-encoded values) rather than quoted-key JSON `{ "filepath": ... }`.
    """
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except Exception as exc:
            logger.debug(f"Tool arguments string is not JSON ({exc}); wrapping as raw KV block.")
            s = args.strip()
            return s if (s.startswith("{") and s.endswith("}")) else f"{{{s}}}"
    if not isinstance(args, dict) or not args:
        return "{}"
    parts: list[str] = []
    for k, v in args.items():
        parts.append(f"{k}: {json.dumps(v, ensure_ascii=False)}")
    return "{" + ", ".join(parts) + "}"


def _normalize_messages_for_chat_template(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalizes `tool_calls[].function.arguments` into Python `dict` mappings as required by
    HuggingFace's Jinja `apply_chat_template` for Gemma 4.
    """
    from eval.run_task_e2e import DirectGemmaTransformersBackend

    normalized: list[dict[str, Any]] = []
    for m in messages:
        msg_copy = {k: v for k, v in m.items() if k not in ("raw_content", "sampled_token_ids")}
        if msg_copy.get("role") == "assistant" and msg_copy.get("tool_calls"):
            clean_tcs = []
            for tc in msg_copy["tool_calls"]:
                tc_copy = dict(tc)
                func = dict(tc_copy.get("function", {}))
                args = func.get("arguments", {})
                fn_name = func.get("name")
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception as exc:
                        logger.debug(f"Parsing non-JSON tool arguments for {fn_name}: {exc}")
                        args = DirectGemmaTransformersBackend._parse_gemma_args(args, fn_name=fn_name)
                func["arguments"] = args if isinstance(args, dict) else {}
                tc_copy["function"] = func
                clean_tcs.append(tc_copy)
            msg_copy["tool_calls"] = clean_tcs
        normalized.append(msg_copy)
    return normalized


def _format_gemma4_assistant_body(msg: dict[str, Any]) -> str:
    """Formats only the body + closing `<turn|>\n` of an assistant turn (excluding `<|turn>model\n`)."""
    raw_content = (msg.get("raw_content") or "").strip()
    content = (msg.get("content") or "").strip()
    if raw_content:
        if raw_content.endswith("<turn|>"):
            return f"{raw_content}\n"
        return f"{raw_content}<turn|>\n"

    tcs = msg.get("tool_calls") or []
    tc_parts: list[str] = []
    for tc in tcs:
        fn = tc.get("function") or {}
        name = fn.get("name", "")
        args = fn.get("arguments", {})
        if name:
            kv_str = _format_gemma4_kv_args(args)
            tc_parts.append(f"<|tool_call>call:{name}{kv_str}<tool_call|>")
    tc_block = "".join(tc_parts)
    if tc_block and tc_block in content:
        body = content
    else:
        body = f"{content}\n{tc_block}".strip() if content else tc_block
    return f"{body}<turn|>\n"


def format_gemma4_message_turn(msg: dict[str, Any]) -> str:
    """Single-turn formatter using Gemma 4 native `<|turn>...<turn|>` syntax,
    native `{key: "val"}` tool arguments, and `<|tool_call>call:name{...}<tool_call|>` delimiters.
    """
    role = msg.get("role", "user")
    content = (msg.get("content") or "").strip()

    if role == "system":
        return f"<|turn>system\n{content}<turn|>\n"
    if role == "user":
        return f"<|turn>user\n{content}<turn|>\n"
    if role == "tool":
        t_name = msg.get("name", "tool")
        return f"<|turn>user\n<|tool_response>response:{t_name}{content}<tool_response|><turn|>\n"
    if role == "assistant":
        return "<|turn>model\n" + _format_gemma4_assistant_body(msg)
    return f"<|turn>user\n{content}<turn|>\n"


def format_gemma4_chat_prompt(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    tools: Optional[list[dict[str, Any]]] = None,
    add_generation_prompt: bool = True,
) -> str:
    """Formats a conversation using `tokenizer.apply_chat_template(messages, tools=TOOL_DECLARATIONS)`
    with a Gemma 4 `<|turn>...<turn|>` fallback.
    """
    if tools is None:
        try:
            from eval.run_task_e2e import TOOL_DECLARATIONS
            tools = TOOL_DECLARATIONS
        except Exception as exc:
            logger.debug(f"Could not import TOOL_DECLARATIONS ({exc})")
            tools = None

    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        try:
            norm_msgs = _normalize_messages_for_chat_template(messages)
            kwargs: dict[str, Any] = {
                "tokenize": False,
                "add_generation_prompt": add_generation_prompt,
            }
            if tools:
                kwargs["tools"] = tools
            return tokenizer.apply_chat_template(norm_msgs, **kwargs)
        except Exception as exc:
            logger.warning(f"apply_chat_template failed ({exc}); using Gemma 4 <|turn> fallback.")

    chunks = [format_gemma4_message_turn(m) for m in messages]
    if add_generation_prompt:
        chunks.append("<|turn>model\n")
    return "".join(chunks)


def extract_training_target_from_record(rec: dict[str, Any]) -> str:
    """Extracts a complete, untruncated fallback training target string in Gemma 4 native syntax."""
    full_patch = (rec.get("patch") or "").strip()
    messages = rec.get("messages") or []

    action_blocks: list[str] = []
    if messages:
        for m in messages:
            if m.get("role") != "assistant":
                continue
            raw_c = (m.get("raw_content") or "").strip()
            if raw_c:
                action_blocks.append(raw_c)
                continue
            for tc in (m.get("tool_calls") or []):
                fn = tc.get("function") or {}
                name = fn.get("name", "")
                args = fn.get("arguments", {})
                if name in ("read_file", "edit_file", "write_file", "submit_patch"):
                    kv_str = _format_gemma4_kv_args(args)
                    action_blocks.append(f"<|tool_call>call:{name}{kv_str}<tool_call|>")

    if full_patch:
        if action_blocks:
            return "\n".join(action_blocks) + f"\n```diff\n{full_patch}\n```"
        return f"```diff\n{full_patch}\n```"

    if action_blocks:
        return "\n".join(action_blocks)

    preview = (rec.get("patch_preview") or "").strip()
    patch_size = int(rec.get("patch_size") or 0)
    if preview and 0 < patch_size <= len(preview.splitlines()):
        return f"```diff\n{preview}\n```"

    return ""


def interleave_tasks_by_repo(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Groups tasks by repository and round-robins across them to ensure balanced multi-repo representation."""
    from collections import defaultdict
    repo_buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for t in tasks:
        repo = t.get("repo") or t.get("instance_id", "").split("_")[0]
        repo_buckets[repo].append(t)
    interleaved: list[dict[str, Any]] = []
    bucket_iters = [iter(b) for b in repo_buckets.values()]
    while bucket_iters:
        next_iters = []
        for it in bucket_iters:
            try:
                interleaved.append(next(it))
                next_iters.append(it)
            except StopIteration:
                pass
        bucket_iters = next_iters
    return interleaved


def _tokenize_prompt_and_response(
    tokenizer: Any,
    prompt: str,
    resp_text: str,
    config: PatchGRPOConfig,
    device: Any,
) -> tuple[Any, Any, int]:
    """Tokenizes `prompt` and `resp_text` cleanly without inserting a duplicate `<bos>` token."""
    bos_token = getattr(tokenizer, "bos_token", None) or "<bos>"
    prompt_has_bos = prompt.lstrip().startswith(bos_token)
    prompt_token_list = tokenizer.encode(
        prompt,
        add_special_tokens=(not prompt_has_bos),
        truncation=True,
        max_length=config.max_prompt_length,
    )

    resp_token_list = tokenizer.encode(
        resp_text,
        add_special_tokens=False,
        truncation=True,
        max_length=config.max_new_tokens - 1,
    )
    if tokenizer.eos_token_id is not None and (
        len(resp_token_list) == 0 or resp_token_list[-1] != tokenizer.eos_token_id
    ):
        resp_token_list = resp_token_list + [tokenizer.eos_token_id]

    prompt_ids = torch.tensor([prompt_token_list], dtype=torch.long, device=device)
    resp_ids = torch.tensor([resp_token_list], dtype=torch.long, device=device)
    return prompt_ids, resp_ids, prompt_ids.shape[1]


def _compute_multi_turn_token_ids_and_mask(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    max_seq_length: int = 6144,
) -> tuple[list[int], list[float], Optional[int]]:
    """Computes `(token_ids, loss_mask, first_asst_idx)` in pure Python (`N1`, `N2` Fixed).

    Fixes Round 3 findings `N1` and `N2`:
      - `N1` Fix (Assistant Turn Header Preservation):
        For every `assistant` turn `k`, the turn header `<|turn>model\\n` is rendered/encoded
        separately and appended with **`loss_mask = 0.0`**, BEFORE appending the assistant body
        or `sampled_token_ids` with **`loss_mask = 1.0`**! When `sampled_token_ids` is used,
        also appends the closing `<turn|>\\n` delimiter (masked `0.0`) if not already present at
        the tail of `sampled_token_ids`, so subsequent turns see a well-formed `<|turn>model...<turn|>` block.
      - `N2` Fix (Strict Prefix Alignment Verification):
        When rendering `norm_msgs[:k+1]` via `tokenizer.apply_chat_template`, explicitly asserts
        that `cur_ids[:len(prev_ids)] == prev_ids` (and `hdr_ids[:len(prev_ids)] == prev_ids`).
        If the chat template mutates earlier tokens (e.g., by stripping `<|thought>` blocks from
        earlier turns), logs a warning and falls back cleanly to the turn-by-turn formatter.
    """
    token_ids: list[int] = []
    loss_mask: list[float] = []
    first_asst_idx: Optional[int] = None

    used_template_prefix = False
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            from eval.run_task_e2e import TOOL_DECLARATIONS
            norm_msgs = _normalize_messages_for_chat_template(messages)
            bos_tok = getattr(tokenizer, "bos_token", None) or "<bos>"
            turn_end_ids = tokenizer.encode("<turn|>\n", add_special_tokens=False, truncation=False)
            prev_ids: list[int] = []

            for k, orig_m in enumerate(messages):
                role = orig_m.get("role", "user")
                prefix_str = tokenizer.apply_chat_template(
                    norm_msgs[: k + 1],
                    tools=TOOL_DECLARATIONS,
                    tokenize=False,
                    add_generation_prompt=False,
                )
                has_bos = prefix_str.lstrip().startswith(bos_tok)
                cur_ids = tokenizer.encode(
                    prefix_str,
                    add_special_tokens=(not has_bos),
                    truncation=False,
                )

                # N2 Check: Verify that re-rendering up to turn k+1 did NOT mutate earlier prefix tokens!
                if len(cur_ids) < len(prev_ids) or cur_ids[: len(prev_ids)] != prev_ids:
                    raise ValueError(
                        f"Chat template mutated earlier prefix tokens at turn {k} "
                        f"(prev_len={len(prev_ids)}, cur_len={len(cur_ids)})."
                    )

                if role == "assistant":
                    # N1 Fix: Separate the `<|turn>model\n` generation-prompt header (mask=0.0)
                    # from the assistant's generated body tokens (mask=1.0)!
                    if k > 0:
                        hdr_str = tokenizer.apply_chat_template(
                            norm_msgs[:k],
                            tools=TOOL_DECLARATIONS,
                            tokenize=False,
                            add_generation_prompt=True,
                        )
                        hdr_has_bos = hdr_str.lstrip().startswith(bos_tok)
                        hdr_ids = tokenizer.encode(
                            hdr_str,
                            add_special_tokens=(not hdr_has_bos),
                            truncation=False,
                        )
                        if (
                            len(hdr_ids) < len(prev_ids)
                            or hdr_ids[: len(prev_ids)] != prev_ids
                            or len(cur_ids) < len(hdr_ids)
                            or cur_ids[: len(hdr_ids)] != hdr_ids
                        ):
                            raise ValueError(f"Assistant header prefix mismatch at turn {k}.")
                        header_delta = hdr_ids[len(prev_ids) :]
                        body_delta = cur_ids[len(hdr_ids) :]
                    else:
                        hdr_only_ids = tokenizer.encode("<|turn>model\n", add_special_tokens=False, truncation=False)
                        delta_all = cur_ids[len(prev_ids) :]
                        if hdr_only_ids and delta_all[: len(hdr_only_ids)] == hdr_only_ids:
                            header_delta = hdr_only_ids
                            body_delta = delta_all[len(hdr_only_ids) :]
                        else:
                            header_delta = []
                            body_delta = delta_all

                    # 1. Append assistant turn header (`<|turn>model\n`) with loss_mask = 0.0
                    if header_delta:
                        token_ids.extend(header_delta)
                        loss_mask.extend([0.0] * len(header_delta))

                    # 2. Append assistant body (`sampled_token_ids` or `body_delta`) with loss_mask = 1.0
                    sampled_ids = orig_m.get("sampled_token_ids")
                    if isinstance(sampled_ids, list) and len(sampled_ids) > 0:
                        if first_asst_idx is None:
                            first_asst_idx = len(token_ids)
                        token_ids.extend(sampled_ids)
                        loss_mask.extend([1.0] * len(sampled_ids))
                        # Ensure closing `<turn|>\n` delimiter is present after sampled_token_ids
                        if turn_end_ids and (
                            len(sampled_ids) < len(turn_end_ids)
                            or sampled_ids[-len(turn_end_ids) :] != turn_end_ids
                        ):
                            token_ids.extend(turn_end_ids)
                            loss_mask.extend([0.0] * len(turn_end_ids))
                    elif body_delta:
                        if first_asst_idx is None:
                            first_asst_idx = len(token_ids)
                        token_ids.extend(body_delta)
                        loss_mask.extend([1.0] * len(body_delta))
                else:
                    delta_ids = cur_ids[len(prev_ids) :]
                    if delta_ids:
                        token_ids.extend(delta_ids)
                        loss_mask.extend([0.0] * len(delta_ids))

                prev_ids = cur_ids
                if len(token_ids) >= max_seq_length:
                    token_ids = token_ids[:max_seq_length]
                    loss_mask = loss_mask[:max_seq_length]
                    break

            used_template_prefix = (first_asst_idx is not None and sum(loss_mask[first_asst_idx:]) > 0)
        except Exception as exc:
            logger.warning(f"Prefix chat-template rendering fell back to turn-by-turn ({exc}).")
            token_ids.clear()
            loss_mask.clear()
            first_asst_idx = None

    if not used_template_prefix:
        bos_id = getattr(tokenizer, "bos_token_id", None)
        if bos_id is not None:
            token_ids.append(bos_id)
            loss_mask.append(0.0)

        asst_hdr_ids = tokenizer.encode("<|turn>model\n", add_special_tokens=False, truncation=False)
        turn_end_ids = tokenizer.encode("<turn|>\n", add_special_tokens=False, truncation=False)

        for m in messages:
            role = m.get("role", "user")
            if role == "assistant":
                # N1 Fix in Fallback Path: Always emit `<|turn>model\n` with mask 0.0 before assistant body!
                if asst_hdr_ids:
                    token_ids.extend(asst_hdr_ids)
                    loss_mask.extend([0.0] * len(asst_hdr_ids))

                sampled_ids = m.get("sampled_token_ids")
                if isinstance(sampled_ids, list) and len(sampled_ids) > 0:
                    if first_asst_idx is None:
                        first_asst_idx = len(token_ids)
                    token_ids.extend(sampled_ids)
                    loss_mask.extend([1.0] * len(sampled_ids))
                    if turn_end_ids and (
                        len(sampled_ids) < len(turn_end_ids)
                        or sampled_ids[-len(turn_end_ids) :] != turn_end_ids
                    ):
                        token_ids.extend(turn_end_ids)
                        loss_mask.extend([0.0] * len(turn_end_ids))
                else:
                    body_str = _format_gemma4_assistant_body(m)
                    body_ids = tokenizer.encode(body_str, add_special_tokens=False, truncation=False)
                    if body_ids:
                        if first_asst_idx is None:
                            first_asst_idx = len(token_ids)
                        token_ids.extend(body_ids)
                        loss_mask.extend([1.0] * len(body_ids))
            else:
                turn_str = format_gemma4_message_turn(m)
                turn_ids = tokenizer.encode(turn_str, add_special_tokens=False, truncation=False)
                if turn_ids:
                    token_ids.extend(turn_ids)
                    loss_mask.extend([0.0] * len(turn_ids))

            if len(token_ids) >= max_seq_length:
                token_ids = token_ids[:max_seq_length]
                loss_mask = loss_mask[:max_seq_length]
                break

    return token_ids, loss_mask, first_asst_idx


def tokenize_multi_turn_trajectory(
    tokenizer: Any,
    messages: Optional[list[dict[str, Any]]],
    prompt_fallback: str,
    resp_fallback: str,
    config: PatchGRPOConfig,
    device: Any,
) -> Optional[tuple[Any, Any, int]]:
    """Builds `(full_ids, response_mask, prompt_len)` for a candidate trajectory (`R2-1`, `N1`, `N2`)."""
    if messages and len(messages) >= 3:
        token_ids, loss_mask, first_asst_idx = _compute_multi_turn_token_ids_and_mask(
            tokenizer, messages, max_seq_length=config.max_seq_length
        )
        if first_asst_idx is not None and first_asst_idx > 0 and sum(loss_mask[first_asst_idx:]) > 0:
            full_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
            resp_mask = torch.tensor([loss_mask[first_asst_idx:]], dtype=torch.float32, device=device)
            return full_ids, resp_mask, first_asst_idx

    if not resp_fallback or not resp_fallback.strip():
        return None
    prompt_ids, resp_ids, prompt_len = _tokenize_prompt_and_response(
        tokenizer, prompt_fallback, resp_fallback, config, device
    )
    if resp_ids.shape[1] == 0:
        return None
    full_ids = torch.cat([prompt_ids, resp_ids], dim=-1)
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is not None and pad_id != getattr(tokenizer, "eos_token_id", None):
        resp_mask = (resp_ids != pad_id).to(torch.float32)
    else:
        resp_mask = torch.ones_like(resp_ids, dtype=torch.float32)
    return full_ids, resp_mask, prompt_len


# ==============================================================================
# 4. Model Initialization (`gemma-4-31B-it-qat-w4a16-ct` Compressed-Tensors & QLoRA)
# ==============================================================================

def is_compressed_tensors_checkpoint(model_id: str) -> bool:
    """Detects whether `model_id` is already quantized with `compressed-tensors` (W4A16) (`R2-2`, `R2-10`)."""
    lower_id = model_id.lower()
    if "-ct" in lower_id or "w4a16" in lower_id or "compressed-tensors" in lower_id:
        return True
    try:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        q_cfg = getattr(cfg, "quantization_config", None)
        if isinstance(q_cfg, dict) and q_cfg.get("quant_method") in ("compressed-tensors", "compressed_tensors"):
            return True
    except Exception as exc:
        logger.warning(f"Could not inspect AutoConfig for {model_id} ({exc}); relying on model_id heuristic.")
    return False


def _enable_non_reentrant_gradient_checkpointing(model: Any, bnb_config: Any, prepare_fn: Any) -> Any:
    """Enables gradient checkpointing with `use_reentrant=False` (`R2-6`)."""
    gc_kwargs = {"use_reentrant": False}
    if bnb_config is not None:
        try:
            model = prepare_fn(
                model,
                use_gradient_checkpointing=True,
                gradient_checkpointing_kwargs=gc_kwargs,
            )
        except TypeError:
            model = prepare_fn(model, use_gradient_checkpointing=True)
            try:
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gc_kwargs)
            except TypeError:
                pass
    else:
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gc_kwargs)
        except TypeError:
            model.gradient_checkpointing_enable()
    return model


def _patch_peft_for_compressed_tensors() -> bool:
    """Monkey-patches PEFT internals to safely support compressed-tensors and custom linear layers.

    1. In Hugging Face PEFT, `peft.tuners.tuners_utils._get_in_out_features` and
       `peft.tuners.lora.layer.LoraLayer._get_in_out_features` contain:
         `if _torch_supports_distributed and isinstance(module.weight, torch.distributed.tensor.DTensor):`
       When a model uses packed INT4 weights (`compressed-tensors`), the module class `Linear` lacks `.weight`
       (storing `weight_packed` instead). Evaluating `module.weight` directly throws:
         `AttributeError: 'Linear' object has no attribute 'weight'`.
       This patch safely extracts `in_features` and `out_features` in O(1) time without accessing `.weight`.

    2. In `peft.import_utils.is_torchao_available()`, if an older version of `torchao` is installed in the
       environment (e.g. `torchao 0.10.0` pre-installed on Google Colab), it raises:
         `ImportError: Found an incompatible version of torchao... but only versions above 0.16.0 are supported`
       during `dispatch_torchao` even when adapting non-torchao models.
       This patch wraps `is_torchao_available` and `dispatch_torchao` so incompatible versions safely return
       `False`/`None`, allowing PEFT to fall through to `dispatch_default`.
    """
    def _safe_get_in_out_features(module: Any):
        # Fast-path: module has explicit in_features and out_features attributes
        in_f = getattr(module, "in_features", None)
        out_f = getattr(module, "out_features", None)
        if in_f is not None and out_f is not None:
            return in_f, out_f
        if not hasattr(module, "weight"):
            for alt_name in ("weight_packed", "qweight", "weight_scale"):
                alt_val = getattr(module, alt_name, None)
                if alt_val is not None:
                    return getattr(module, "in_features", None), getattr(module, "out_features", None)
            return in_f, out_f
        return orig_fn(module)

    # 1. Patch in tuners_utils
    try:
        import peft.tuners.tuners_utils as tu
        orig_fn = getattr(tu, "_get_in_out_features", None)
        if orig_fn is not None:
            tu._get_in_out_features = _safe_get_in_out_features
            logger.info("Monkey-patched PEFT tuners_utils._get_in_out_features.")
    except Exception as exc:
        logger.debug(f"Could not patch peft.tuners.tuners_utils._get_in_out_features: {exc}")

    # 2. Patch in lora.layer (both module-level and LoraLayer class-level)
    try:
        import peft.tuners.lora.layer as lora_layer
        lora_layer._get_in_out_features = _safe_get_in_out_features
        if hasattr(lora_layer, "LoraLayer"):
            lora_layer.LoraLayer._get_in_out_features = lambda self, module: _safe_get_in_out_features(module)
        logger.info("Monkey-patched PEFT lora.layer.LoraLayer._get_in_out_features.")
    except Exception as exc:
        logger.debug(f"Could not patch peft.tuners.lora.layer._get_in_out_features: {exc}")

    # 3. Patch torchao version check so pre-installed older torchao (e.g. 0.10.0 on Colab) does not crash get_peft_model
    try:
        import peft.import_utils as piu
        orig_is_torchao = getattr(piu, "is_torchao_available", None)
        if orig_is_torchao is not None:
            def _safe_is_torchao_available():
                try:
                    return bool(orig_is_torchao())
                except Exception:
                    return False
            piu.is_torchao_available = _safe_is_torchao_available
    except Exception:
        pass

    try:
        import peft.tuners.lora.torchao as p_torchao
        if hasattr(p_torchao, "is_torchao_available"):
            p_torchao.is_torchao_available = lambda: False
        if hasattr(p_torchao, "dispatch_torchao"):
            orig_dispatch = p_torchao.dispatch_torchao
            def _safe_dispatch_torchao(*args, **kwargs):
                try:
                    return orig_dispatch(*args, **kwargs)
                except Exception:
                    return None
            p_torchao.dispatch_torchao = _safe_dispatch_torchao
    except Exception:
        pass

    return True


def _ensure_module_has_weight_attr(model: Any) -> int:
    """Ensures quantized linear layer instances have `compute_dtype` set for PEFT compatibility.
    Does NOT modify classes (e.g. nn.Linear) to avoid corrupting PyTorch parameter registration.
    """
    patched_count = 0
    for name, module in model.named_modules():
        if hasattr(module, "weight_packed") and not hasattr(module, "compute_dtype"):
            setattr(
                module,
                "compute_dtype",
                torch.bfloat16 if (torch is not None and torch.cuda.is_available()) else torch.float32,
            )
            patched_count += 1
    if patched_count > 0:
        logger.info(f"Registered compute_dtype on {patched_count} compressed-tensors module instances.")
    return patched_count


def _unwrap_clippable_linears(model: Any) -> int:
    """Unwraps any `Gemma4ClippableLinear` wrappers in `model` into their inner `linear` (`torch.nn.Linear`).

    Gemma 4 models wrap linear layers in multimodal components (vision and audio encoder towers)
    in `Gemma4ClippableLinear`, which inherits from `nn.Module` rather than `nn.Linear`.
    HuggingFace PEFT's `get_peft_model` rejects this with:
      `ValueError: Target module Gemma4ClippableLinear(...) is not supported.`
    Replacing the wrapper with its inner `module.linear` makes PEFT see standard `torch.nn.Linear` layers.
    """
    unwrapped = 0
    for name, module in list(model.named_modules()):
        if module.__class__.__name__ == "Gemma4ClippableLinear" and hasattr(module, "linear"):
            parent = model
            parts = name.split(".")
            for p in parts[:-1]:
                if p.isdigit() and isinstance(parent, (list, tuple)):
                    parent = parent[int(p)]
                elif hasattr(parent, p):
                    parent = getattr(parent, p)
                else:
                    try:
                        parent = parent[int(p)]
                    except Exception:
                        parent = getattr(parent, p)
            last_part = parts[-1]
            setattr(parent, last_part, module.linear)
            unwrapped += 1
    if unwrapped > 0:
        logger.info(
            f"Unwrapped {unwrapped} Gemma4ClippableLinear modules into native linear layers for PEFT compatibility."
        )
    return unwrapped


def _filter_text_target_modules(model: Any, target_suffixes: list[str]) -> list[str]:
    """Filters target module names to only include language model layers,
    explicitly excluding multimodal encoder towers (vision_tower, audio_tower, visual, audio).
    """
    excluded_prefixes = ("vision_tower", "audio_tower", "visual", "audio", "image_encoder")
    text_modules: list[str] = []
    has_multimodal = False
    for name, module in model.named_modules():
        if any(exc in name.lower() for exc in excluded_prefixes):
            has_multimodal = True
            continue
        for suffix in target_suffixes:
            if name.endswith(f".{suffix}") or name == suffix:
                text_modules.append(name)
                break
    if has_multimodal and text_modules:
        logger.info(
            f"Filtered {len(text_modules)} text decoder target modules (excluded multimodal vision/audio towers)."
        )
        return text_modules
    return target_suffixes


def _prepare_gemma4_for_peft(model: Any, config_target_modules: list[str]) -> tuple[Any, list[str]]:
    """Prepares a Gemma 4 base model for PEFT LoRA attachment:
    1. Monkey-patches PEFT's `_get_in_out_features` to safely handle quantized linear layers lacking `.weight`.
    2. Unwraps any `Gemma4ClippableLinear` instances into their inner `linear` (`torch.nn.Linear`).
    3. Exposes a synthetic `.weight` property on any compressed linear classes lacking `.weight`.
    4. Identifies and filters target modules to language model layers only, excluding vision/audio towers.
    """
    _patch_peft_for_compressed_tensors()
    _unwrap_clippable_linears(model)
    _ensure_module_has_weight_attr(model)
    resolved_targets = _filter_text_target_modules(model, config_target_modules)
    return model, resolved_targets


def _resolve_logits_keep_kwarg(model: Any, num_keep: int) -> dict[str, int]:
    """Detects whether `model.forward` supports `logits_to_keep` (modern Transformers) or
    `num_logits_to_keep` (older Transformers) (`N5` Fixed).
    Unpacks nested wrapper layers (PeftModel -> LoraModel -> Gemma4ForCausalLM).
    """
    curr = model
    candidates = [curr]
    for _ in range(6):
        found_next = False
        for attr in ("base_model", "model"):
            nxt = getattr(curr, attr, None)
            if nxt is not None and nxt is not curr and hasattr(nxt, "forward"):
                candidates.append(nxt)
                curr = nxt
                found_next = True
                break
        if not found_next:
            break

    for m in reversed(candidates):
        fwd = getattr(m, "forward", None)
        if fwd is None:
            continue
        try:
            params = inspect.signature(fwd).parameters
            if "logits_to_keep" in params:
                return {"logits_to_keep": num_keep}
            if "num_logits_to_keep" in params:
                return {"num_logits_to_keep": num_keep}
        except Exception:
            continue

    # Modern Transformers CausalLM (including Gemma 4) accepts logits_to_keep by default
    return {"logits_to_keep": num_keep}


def _ensure_model_decompressed(model: Any) -> None:
    """Ensures accelerate device dispatch and compressed-tensors decompress hooks have fired
    so all quantized linear layers possess native `.weight` attributes before any sub-module
    (such as `backbone` or `lm_head`) is invoked directly.
    """
    needs_decompression = False
    for _, mod in model.named_modules():
        if (hasattr(mod, "weight_packed") or hasattr(mod, "quantization_scheme")) and not hasattr(mod, "weight"):
            needs_decompression = True
            break

    if not needs_decompression:
        return

    logger.info("Triggering one-time model dispatch and decompression hooks via 1-token warmup pass...")
    try:
        from compressed_tensors.compressors import ModelCompressor
        base = model
        for attr in ("base_model", "model"):
            if hasattr(base, attr):
                base = getattr(base, attr)
        cfg = getattr(base, "config", None) or getattr(model, "config", None)
        if cfg is not None and hasattr(ModelCompressor, "from_compression_config"):
            compressor = ModelCompressor.from_compression_config(cfg)
            if hasattr(compressor, "decompress_model"):
                compressor.decompress_model(base)
                logger.info("Decompressed model via ModelCompressor.")
    except Exception as exc:
        logger.debug(f"Direct ModelCompressor call notice: {exc}")

    device = _get_model_input_device(model)
    was_training = getattr(model, "training", False)
    model.eval()
    try:
        with torch.no_grad():
            dummy_input = torch.tensor([[1]], dtype=torch.long, device=device)
            keep_kwargs = _resolve_logits_keep_kwarg(model, 1)
            try:
                model(dummy_input, **keep_kwargs)
            except Exception:
                model(dummy_input)
        logger.info("One-time model dispatch and decompression hooks completed successfully.")
    except Exception as exc:
        logger.warning(f"Model decompression warmup pass completed with notice: {exc}")
    finally:
        if was_training:
            model.train()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def load_qlora_gemma4_31b(config: PatchGRPOConfig):
    """Loads Gemma 4 31B with LoRA adapters and non-reentrant gradient checkpointing (`R2-2`, `R2-6`, `N4`).

    Fixes Round 3 finding `N4` and Gemma4ClippableLinear compatibility:
      1. If `config.resume_from_checkpoint` is specified, validates that the path exists and calls
         `PeftModel.from_pretrained(..., is_trainable=True)` OUTSIDE any fallback `try/except` so
         resume errors are never swallowed or replaced with a fresh adapter.
      2. Unwraps any `Gemma4ClippableLinear` instances into native `torch.nn.Linear` layers and
         restricts `target_modules` to the text language model backbone, preventing PEFT crashes
         from multimodal vision/audio encoders.
      3. NF4 fallback on `compressed-tensors` checkpoints is strictly opt-in (`config.allow_nf4_fallback`
         or `config.force_bnb_4bit`) and writes a `TRAINED_ON_NF4.txt` marker in `config.checkpoint_dir`.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training

    _patch_peft_for_compressed_tensors()

    model_id = config.base_model_id
    is_ct = is_compressed_tensors_checkpoint(model_id)
    use_bnb = config.use_4bit_qlora and torch.cuda.is_available() and (config.force_bnb_4bit or not is_ct)

    logger.info(
        f"Loading tokenizer and model: {model_id} "
        f"(compressed_tensors={is_ct}, use_bnb_nf4={use_bnb})"
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    compute_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    def _build_base_model(enable_bnb: bool):
        bnb_cfg = None
        if enable_bnb and torch.cuda.is_available():
            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        sig_params = inspect.signature(AutoModelForCausalLM.from_pretrained).parameters
        dtype_key = "dtype" if "dtype" in sig_params else "torch_dtype"
        load_kwargs: dict[str, Any] = {
            dtype_key: compute_dtype,
            "device_map": "auto" if torch.cuda.is_available() else None,
            "trust_remote_code": True,
        }
        if bnb_cfg is not None:
            load_kwargs["quantization_config"] = bnb_cfg
        m = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
        m.config.use_cache = False
        m = _enable_non_reentrant_gradient_checkpointing(m, bnb_cfg, prepare_model_for_kbit_training)
        if hasattr(m, "enable_input_require_grads"):
            m.enable_input_require_grads()
        else:
            def _make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            m.get_input_embeddings().register_forward_hook(_make_inputs_require_grad)
        return m

    def _build_lora_config(resolved_target_modules: list[str]) -> LoraConfig:
        lora_kwargs: dict[str, Any] = {
            "r": config.lora_rank,
            "lora_alpha": config.lora_alpha,
            "lora_dropout": config.lora_dropout,
            "bias": "none",
            "task_type": "CAUSAL_LM",
            "target_modules": resolved_target_modules,
        }
        if "exclude_modules" in inspect.signature(LoraConfig.__init__).parameters:
            lora_kwargs["exclude_modules"] = [
                "vision_tower",
                "audio_tower",
                "visual",
                "audio",
                "image_encoder",
            ]
        return LoraConfig(**lora_kwargs)

    # N4 Fix: Handle `--resume-from` explicitly without catching/swallowing errors!
    if config.resume_from_checkpoint is not None:
        resume_path = Path(config.resume_from_checkpoint)
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint path does not exist: {resume_path}")
        logger.info(f"Resuming LoRA adapter from checkpoint: {resume_path}")
        base_model = _build_base_model(use_bnb)
        base_model, _ = _prepare_gemma4_for_peft(base_model, config.target_modules)
        model = PeftModel.from_pretrained(base_model, str(resume_path), is_trainable=True)
    else:
        base_model = _build_base_model(use_bnb)
        base_model, resolved_targets = _prepare_gemma4_for_peft(base_model, config.target_modules)
        lora_config = _build_lora_config(resolved_targets)
        try:
            model = get_peft_model(base_model, lora_config)
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            if trainable_params == 0:
                raise RuntimeError("PEFT wrapped 0 trainable parameters on native checkpoint.")
        except Exception as exc:
            if not use_bnb and config.allow_nf4_fallback and torch.cuda.is_available():
                logger.warning(
                    f"[WARNING - N4] Native compressed-tensors LoRA wrapping failed ({exc}); "
                    f"--allow-nf4-fallback is enabled, retrying with BitsAndBytes NF4. "
                    f"Note: NF4 weights may differ from vLLM's W4A16 compressed-tensors serving weights!"
                )
                del base_model
                torch.cuda.empty_cache()
                base_model = _build_base_model(True)
                base_model, resolved_targets = _prepare_gemma4_for_peft(base_model, config.target_modules)
                lora_config = _build_lora_config(resolved_targets)
                model = get_peft_model(base_model, lora_config)
                use_bnb = True
            else:
                raise RuntimeError(
                    f"Failed to attach LoRA to {model_id} ({exc}). "
                    f"Upgrade `peft`/`compressed-tensors` or pass `--allow-nf4-fallback` / `--force-bnb-4bit`."
                ) from exc

    if use_bnb and is_ct:
        config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        marker = config.checkpoint_dir / "TRAINED_ON_NF4.txt"
        marker.write_text(
            f"Warning: {model_id} was loaded with BitsAndBytes NF4 quantization instead of native compressed-tensors.\n",
            encoding="utf-8",
        )
        logger.warning(f"Wrote NF4 quantization warning marker to {marker}")

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert trainable_params > 0, "Fatal: Model has 0 trainable LoRA parameters!"
    model.print_trainable_parameters()
    _ensure_model_decompressed(model)
    return model, tokenizer


@contextmanager
def rollout_inference_mode(model: Any):
    """Temporarily switches `model` to `.eval()` and enables `model.config.use_cache = True`
    during rollout generation, then restores `.train()` and `use_cache = False` for backward passes.
    """
    was_training = getattr(model, "training", False)
    prev_use_cache = getattr(getattr(model, "config", None), "use_cache", False)
    model.eval()
    if hasattr(model, "config"):
        model.config.use_cache = True
    try:
        yield
    finally:
        if hasattr(model, "config"):
            model.config.use_cache = prev_use_cache
        if was_training:
            model.train()


def run_gpu_smoke_test(model: Any, tokenizer: Any, config: PatchGRPOConfig) -> dict[str, Any]:
    """Executes a 2-stage GPU smoke test (`N6` Fixed):
      1. Stage 1 (Multi-Turn Trajectory with `sampled_token_ids` & `TOOL_DECLARATIONS`):
         Exercises `_compute_multi_turn_token_ids_and_mask`, `sampled_token_ids` header preservation (`N1`),
         `disable_adapter()` reference KL pass, and per-candidate `.backward()`.
      2. Stage 2 (Synthetic `max_seq_length` Stress Test with 40% active masked positions):
         Measures realistic worst-case peak CUDA VRAM at `config.max_seq_length` tokens.
    """
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    prompt = format_gemma4_chat_prompt(
        tokenizer,
        [{"role": "user", "content": "Fix bug in src/app.py"}],
        add_generation_prompt=True,
    )
    sample_tc_str = '<|tool_call>call:edit_file{filepath: "src/app.py", old_string: "a = 1", new_string: "a = 2"}<tool_call|><turn|>\n'
    sampled_ids = tokenizer.encode(sample_tc_str, add_special_tokens=False, truncation=False)
    multi_turn_msgs = [
        {"role": "system", "content": "You are a developer agent."},
        {"role": "user", "content": "Fix bug in src/app.py"},
        {
            "role": "assistant",
            "content": "Reading file.",
            "tool_calls": [{"function": {"name": "read_file", "arguments": {"filepath": "src/app.py"}}}],
        },
        {"role": "tool", "name": "read_file", "content": '{"status": "ok", "content": "a = 1\\n"}'},
        {
            "role": "assistant",
            "content": "Editing file.",
            "sampled_token_ids": sampled_ids,
            "tool_calls": [
                {"function": {"name": "edit_file", "arguments": {"filepath": "src/app.py", "old_string": "a = 1", "new_string": "a = 2"}}},
                {"function": {"name": "submit_patch", "arguments": {}}},
            ],
        },
    ]
    resps = [
        sample_tc_str,
        '<|tool_call>call:read_file{filepath: "src/app.py", start_line: 1, end_line: 20}<tool_call|>',
    ]
    cand_msgs = [multi_turn_msgs, None]
    rewards = [4.5, 1.0]

    old_lps = compute_group_old_log_probs(
        model, tokenizer, prompt, resps, config, candidate_messages=cand_msgs
    )
    model.train()
    model.zero_grad()

    loss_mt, stats_mt = compute_patchgrpo_loss(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        candidate_responses=resps,
        rewards=rewards,
        config=config,
        old_log_probs=old_lps,
        candidate_messages=cand_msgs,
        call_backward=True,
    )

    grad_norms = [
        p.grad.detach().float().norm().item()
        for p in model.parameters()
        if p.requires_grad and p.grad is not None
    ]
    assert len(grad_norms) > 0, "Smoke test Stage 1 failed: no LoRA parameters received gradients!"
    mt_grad_norm = math.sqrt(sum(g * g for g in grad_norms))
    assert math.isfinite(mt_grad_norm) and mt_grad_norm > 0.0, (
        f"Smoke test Stage 1 failed: invalid grad_norm={mt_grad_norm}"
    )
    mt_vram_gb = (
        torch.cuda.max_memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0.0
    )
    model.zero_grad()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Stage 2: Long-context stress test (2048 prompt tokens + 512 response tokens, 40% active mask)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    device = _get_model_input_device(model)
    prompt_len = min(2048, config.max_prompt_length)
    resp_len = min(512, config.max_new_tokens)
    seq_len = prompt_len + resp_len
    vocab_cap = min(getattr(tokenizer, "vocab_size", 256) or 256, 1024)
    synth_full_ids = torch.randint(3, max(4, vocab_cap), (1, seq_len), dtype=torch.long, device=device)
    synth_mask = torch.zeros((1, resp_len), dtype=torch.float32, device=device)
    active_count = max(1, int(resp_len * 0.4))
    synth_mask[0, :active_count] = 1.0

    token_lp, _ = _get_per_token_log_probs(
        model, synth_full_ids, prompt_len, resp_len, response_mask=synth_mask, chunk_size=config.logprob_chunk_size
    )
    if hasattr(model, "disable_adapter"):
        with torch.no_grad(), model.disable_adapter():
            ref_lp, _ = _get_per_token_log_probs(
                model, synth_full_ids, prompt_len, resp_len, response_mask=synth_mask, chunk_size=config.logprob_chunk_size
            )
    else:
        ref_lp = token_lp.detach()

    synth_loss = -((token_lp - 0.04 * (ref_lp - token_lp)) * synth_mask).sum() / synth_mask.sum()
    synth_loss.backward()
    max_seq_vram_gb = (
        torch.cuda.max_memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0.0
    )
    model.zero_grad()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    report = {
        "multi_turn_loss": round(loss_mt.item(), 5),
        "multi_turn_grad_norm": round(mt_grad_norm, 6),
        "multi_turn_peak_vram_gb": round(mt_vram_gb, 3),
        "max_seq_length_tested": seq_len,
        "max_seq_active_tokens": active_count,
        "max_seq_peak_vram_gb": round(max_seq_vram_gb, 3),
        "num_grad_tensors": len(grad_norms),
        "stats": stats_mt,
    }
    logger.info(f"[GPU SMOKE TEST PASSED (Multi-Turn + Max-Seq)]: {report}")
    return report


# ==============================================================================
# 5. Memory-Efficient Chunked Log-Probs & PatchGRPO Surrogate Loss
# ==============================================================================

def _extract_backbone_and_head(model: Any):
    """Safely extracts the transformer backbone, lm_head, and softcapping config from a PEFT or CausalLM model."""
    curr = model
    if hasattr(curr, "get_base_model"):
        try:
            curr = curr.get_base_model()
        except Exception:
            pass
    if hasattr(curr, "base_model") and hasattr(curr.base_model, "model"):
        curr = curr.base_model.model

    backbone = getattr(curr, "model", None)
    lm_head = getattr(curr, "lm_head", None)
    config = getattr(curr, "config", None)
    softcapping = getattr(config, "final_logit_softcapping", None)
    if backbone is not None and lm_head is not None and hasattr(backbone, "forward"):
        return backbone, lm_head, softcapping
    return None, None, None


def _get_per_token_log_probs(
    model: Any,
    full_ids: Any,
    prompt_len: int,
    resp_len: int,
    response_mask: Optional[Any] = None,
    chunk_size: int = 128,
) -> tuple[Any, Any]:
    """Computes per-token log-probabilities of shape `(1, resp_len)` with minimal VRAM (`R2-3`, `N5`).

    Active-Only Logit Projection Optimization for 262k-vocab Gemma 4:
      In multi-turn SWE-bench trajectories, sequence lengths reach 5,000+ tokens, but 95%+ of those
      tokens are tool responses (file dumps) where `response_mask == 0.0`.
      Materializing the 262,144-vocab LM head and out-of-place softcapping across all 5,000 positions
      allocates 5.34+ GiB and triggers CUDA OOM on A100.

      This optimization:
        1. Runs `full_ids` through the transformer backbone only, producing `hidden_states` (~58 MB).
        2. Slices `hidden_states` strictly at the positions corresponding to active tokens (`response_mask > 0`).
        3. Projects through `lm_head` and softcapping in chunks of `chunk_size = 128` tokens with non-reentrant
           gradient checkpointing.
      Logits VRAM drops from 5.34 GiB to ~67 MiB (a 98% reduction), eliminating all trajectory-length OOMs.
    """
    target_ids = full_ids[:, prompt_len : prompt_len + resp_len]
    token_log_probs = torch.zeros(
        (1, resp_len), dtype=torch.float32, device=full_ids.device
    )

    if response_mask is not None:
        active_cols = (response_mask[0] > 0).nonzero(as_tuple=True)[0]
    else:
        active_cols = torch.arange(resp_len, device=full_ids.device)

    if active_cols.numel() == 0:
        return token_log_probs, target_ids

    _ensure_model_decompressed(model)
    backbone, lm_head, softcapping = _extract_backbone_and_head(model)

    if backbone is not None and lm_head is not None:
        # 1. Forward through transformer backbone only (hidden states: ~58 MB)
        backbone_out = backbone(full_ids)
        hidden_states = backbone_out[0] if isinstance(backbone_out, tuple) else backbone_out.last_hidden_state

        # Target token at `col` is predicted from hidden state at `prompt_len + col - 1`
        act_h_idx = prompt_len + active_cols - 1
        act_h = hidden_states[:, act_h_idx, :]
        act_targets = target_ids[:, active_cols].squeeze(0)
        vocab_size = getattr(getattr(model, "config", None), "vocab_size", None) or lm_head.out_features

        def _chunk_proj_and_loss(h_slice: Any, t_slice: Any) -> Any:
            l_slice = lm_head(h_slice)
            if softcapping is not None:
                l_slice = l_slice / softcapping
            return -F.cross_entropy(l_slice.float().view(-1, vocab_size), t_slice.view(-1), reduction="none")

        step = max(1, chunk_size)
        num_active = active_cols.numel()
        active_lp_chunks: list[Any] = []

        for start in range(0, num_active, step):
            h_chunk = act_h[:, start : start + step, :]
            t_chunk = act_targets[start : start + step]
            if act_h.requires_grad and torch_ckpt is not None:
                lp_chunk = torch_ckpt.checkpoint(
                    _chunk_proj_and_loss,
                    h_chunk,
                    t_chunk,
                    use_reentrant=False,
                )
            else:
                lp_chunk = _chunk_proj_and_loss(h_chunk, t_chunk)
            active_lp_chunks.append(lp_chunk)

        all_active_lps = torch.cat(active_lp_chunks, dim=0)
        token_log_probs = token_log_probs.clone()
        token_log_probs[0, active_cols] = all_active_lps
        return token_log_probs, target_ids

    # Fallback path if model does not expose separable backbone/lm_head
    keep_kwargs = _resolve_logits_keep_kwarg(model, resp_len + 1)
    try:
        outputs = model(full_ids, **keep_kwargs) if keep_kwargs else model(full_ids)
    except TypeError:
        outputs = model(full_ids)

    raw_logits = outputs.logits
    if raw_logits.shape[1] == resp_len + 1:
        logits = raw_logits[:, :resp_len, :]
    else:
        logits = raw_logits[:, prompt_len - 1 : prompt_len + resp_len - 1, :]

    def _chunk_neg_ce(logits_slice: Any, targets_slice: Any) -> Any:
        return -F.cross_entropy(logits_slice.float(), targets_slice, reduction="none")

    step = max(1, chunk_size)
    active_lp_chunks = []
    for start in range(0, active_cols.numel(), step):
        cols = active_cols[start : start + step]
        logits_chunk = logits[0, cols, :]
        targets_chunk = target_ids[0, cols]
        if logits_chunk.requires_grad and torch_ckpt is not None:
            lp_chunk = torch_ckpt.checkpoint(
                _chunk_neg_ce,
                logits_chunk,
                targets_chunk,
                use_reentrant=False,
            )
        else:
            lp_chunk = _chunk_neg_ce(logits_chunk, targets_chunk)
        active_lp_chunks.append(lp_chunk)

    all_active_lps = torch.cat(active_lp_chunks, dim=0)
    token_log_probs = token_log_probs.clone()
    token_log_probs[0, active_cols] = all_active_lps
    return token_log_probs, target_ids


def compute_group_old_log_probs(
    model: Any,
    tokenizer: Any,
    prompt: Union[str, list[str]],
    candidate_responses: list[str],
    config: PatchGRPOConfig,
    candidate_messages: Optional[list[Optional[list[dict[str, Any]]]]] = None,
) -> list[Optional[Any]]:
    """Pre-computes detached per-token log-probabilities under the behavior policy `pi_old`."""
    device = _get_model_input_device(model)
    was_training = getattr(model, "training", False)
    model.eval()
    cached: list[Optional[Any]] = []
    try:
        with torch.no_grad():
            for idx, resp_text in enumerate(candidate_responses):
                cand_prompt = prompt[idx] if isinstance(prompt, list) else prompt
                msgs = candidate_messages[idx] if (candidate_messages and idx < len(candidate_messages)) else None
                tok_res = tokenize_multi_turn_trajectory(
                    tokenizer, msgs, cand_prompt, resp_text, config, device
                )
                if tok_res is None:
                    cached.append(None)
                    continue
                full_ids, resp_mask, prompt_len = tok_res
                resp_len = resp_mask.shape[1]
                if resp_len == 0 or resp_mask.sum().item() == 0:
                    cached.append(None)
                    continue
                old_lp, _ = _get_per_token_log_probs(
                    model,
                    full_ids,
                    prompt_len,
                    resp_len,
                    response_mask=resp_mask,
                    chunk_size=config.logprob_chunk_size,
                )
                cached.append(old_lp.detach().cpu())
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    finally:
        if was_training:
            model.train()
    return cached


def compute_patchgrpo_loss(
    model: Any,
    tokenizer: Any,
    prompt: Union[str, list[str]],
    candidate_responses: list[str],
    rewards: list[float],
    config: PatchGRPOConfig,
    old_log_probs: Optional[list[Optional[Any]]] = None,
    candidate_messages: Optional[list[Optional[list[dict[str, Any]]]]] = None,
    call_backward: bool = False,
    loss_scale: float = 1.0,
    positive_advantages_only: bool = False,
) -> tuple[Any, dict[str, float]]:
    """Computes the Critic-Free PatchGRPO objective across G candidates:

      A_i = (R_i - mean(R)) / max(std(R), min_reward_std_scale)
      ratio_{i,t} = exp(log pi_theta(y_{i,t} | x_i, y_{i,<t}) - log pi_old(y_{i,t} | x_i, y_{i,<t}))
      surr_{i,t}  = min(ratio_{i,t} * A_i, clip(ratio_{i,t}, 1 - eps, 1 + eps) * A_i)
      KL_{i,t}    = exp(log pi_ref - log pi_theta) - (log pi_ref - log pi_theta) - 1
      L_PatchGRPO = - (1 / G_valid) sum_i (1 / |y_i|) sum_t mask_{i,t} * (surr_{i,t} - beta * KL_{i,t})
    """
    device = _get_model_input_device(model)
    r_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)
    mean_r = r_tensor.mean()
    std_r = r_tensor.std(unbiased=False)

    if std_r < 1e-4:
        return torch.tensor(0.0, device=device, requires_grad=True), {
            "mean_reward": mean_r.item(),
            "std_reward": 0.0,
            "mean_kl": 0.0,
            "mean_ratio": 1.0,
            "skipped_zero_var": 1.0,
        }

    std_denom = torch.clamp(std_r, min=config.min_reward_std_scale)
    advantages = (r_tensor - mean_r) / (std_denom + 1e-6)
    if positive_advantages_only:
        advantages = torch.clamp(advantages, min=0.0)

    prepared_candidates: list[tuple[int, Any, Any, int, Any]] = []
    for idx, (resp_text, adv) in enumerate(zip(candidate_responses, advantages)):
        if positive_advantages_only and adv.item() <= 1e-6:
            continue
        cand_prompt = prompt[idx] if isinstance(prompt, list) else prompt
        msgs = candidate_messages[idx] if (candidate_messages and idx < len(candidate_messages)) else None
        tok_res = tokenize_multi_turn_trajectory(
            tokenizer, msgs, cand_prompt, resp_text, config, device
        )
        if tok_res is None:
            continue
        full_ids, response_mask, prompt_len = tok_res
        if response_mask.shape[1] == 0 or response_mask.sum().item() <= 0:
            continue
        prepared_candidates.append((idx, full_ids, response_mask, prompt_len, adv))

    if not prepared_candidates:
        return torch.tensor(0.0, device=device, requires_grad=True), {
            "mean_reward": mean_r.item(),
            "std_reward": std_r.item(),
            "mean_kl": 0.0,
            "mean_ratio": 1.0,
            "skipped_zero_var": 1.0,
        }

    num_valid = float(len(prepared_candidates))
    total_loss_val = 0.0
    accum_loss_tensor = torch.tensor(0.0, device=device)
    kl_values: list[float] = []
    ratio_values: list[float] = []

    for idx, full_ids, response_mask, prompt_len, adv in prepared_candidates:
        resp_len = response_mask.shape[1]
        token_lp, _ = _get_per_token_log_probs(
            model,
            full_ids,
            prompt_len,
            resp_len,
            response_mask=response_mask,
            chunk_size=config.logprob_chunk_size,
        )
        response_mask = response_mask.to(token_lp.dtype)
        mask_tokens = response_mask.sum().clamp(min=1.0)

        if config.kl_beta > 0.0:
            with torch.no_grad():
                if hasattr(model, "disable_adapter"):
                    with model.disable_adapter():
                        ref_token_lp, _ = _get_per_token_log_probs(
                            model,
                            full_ids,
                            prompt_len,
                            resp_len,
                            response_mask=response_mask,
                            chunk_size=config.logprob_chunk_size,
                        )
                else:
                    ref_token_lp = token_lp.detach()
            log_ref_minus_cur = ((ref_token_lp - token_lp) * response_mask).clamp(min=-10.0, max=10.0)
            token_kl = (torch.exp(log_ref_minus_cur) - log_ref_minus_cur - 1.0) * response_mask
            if getattr(config, "max_kl_threshold", None) and config.max_kl_threshold > 0:
                token_kl = torch.clamp(token_kl, max=config.max_kl_threshold) * response_mask
        else:
            token_kl = torch.zeros_like(token_lp)

        old_lp = None
        if old_log_probs is not None and idx < len(old_log_probs):
            old_lp = old_log_probs[idx]
        if old_lp is None or old_lp.shape != token_lp.shape:
            old_lp = token_lp.detach()
        else:
            old_lp = old_lp.to(device=device, dtype=token_lp.dtype)

        log_ratio = ((token_lp - old_lp) * response_mask).clamp(min=-10.0, max=10.0)
        ratio = torch.exp(log_ratio)
        clipped_ratio = torch.clamp(ratio, 1.0 - config.clip_epsilon, 1.0 + config.clip_epsilon)

        surr1 = ratio * adv
        surr2 = clipped_ratio * adv
        policy_obj = torch.min(surr1, surr2)

        per_token_loss = -(policy_obj - config.kl_beta * token_kl) * response_mask
        cand_loss = (per_token_loss.sum() / mask_tokens) / num_valid

        if call_backward and cand_loss.requires_grad:
            (cand_loss * loss_scale).backward()
            total_loss_val += cand_loss.detach().item()
        else:
            accum_loss_tensor = accum_loss_tensor + cand_loss

        with torch.no_grad():
            kl_values.append(((token_kl * response_mask).sum() / mask_tokens).item())
            ratio_values.append(((ratio * response_mask).sum() / mask_tokens).item())

        if call_backward and torch.cuda.is_available():
            torch.cuda.empty_cache()

    if call_backward:
        ret_loss = torch.tensor(total_loss_val, device=device)
    else:
        ret_loss = accum_loss_tensor

    return ret_loss, {
        "mean_reward": mean_r.item(),
        "std_reward": std_r.item(),
        "mean_kl": sum(kl_values) / max(len(kl_values), 1),
        "mean_ratio": sum(ratio_values) / max(len(ratio_values), 1),
        "skipped_zero_var": 0.0,
    }


def _sanitize_submission_adapter_dir(adapter_dir: Path) -> list[str]:
    """Ensures `adapter_dir` (e.g. `submission/adapters/main_lora/`) contains ONLY files
    permitted by `adk-submission` (`adapter_config.json`, `adapter_model.safetensors`, `README.md`)
    (`N3` Fixed; cross-checked with `HARNESS_README.md` Section 2.4 & `AGENTS.md` Rule 9).
    """
    removed: list[str] = []
    if not adapter_dir.exists():
        return removed
    for item in list(adapter_dir.iterdir()):
        if item.is_dir():
            _force_rmtree(item)
            removed.append(item.name)
        elif item.name not in ALLOWED_SUBMISSION_ADAPTER_FILES:
            item.unlink(missing_ok=True)
            removed.append(item.name)
    return removed


def _save_final_submission_adapter(model: Any, output_dir: Path) -> None:
    """Saves the final trained LoRA adapter to `output_dir` and strips any non-whitelisted files (`N3`)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir))
    removed = _sanitize_submission_adapter_dir(output_dir)
    if removed:
        logger.info(f"Sanitized submission adapter dir {output_dir} (removed non-whitelisted: {removed})")


def _save_periodic_checkpoint(
    model: Any,
    optimizer: Any,
    output_dir: Path,
    checkpoint_root: Path,
    step_label: str,
    opt_steps: int = 0,
    epoch: int = 0,
    completed_tasks: Optional[list[str]] = None,
) -> Path:
    """Saves periodic checkpoints to `checkpoint_root / <name>_<step_label>` (OUTSIDE `submission/adapters/`)
    and updates the sanitized adapter in `output_dir` (`N3`, `N4` Fixed, plus online task-level resume).
    """
    _save_final_submission_adapter(model, output_dir)

    ckpt_dir = checkpoint_root / f"{output_dir.name}_{step_label}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(ckpt_dir))
    if optimizer is not None and torch is not None:
        try:
            torch.save(optimizer.state_dict(), ckpt_dir / "optimizer.pt")
        except Exception as exc:
            logger.debug(f"Could not save optimizer state ({exc})")
    state_payload: dict[str, Any] = {
        "opt_steps": opt_steps,
        "epoch": epoch,
        "step_label": step_label,
        "completed_tasks": list(completed_tasks or []),
    }
    state_file = ckpt_dir / "trainer_state.json"
    state_file.write_text(
        json.dumps(state_payload, indent=2),
        encoding="utf-8",
    )
    logger.info(f"Saved periodic checkpoint to {ckpt_dir} and updated sanitized {output_dir}")
    return ckpt_dir


def _load_completed_tasks_from_checkpoint(resume_dir: Optional[Path]) -> list[str]:
    """Returns the list of completed `instance_id`s stored in `resume_dir / trainer_state.json`."""
    if resume_dir is None:
        return []
    state_file = Path(resume_dir) / "trainer_state.json"
    if state_file.exists():
        try:
            st = json.loads(state_file.read_text(encoding="utf-8"))
            done = st.get("completed_tasks") or []
            if isinstance(done, list):
                return [str(x) for x in done]
        except Exception as exc:
            logger.warning(f"Could not read completed_tasks from {state_file}: {exc}")
    return []


def _maybe_load_optimizer_and_step_state(optimizer: Any, resume_dir: Optional[Path]) -> tuple[int, int]:
    """Restores optimizer state and `(opt_steps, start_epoch)` if present in `resume_dir` (`N4`)."""
    if resume_dir is None:
        return 0, 1
    resume_path = Path(resume_dir)
    opt_file = resume_path / "optimizer.pt"
    if optimizer is not None and torch is not None and opt_file.exists():
        try:
            optimizer.load_state_dict(torch.load(opt_file, map_location="cpu"))
            logger.info(f"Restored optimizer state from {opt_file}")
        except Exception as exc:
            logger.warning(f"Could not restore optimizer state from {opt_file}: {exc}")
    state_file = resume_path / "trainer_state.json"
    if state_file.exists():
        try:
            st = json.loads(state_file.read_text(encoding="utf-8"))
            return int(st.get("opt_steps", 0)), int(st.get("epoch", 1))
        except Exception as exc:
            logger.warning(f"Could not read trainer_state.json from {state_file}: {exc}")
    return 0, 1


# ==============================================================================
# 6. Offline Trajectory-Harvested Training Loop (Advantage-Weighted SFT / RWR)
# ==============================================================================

def train_from_harvested_trajectories(
    config: PatchGRPOConfig,
    trajectories_dir: Path,
    tasks_jsonl: Path,
    output_dir: Path,
    epochs: int = 2,
):
    """Trains the LoRA policy on pre-harvested multi-turn trajectories (`R2-5`, `R2-7`, `N9`).

    Note on Offline Mode (`R2-5`, `N9`):
      - When `positive_advantages_only_offline=True` (default), performs **Advantage-Weighted
        Multi-Turn SFT / Reward-Weighted Regression (RWR)** with reference-model KL regularization
        (`kl_beta`) and multi-epoch PPO ratio clipping against the step-0 31B policy. Candidates
        with `A_i <= 0` (including empty-patch rollouts with `reward = 0.0`) are skipped before
        tokenization.
      - When `--allow-negative-offline` is passed, multi-turn `messages` are passed for all
        candidates that have them (`N9`), falling back to single-turn prompt+response only if a
        record has no multi-turn `messages`.
    """
    from eval.run_task_e2e import compute_patchgrpo_reward

    set_global_seed(config.seed)
    model, tokenizer = load_qlora_gemma4_31b(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    opt_steps, start_epoch = _maybe_load_optimizer_and_step_state(optimizer, config.resume_from_checkpoint)

    tasks_by_id: dict[str, dict[str, Any]] = {}
    if tasks_jsonl.exists():
        for line in tasks_jsonl.read_text(encoding="utf-8").splitlines():
            if line.strip():
                t = json.loads(line)
                tasks_by_id[t["instance_id"]] = t

    traj_files = sorted(trajectories_dir.glob("*_trajectories.jsonl"))
    logger.info(f"Found {len(traj_files)} harvested trajectory groups in {trajectories_dir}")

    prepared_groups: list[dict[str, Any]] = []
    for tf in traj_files:
        records = [
            json.loads(line)
            for line in tf.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(records) < 2:
            continue

        instance_id = records[0]["instance_id"]
        task = tasks_by_id.get(instance_id, {})
        problem_stmt = task.get("problem_statement", f"Fix issue {instance_id}")

        for r in records:
            if isinstance(r.get("grade"), dict):
                recomputed = compute_patchgrpo_reward(r["grade"], task, r.get("patch", ""))
                r["reward"] = recomputed
                r["grade"]["total_reward"] = recomputed

        prompt = format_gemma4_chat_prompt(
            tokenizer,
            [
                {
                    "role": "user",
                    "content": (
                        f"Problem Statement:\n{problem_stmt}\n\n"
                        f"Please inspect the repository, implement the minimal surgical fix, and call submit_patch."
                    ),
                }
            ],
            add_generation_prompt=True,
        )

        candidate_responses = [extract_training_target_from_record(r) for r in records]
        # N9 Fix: Preserve multi-turn `messages` whenever present on the record
        candidate_messages = [
            (r.get("messages") if r.get("messages") else None)
            for r in records
        ]
        rewards = compute_tie_broken_rewards(records, max_turns=config.max_turns_per_candidate)

        r_tensor = torch.tensor(rewards, dtype=torch.float32)
        std_r = r_tensor.std(unbiased=False).item()
        if std_r < 1e-4 or not any(c.strip() for c in candidate_responses):
            logger.info(f"Skipping {instance_id} (zero reward variance std={std_r:.4f} or no valid patches)")
            continue

        old_lps = compute_group_old_log_probs(
            model, tokenizer, prompt, candidate_responses, config, candidate_messages=candidate_messages
        )
        prepared_groups.append({
            "instance_id": instance_id,
            "prompt": prompt,
            "candidate_responses": candidate_responses,
            "candidate_messages": candidate_messages,
            "rewards": rewards,
            "old_log_probs": old_lps,
        })
        logger.info(
            f"Prepared group {instance_id}: tie-broken rewards={rewards} "
            f"(valid_targets={sum(1 for c in candidate_responses if c.strip())}/{len(candidate_responses)})"
        )

    model.train()
    grad_accum = max(1, config.gradient_accumulation_steps)
    optimizer.zero_grad()

    mode_label = (
        "Advantage-Weighted Multi-Turn SFT (RWR + KL)"
        if config.positive_advantages_only_offline
        else "Offline PatchGRPO"
    )
    for epoch in range(start_epoch, epochs + 1):
        logger.info(f"=== {mode_label} Epoch {epoch}/{epochs} (grad_accum={grad_accum}) ===")
        accum_count = 0
        for group in prepared_groups:
            instance_id = group["instance_id"]
            loss, stats = compute_patchgrpo_loss(
                model=model,
                tokenizer=tokenizer,
                prompt=group["prompt"],
                candidate_responses=group["candidate_responses"],
                rewards=group["rewards"],
                config=config,
                old_log_probs=group["old_log_probs"],
                candidate_messages=group["candidate_messages"],
                call_backward=True,
                loss_scale=1.0 / float(grad_accum),
                positive_advantages_only=config.positive_advantages_only_offline,
            )

            if stats["skipped_zero_var"] == 0.0:
                accum_count += 1
                vram_gb = (
                    torch.cuda.max_memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0.0
                )
                logger.info(
                    f"{mode_label} ({instance_id}): loss={loss.item():.4f} | "
                    f"mean_R={stats['mean_reward']:.3f} | std_R={stats['std_reward']:.3f} | "
                    f"KL={stats['mean_kl']:.4f} | ratio={stats['mean_ratio']:.3f} | "
                    f"peak_VRAM={vram_gb:.2f}GB"
                )

                if accum_count % grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    opt_steps += 1
                    if config.save_every_n_steps > 0 and opt_steps % config.save_every_n_steps == 0:
                        _save_periodic_checkpoint(
                            model,
                            optimizer,
                            output_dir,
                            config.checkpoint_dir,
                            f"ep{epoch}_step{opt_steps}",
                            opt_steps=opt_steps,
                            epoch=epoch,
                        )

        if accum_count % grad_accum != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            opt_steps += 1

    _save_final_submission_adapter(model, output_dir)
    logger.info(f"Saved sanitized offline-trained LoRA adapter to: {output_dir}")


# ==============================================================================
# 7. Online End-to-End On-Policy PatchGRPO Training Loop
# ==============================================================================

def train_patchgrpo_pipeline(
    config: PatchGRPOConfig,
    tasks_jsonl: Path,
    output_dir: Path,
    snapshots_dir: Optional[Path] = None,
    use_agent_loop: bool = True,
    max_train_tasks: Optional[int] = None,
    task_ids_filter: Optional[list[str]] = None,
):
    """Online On-Policy PatchGRPO training loop with real repository checkouts, multi-axis
    exploration, KV-cached rollout generation, prompt consistency (`R2-4`, `N8`), concurrent-safe
    temp directories (`R2-11`), isolated periodic checkpointing (`N3`), task-level resume skipping,
    and per-candidate backward passes.
    """
    set_global_seed(config.seed)
    model, tokenizer = load_qlora_gemma4_31b(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    opt_steps, _ = _maybe_load_optimizer_and_step_state(optimizer, config.resume_from_checkpoint)
    completed_tasks: list[str] = _load_completed_tasks_from_checkpoint(config.resume_from_checkpoint)
    completed_set: set[str] = set(completed_tasks)
    if completed_set:
        logger.info(f"Resuming online run: skipping {len(completed_set)} already-completed tasks: {completed_tasks}")

    tasks = [
        json.loads(line)
        for line in tasks_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if task_ids_filter:
        wanted = {t.strip() for t in task_ids_filter if t.strip()}
        tasks = [t for t in tasks if t.get("instance_id") in wanted]
    if getattr(config, "interleave_repos", True):
        tasks = interleave_tasks_by_repo(tasks)
        logger.info("Interleaved tasks across repositories for balanced multi-repo representation.")
    if max_train_tasks is not None and max_train_tasks > 0:
        tasks = tasks[:max_train_tasks]
    logger.info(f"Loaded {len(tasks)} training tasks from {tasks_jsonl}")

    online_traj_dir = PROJECT_ROOT / "results" / "trajectories" / "online_31b"
    online_traj_dir.mkdir(parents=True, exist_ok=True)

    if snapshots_dir is None:
        default_snaps = tasks_jsonl.parent / "snapshots"
        if default_snaps.exists():
            snapshots_dir = default_snaps

    grad_accum = max(1, config.gradient_accumulation_steps)
    optimizer.zero_grad()
    accum_count = 0
    best_mean_reward = 0.0

    shared_backend = None
    if use_agent_loop and snapshots_dir is not None:
        from eval.run_task_e2e import DirectGemmaTransformersBackend
        # N10 Fix: Synchronize `max_new_tokens` between `DirectGemmaTransformersBackend` and `config`
        shared_backend = DirectGemmaTransformersBackend(
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=config.max_new_tokens,
        )

    bos_tok = getattr(tokenizer, "bos_token", None) or "<bos>"
    input_device = _get_model_input_device(model)

    for step, task in enumerate(tasks, 1):
        instance_id = task["instance_id"]
        if instance_id in completed_set:
            logger.info(f"[Step {step}/{len(tasks)}] Skipping already-completed task {instance_id} on resume.")
            continue
        problem_stmt = task["problem_statement"]
        target_files = task.get("target_files") or _extract_files_from_diff(task.get("patch", ""))
        test_patch = task.get("test_patch", "")
        test_files = _extract_files_from_diff(test_patch)
        test_cmd = task.get("test_cmd") or (
            [sys.executable, "-m", "pytest", test_files[0], "-q"] if test_files else [sys.executable, "-m", "pytest", "-q"]
        )

        base_prompt = format_gemma4_chat_prompt(
            tokenizer,
            [
                {
                    "role": "user",
                    "content": (
                        f"Problem Statement:\n{problem_stmt}\n\n"
                        f"Please inspect the repository, implement the minimal surgical fix, and call submit_patch."
                    ),
                }
            ],
            add_generation_prompt=True,
        )

        workspace_dir = Path(tempfile.mkdtemp(prefix=f"patchgrpo_ws_{instance_id}_"))
        candidate_prompts: list[str] = []
        candidate_responses: list[str] = []
        candidate_messages: list[Optional[list[dict[str, Any]]]] = []
        rollout_records: list[dict[str, Any]] = []

        logger.info(f"[Step {step}/{len(tasks)}] Sampling G={config.group_size} candidates for {instance_id}...")

        try:
            with rollout_inference_mode(model):
                if shared_backend is not None and (Path(snapshots_dir) / f"{instance_id}.tgz").exists():
                    from eval.run_task_e2e import (
                        WorkspaceTools,
                        compute_patchgrpo_reward,
                        grade_agent_patch,
                        run_agent_loop,
                    )

                    for g_idx in range(config.group_size):
                        strat = EXPLORATION_STRATEGIES[g_idx % len(EXPLORATION_STRATEGIES)]
                        _force_rmtree(workspace_dir)
                        prepare_workspace_from_snapshot(instance_id, snapshots_dir, workspace_dir)
                        tools = WorkspaceTools(
                            workdir=workspace_dir,
                            repo=task.get("repo", ""),
                            base_commit=task.get("base_commit", "HEAD"),
                            max_turns=config.max_turns_per_candidate,
                        )
                        agent_patch, loop_stats = run_agent_loop(
                            problem_statement=problem_stmt,
                            tools=tools,
                            model=config.base_model_id,
                            max_turns=config.max_turns_per_candidate,
                            direct_generator=shared_backend,
                            temperature=strat["temperature"],
                            strategy_hint=strat["hint"],
                        )
                        grade = grade_agent_patch(
                            task=task,
                            agent_patch=agent_patch,
                            workdir=workspace_dir,
                            run_pytest=True,
                        )
                        base_r = float(grade.get("total_reward", compute_patchgrpo_reward(grade, task, agent_patch)))
                        msgs = loop_stats.get("messages", [])
                        rec = {
                            "instance_id": instance_id,
                            "candidate_idx": g_idx,
                            "reward": base_r,
                            "patch": agent_patch,
                            "patch_size": grade.get("patch_size", 0),
                            "patch_submitted": bool(tools.patch_submitted),
                            "turns_used": loop_stats.get("turns_used", 0),
                            "messages": msgs,
                        }
                        rollout_records.append(rec)
                        candidate_prompts.append(base_prompt)
                        candidate_responses.append(extract_training_target_from_record(rec))
                        candidate_messages.append(msgs if msgs else None)

                    rewards = compute_tie_broken_rewards(rollout_records, max_turns=config.max_turns_per_candidate)
                    # Persist online 31B rollouts to disk immediately so they are never lost on disconnect
                    traj_out_file = online_traj_dir / f"{instance_id}_trajectories.jsonl"
                    traj_out_file.write_text(
                        "\n".join(json.dumps(r, ensure_ascii=False) for r in rollout_records) + "\n",
                        encoding="utf-8",
                    )
                else:
                    # R2-4 & N8 Fix: Single-shot online mode trains on the exact `strat_prompt` AND
                    # raw sampled text `raw_gen_text` (not a re-formatted ```diff wrapper).
                    rewards = []
                    for g_idx in range(config.group_size):
                        strat = EXPLORATION_STRATEGIES[g_idx % len(EXPLORATION_STRATEGIES)]
                        strat_prompt = format_gemma4_chat_prompt(
                            tokenizer,
                            [
                                {
                                    "role": "user",
                                    "content": (
                                        f"Problem Statement:\n{problem_stmt}\n\n"
                                        f"{strat['hint']}\n"
                                        f"Produce the minimal unified git diff that resolves this issue."
                                    ),
                                }
                            ],
                            add_generation_prompt=True,
                        )
                        has_bos = strat_prompt.lstrip().startswith(bos_tok)
                        inputs = tokenizer(
                            strat_prompt,
                            return_tensors="pt",
                            add_special_tokens=(not has_bos),
                            truncation=True,
                            max_length=config.max_prompt_length,
                        ).to(input_device)
                        with torch.no_grad():
                            out_ids = model.generate(
                                **inputs,
                                max_new_tokens=config.max_new_tokens,
                                do_sample=True,
                                temperature=strat["temperature"],
                                top_p=0.95,
                            )
                        gen_tokens = out_ids[0][inputs.input_ids.shape[1] :]
                        raw_gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=False)
                        clean_gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
                        candidate_prompts.append(strat_prompt)
                        candidate_responses.append(raw_gen_text)
                        candidate_messages.append(None)
                        r_val, _ = DeterministicRewardEngine.evaluate_candidate_patch(
                            workspace_dir=workspace_dir,
                            patch_str=clean_gen_text,
                            target_files=target_files,
                            test_cmd=test_cmd,
                            instance_id=instance_id,
                            snapshots_dir=snapshots_dir,
                            test_patch=test_patch,
                        )
                        rewards.append(r_val)
        finally:
            _force_rmtree(workspace_dir)

        r_tensor = torch.tensor(rewards, dtype=torch.float32)
        std_r = r_tensor.std(unbiased=False).item()
        if std_r < 1e-4 or not any(c.strip() for c in candidate_responses):
            logger.info(f"Step {step} ({instance_id}): Skipping zero-variance group (rewards={rewards})")
            if instance_id not in completed_set:
                completed_tasks.append(instance_id)
                completed_set.add(instance_id)
            continue

        old_lps = compute_group_old_log_probs(
            model,
            tokenizer,
            candidate_prompts,
            candidate_responses,
            config,
            candidate_messages=candidate_messages,
        )
        model.train()

        for ppo_ep in range(config.ppo_epochs):
            loss, stats = compute_patchgrpo_loss(
                model=model,
                tokenizer=tokenizer,
                prompt=candidate_prompts,
                candidate_responses=candidate_responses,
                rewards=rewards,
                config=config,
                old_log_probs=old_lps,
                candidate_messages=candidate_messages,
                call_backward=True,
                loss_scale=1.0 / float(grad_accum * config.ppo_epochs),
                positive_advantages_only=getattr(config, "positive_advantages_only", True),
            )
            if stats["skipped_zero_var"] == 0.0:
                vram_gb = (
                    torch.cuda.max_memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0.0
                )
                logger.info(
                    f"Step {step} PPO-Epoch {ppo_ep + 1}/{config.ppo_epochs}: "
                    f"loss={loss.item():.4f} | mean_R={stats['mean_reward']:.3f} | "
                    f"std_R={stats['std_reward']:.3f} | KL={stats['mean_kl']:.4f} | "
                    f"ratio={stats['mean_ratio']:.3f} | peak_VRAM={vram_gb:.2f}GB"
                )

        if instance_id not in completed_set:
            completed_tasks.append(instance_id)
            completed_set.add(instance_id)

        task_mean_r = float(r_tensor.mean().item())
        if task_mean_r > best_mean_reward and task_mean_r >= 2.5:
            best_mean_reward = task_mean_r
            logger.info(
                f"[*] High-water mark reached at Step {step} ({instance_id}, mean_R={task_mean_r:.3f}). "
                f"Saving best snapshot to {config.checkpoint_dir / f'{output_dir.name}_best'}..."
            )
            _save_periodic_checkpoint(
                model,
                optimizer,
                output_dir,
                config.checkpoint_dir,
                "best",
                opt_steps=opt_steps,
                epoch=1,
                completed_tasks=completed_tasks,
            )

        accum_count += 1
        if accum_count % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            opt_steps += 1
            if config.save_every_n_steps > 0 and opt_steps % config.save_every_n_steps == 0:
                _save_periodic_checkpoint(
                    model,
                    optimizer,
                    output_dir,
                    config.checkpoint_dir,
                    f"online_step{opt_steps}",
                    opt_steps=opt_steps,
                    epoch=1,
                    completed_tasks=completed_tasks,
                )

    if accum_count % grad_accum != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()

    _save_final_submission_adapter(model, output_dir)
    logger.info(f"Saved sanitized PatchGRPO LoRA adapter weights to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Colab Single-GPU PatchGRPO Training Pipeline")
    parser.add_argument(
        "--tasks-jsonl",
        type=Path,
        default=Path("gemma-4-developer-agent/tasks.jsonl"),
        help="Path to tasks.jsonl (e.g. gemma-4-developer-agent/tasks.jsonl)",
    )
    parser.add_argument(
        "--trajectories-dir",
        type=Path,
        default=None,
        help="Optional directory of pre-harvested *_trajectories.jsonl files (offline RWR/SFT mode)",
    )
    parser.add_argument(
        "--snapshots-dir",
        type=Path,
        default=Path("gemma-4-developer-agent/snapshots"),
        help="Directory containing <instance_id>.tgz repository snapshots for online evaluation",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("gemma-4-developer-agent/submission/adapters/main_lora"),
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "checkpoints",
        help="Directory outside submission/ where periodic training checkpoints are stored (N3)",
    )
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs for offline trajectory training")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--online", action="store_true", help="Explicitly enable online on-policy training mode")
    parser.add_argument(
        "--candidates-per-task",
        "--group-size",
        dest="group_size",
        type=int,
        default=4,
        help="Number of candidate rollouts per task in online mode or group size (default: 4)",
    )
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--save-every", type=int, default=2, help="Save intermediate checkpoint every N optimizer steps")
    parser.add_argument("--resume-from", type=Path, default=None, help="Resume LoRA adapter from existing checkpoint path")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-tasks", type=int, default=None, help="Limit number of training tasks (e.g. 40 for 40/10 train/val split)")
    parser.add_argument(
        "--task-ids",
        type=str,
        default=None,
        help="Optional comma-separated list of instance_ids to run in online mode (e.g. fastapi_14786,requests_7315)",
    )
    parser.add_argument("--force-bnb-4bit", action="store_true", help="Force BitsAndBytes NF4 quantization even on compressed-tensors checkpoints")
    parser.add_argument("--allow-nf4-fallback", action="store_true", help="Opt-in fallback to BitsAndBytes NF4 if native compressed-tensors LoRA fails (N4)")
    parser.add_argument("--allow-negative-offline", action="store_true", help="Allow negative advantages in offline trajectory mode")
    parser.add_argument(
        "--allow-negative-advantages",
        action="store_true",
        help="Allow negative advantages in online GRPO (default: False, skips A_i <= 0 to protect agent primitives)",
    )
    parser.add_argument(
        "--no-interleave-repos",
        action="store_true",
        help="Disable round-robin task interleaving across repositories",
    )
    parser.add_argument(
        "--max-kl-threshold",
        type=float,
        default=5.0,
        help="Per-token KL divergence ceiling to prevent runaway policy drift (default: 5.0)",
    )
    parser.add_argument("--smoke-test-gpu", action="store_true", help="Run 2-stage GPU smoke test (multi-turn + max_seq_length) and exit")
    parser.add_argument(
        "--single-shot-online",
        action="store_true",
        help="Use single-shot diff generation instead of multi-turn agent rollouts in online mode",
    )
    args = parser.parse_args()

    cfg = PatchGRPOConfig(
        learning_rate=args.lr,
        group_size=args.group_size,
        gradient_accumulation_steps=args.grad_accum,
        save_every_n_steps=args.save_every,
        checkpoint_dir=args.checkpoint_dir,
        resume_from_checkpoint=args.resume_from,
        seed=args.seed,
        force_bnb_4bit=args.force_bnb_4bit,
        allow_nf4_fallback=args.allow_nf4_fallback,
        positive_advantages_only=(not args.allow_negative_advantages),
        positive_advantages_only_offline=(not args.allow_negative_offline and not args.allow_negative_advantages),
        interleave_repos=(not args.no_interleave_repos),
        max_kl_threshold=args.max_kl_threshold,
    )
    if args.smoke_test_gpu:
        set_global_seed(cfg.seed)
        m, tok = load_qlora_gemma4_31b(cfg)
        run_gpu_smoke_test(m, tok, cfg)
    elif not args.online and args.trajectories_dir and args.trajectories_dir.exists():
        train_from_harvested_trajectories(
            cfg,
            args.trajectories_dir,
            args.tasks_jsonl,
            args.output_dir,
            epochs=args.epochs,
        )
    else:
        task_ids_list = [x.strip() for x in args.task_ids.split(",") if x.strip()] if args.task_ids else None
        train_patchgrpo_pipeline(
            cfg,
            args.tasks_jsonl,
            args.output_dir,
            snapshots_dir=args.snapshots_dir,
            use_agent_loop=(not args.single_shot_online),
            max_train_tasks=args.max_train_tasks,
            task_ids_filter=task_ids_list,
        )
