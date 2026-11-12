"""Comprehensive unit tests for PatchGRPO pipeline (`R2-9`):
  1. `WorkspaceTools` initialization signatures (Blocker 1).
  2. `extract_unified_diff` edge cases (fenced diff, generic fence, raw diff with context spaces, empty).
  3. Unified `[0.0, 5.0]` reward scale and `grade_agent_patch` `total_reward` key (Blocker 2).
  4. `DeterministicRewardEngine.evaluate_candidate_patch` on a synthetic toy `.tgz` snapshot
     (verifying `git apply`, AST check, `pytest`, and clean rollback to baseline `HEAD`).
  5. Gemma 4 native `{key: "val"}` tool-call formatting, `<tool_call|>` delimiter, `raw_content`
     preservation, and `compressed-tensors` detection (`R2-1`, Blockers 3 & 4).
  6. `tokenize_multi_turn_trajectory` prefix-rendered mask alignment (`R2-1`, `R2-9`).
  7. Tie-breaking without char/message noise and `min_reward_std_scale = 0.25`.
  8. Harvested trajectory JSONL consistency (`R2-7`: `patch_submitted`, `turns_used`, `grade.total_reward`).
"""

from __future__ import annotations

import contextlib
import json
import math
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = None

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.run_task_e2e import (
    DirectGemmaTransformersBackend,
     TOOL_DECLARATIONS,
    WorkspaceTools,
    _build_sandbox_env,
    compute_patchgrpo_reward,
    grade_agent_patch,
)
from scripts.colab_patchgrpo_pipeline import (
    DeterministicRewardEngine,
    PatchGRPOConfig,
    _compute_multi_turn_token_ids_and_mask,
    _force_rmtree,
    _format_gemma4_kv_args,
    _get_per_token_log_probs,
    _load_completed_tasks_from_checkpoint,
    _maybe_load_optimizer_and_step_state,
    _resolve_logits_keep_kwarg,
    _sanitize_submission_adapter_dir,
    _save_periodic_checkpoint,
    _patch_peft_for_compressed_tensors,
    _ensure_module_has_weight_attr,
    _ensure_model_decompressed,
    _unwrap_clippable_linears,
    _filter_text_target_modules,
    compute_group_old_log_probs,
    compute_patchgrpo_loss,
    compute_tie_broken_rewards,
    extract_training_target_from_record,
    extract_unified_diff,
    format_gemma4_chat_prompt,
    format_gemma4_message_turn,
    is_compressed_tensors_checkpoint,
)


def test_workspace_tools_init_signatures():
    """Verifies Blocker 1 & S5: both `WorkspaceTools(workdir)` and full `WorkspaceTools(workdir, repo, base_commit, max_turns)` work,
    `_build_sandbox_env` prepends `workdir` to `PYTHONPATH`, and `TOOL_DECLARATIONS` match `submission/agent.yaml`.
    """
    tmp = Path(tempfile.gettempdir())
    t1 = WorkspaceTools(tmp)
    t2 = WorkspaceTools(workdir=tmp, repo="psf/requests", base_commit="HEAD", max_turns=8)
    assert t1.workdir == tmp and t2.max_turns == 8
    env = _build_sandbox_env(tmp)
    assert str(tmp.resolve()) in env.get("PYTHONPATH", "")
    declared_names = [td["function"]["name"] for td in TOOL_DECLARATIONS]
    assert "rollback_patch" not in declared_names
    assert len(declared_names) == 9


def test_extract_unified_diff_cases():
    """Verifies `extract_unified_diff` across fenced, generic, raw, and whitespace-sensitive diffs (`R2-9`)."""
    # 1. Fenced ```diff block with trailing space on blank context line
    raw_fenced = (
        "Reasoning step.\n"
        "```diff\n"
        "--- a/pkg/mod.py\n"
        "+++ b/pkg/mod.py\n"
        "@@ -1,3 +1,3 @@\n"
        " def foo():\n"
        "-    return 1\n"
        "+    return 2\n"
        " \n"
        "```\n"
        "Summary."
    )
    d1 = extract_unified_diff(raw_fenced)
    assert d1.startswith("--- a/pkg/mod.py")
    assert d1.endswith(" \n"), f"Trailing context-line space should be preserved, got {repr(d1)}"

    # 2. Raw unfenced diff in prose
    raw_unfenced = (
        "I will apply this patch:\n"
        "diff --git a/a.py b/a.py\n"
        "--- a/a.py\n"
        "+++ b/a.py\n"
        "@@ -1 +1 @@\n"
        "-x = 0\n"
        "+x = 1\n"
    )
    d2 = extract_unified_diff(raw_unfenced)
    assert "diff --git a/a.py b/a.py" in d2 and "+x = 1" in d2

    # 3. Non-diff prose returns empty string
    assert extract_unified_diff("No diff here, just text.") == ""
    assert extract_unified_diff("") == ""


def test_unified_reward_scale_and_grade_agent_patch():
    """Verifies Blocker 2 & unified reward scale [0.0, 5.0]: `grade_agent_patch` returns `total_reward`."""
    tasks = {
        json.loads(line)["instance_id"]: json.loads(line)
        for line in (REPO_ROOT / "gemma-4-developer-agent" / "tasks.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    task = tasks["requests_7315"]
    traj_file = REPO_ROOT / "results" / "trajectories" / "trajectories_colab" / "requests_7315_trajectories.jsonl"
    rec = json.loads(traj_file.read_text(encoding="utf-8").splitlines()[0])
    winning_patch = rec["patch"]

    ws_dir = Path(tempfile.mkdtemp(prefix="unit_test_ws_requests_7315_"))
    try:
        r_val, metrics = DeterministicRewardEngine.evaluate_candidate_patch(
            workspace_dir=ws_dir,
            patch_str=f"Thought...\n```diff\n{winning_patch}\n```",
            target_files=["src/requests/adapters.py"],
            test_cmd=[sys.executable, "-c", "import ast; ast.parse(open('src/requests/adapters.py').read())"],
            instance_id="requests_7315",
            snapshots_dir=REPO_ROOT / "gemma-4-developer-agent" / "snapshots",
            test_patch=task.get("test_patch", ""),
        )
        assert r_val == 5.0, f"Expected unified 5.0 reward on passing surgical patch, got {r_val} ({metrics})"

        # Empty patch must return 0.0 on the unified scale
        r_empty, _ = DeterministicRewardEngine.evaluate_candidate_patch(
            workspace_dir=ws_dir,
            patch_str="",
            target_files=["src/requests/adapters.py"],
            test_cmd=[sys.executable, "-c", "pass"],
            instance_id="requests_7315",
            snapshots_dir=REPO_ROOT / "gemma-4-developer-agent" / "snapshots",
        )
        assert r_empty == 0.0, f"Expected 0.0 for empty patch, got {r_empty}"

        grade = grade_agent_patch(task=task, agent_patch=winning_patch, workdir=ws_dir, run_pytest=False)
        assert "total_reward" in grade and grade["total_reward"] >= 3.0, f"Missing total_reward in {grade}"
    finally:
        _force_rmtree(ws_dir)


def test_evaluate_candidate_patch_on_toy_snapshot():
    """Verifies `evaluate_candidate_patch` on a synthetic toy snapshot: apply, AST check, test, and rollback to HEAD (`R2-9`)."""
    tmp_root = Path(tempfile.mkdtemp(prefix="toy_snap_test_"))
    try:
        src_repo = tmp_root / "src_repo"
        src_repo.mkdir(parents=True)
        (src_repo / "calc.py").write_text("def add(a, b):\n    return a - b\n", encoding="utf-8", newline="\n")
        (src_repo / "test_calc.py").write_text(
            "from calc import add\ndef test_add():\n    assert add(2, 3) == 5\n",
            encoding="utf-8",
            newline="\n",
        )
        snaps_dir = tmp_root / "snapshots"
        snaps_dir.mkdir()
        snap_tgz = snaps_dir / "toy_1.tgz"
        with tarfile.open(snap_tgz, "w:gz") as tar:
            tar.add(src_repo / "calc.py", arcname="calc.py")
            tar.add(src_repo / "test_calc.py", arcname="test_calc.py")

        ws_dir = tmp_root / "ws"
        good_patch = (
            "--- a/calc.py\n"
            "+++ b/calc.py\n"
            "@@ -1,2 +1,2 @@\n"
            " def add(a, b):\n"
            "-    return a - b\n"
            "+    return a + b\n"
        )
        r_good, m_good = DeterministicRewardEngine.evaluate_candidate_patch(
            workspace_dir=ws_dir,
            patch_str=good_patch,
            target_files=["calc.py"],
            test_cmd=[sys.executable, "-c", "from calc import add; assert add(2, 3) == 5"],
            instance_id="toy_1",
            snapshots_dir=snaps_dir,
        )
        assert r_good == 5.0 and m_good["tests_passed"] and m_good["syntax_valid"]

        # Verify workspace was rolled back to clean baseline (`return a - b`)
        rolled_back_content = (ws_dir / "calc.py").read_text(encoding="utf-8")
        assert "return a - b" in rolled_back_content, f"Workspace was not rolled back to HEAD: {rolled_back_content}"
        status_proc = subprocess.run(["git", "status", "--porcelain"], cwd=ws_dir, capture_output=True, text=True)
        assert status_proc.stdout.strip() == "", f"Dirty git working tree after rollback: {status_proc.stdout}"
    finally:
        _force_rmtree(tmp_root)


def test_gemma4_chat_formatting_and_native_kv_syntax():
    """Verifies `R2-1`, Blocker 3 & 4: native `{key: "val"}` syntax, `<tool_call|>` delimiter, `raw_content`, and `compressed-tensors`."""
    kv = _format_gemma4_kv_args({"filepath": "a.py", "start_line": 1})
    assert kv == '{filepath: "a.py", start_line: 1}', f"Unexpected kv format: {kv}"

    msgs = [
        {"role": "system", "content": "Follow rules."},
        {"role": "user", "content": "Fix bug."},
        {
            "role": "assistant",
            "content": "Reading file.",
            "tool_calls": [{"function": {"name": "read_file", "arguments": {"filepath": "a.py", "start_line": 1}}}],
        },
        {"role": "tool", "name": "read_file", "content": '{"status": "ok"}'},
    ]
    formatted = format_gemma4_chat_prompt(None, msgs, add_generation_prompt=True)
    assert "<|turn>system" in formatted
    assert '<|tool_call>call:read_file{filepath: "a.py", start_line: 1}<tool_call|>' in formatted
    assert "<|tool_response>response:read_file" in formatted
    assert "<start_of_turn>" not in formatted

    # Verify raw_content (with `<|thought>...<thought|>`) is preserved verbatim when present
    asst_with_thought = {
        "role": "assistant",
        "content": " stripped ",
        "raw_content": '<|thought>\nChecking line 1.\n<thought|><|tool_call>call:submit_patch{}<tool_call|>',
    }
    turn_str = format_gemma4_message_turn(asst_with_thought)
    assert "<|thought>\nChecking line 1.\n<thought|>" in turn_str

    assert is_compressed_tensors_checkpoint("google/gemma-4-31B-it-qat-w4a16-ct") is True
    assert is_compressed_tensors_checkpoint("google/gemma-4-e2b-it") is False


def test_tie_breaking_without_char_noise():
    """Verifies Training Design Fix 1: tie-breaking only rewards `submit_patch` and turn efficiency."""
    cfg = PatchGRPOConfig()
    assert cfg.lora_dropout == 0.0
    assert cfg.min_reward_std_scale == 0.25

    r_same = [
        {"reward": 5.0, "patch": "+a = 1\n", "patch_submitted": True, "turns_used": 6},
        {"reward": 5.0, "patch": "+a = 100000000000\n", "patch_submitted": True, "turns_used": 6},
    ]
    adj_same = compute_tie_broken_rewards(r_same, max_turns=8)
    assert adj_same[0] == adj_same[1], f"Char length should not create tie-breaker noise: {adj_same}"

    r_diff = [
        {"reward": 5.0, "patch": "+a = 1\n", "patch_submitted": True, "turns_used": 6},
        {"reward": 5.0, "patch": "+a = 1\n", "patch_submitted": False, "turns_used": 8},
    ]
    adj_diff = compute_tie_broken_rewards(r_diff, max_turns=8)
    assert adj_diff[0] > adj_diff[1], f"Expected submit_patch + fewer turns to score higher: {adj_diff}"


def test_multi_turn_prefix_mask_alignment():
    """Verifies `R2-1` & `R2-9`: prefix-rendered `apply_chat_template` sets `loss_mask = 0.0` on
    system/user/tool turns (and tool declarations) and `loss_mask = 1.0` strictly on assistant turns.
    """
    class MockTemplateTokenizer:
        bos_token = "<bos>"
        bos_token_id = 1
        eos_token_id = 2

        def apply_chat_template(self, messages, tools=None, tokenize=False, add_generation_prompt=False):
            # Include tool declarations in prefix to verify they get masked out (loss_mask == 0.0)
            tool_hdr = f"[TOOLS={len(tools or [])}]\n"
            parts = ["<bos>" + tool_hdr]
            for m in messages:
                parts.append(format_gemma4_message_turn(m))
            if add_generation_prompt:
                parts.append("<|turn>model\n")
            return "".join(parts)

        def encode(self, text, add_special_tokens=True, truncation=False):
            raw = [ord(c) for c in text]
            return ([self.bos_token_id] + raw) if add_special_tokens else raw

    tok = MockTemplateTokenizer()
    messages = [
        {"role": "system", "content": "System prompt."},
        {"role": "user", "content": "Fix bug in app.py."},
        {
            "role": "assistant",
            "content": "Let me read app.py.",
            "tool_calls": [{"function": {"name": "read_file", "arguments": {"filepath": "app.py"}}}],
        },
        {"role": "tool", "name": "read_file", "content": '{"status": "ok", "content": "x = 1"}'},
        {
            "role": "assistant",
            "content": "Applying fix.",
            "tool_calls": [{"function": {"name": "submit_patch", "arguments": {}}}],
        },
    ]
    token_ids, loss_mask, first_asst_idx = _compute_multi_turn_token_ids_and_mask(tok, messages)
    assert len(token_ids) == len(loss_mask)
    assert first_asst_idx is not None and first_asst_idx > 0
    # Prefix before first assistant turn must be 100% masked out (0.0)
    assert all(m == 0.0 for m in loss_mask[:first_asst_idx])
    # Decode only the unmasked tokens (loss_mask == 1.0) and verify they contain ONLY the two assistant turns!
    unmasked_text = "".join(chr(t) for t, m in zip(token_ids, loss_mask) if m == 1.0)
    masked_text = "".join(chr(t) for t, m in zip(token_ids, loss_mask) if m == 0.0 and t > 2)
    assert "Let me read app.py." in unmasked_text
    assert "Applying fix." in unmasked_text
    assert '{"status": "ok", "content": "x = 1"}' not in unmasked_text
    assert '{"status": "ok", "content": "x = 1"}' in masked_text
    assert "[TOOLS=" in masked_text and "[TOOLS=" not in unmasked_text


def test_migrated_trajectories_consistency():
    """Verifies `R2-7`: all harvested trajectories in `trajectories_colab` have `patch_submitted`, `turns_used`, and `grade.total_reward`."""
    traj_dir = REPO_ROOT / "results" / "trajectories" / "trajectories_colab"
    for tf in sorted(traj_dir.glob("*_trajectories.jsonl")):
        for line in tf.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            assert "patch_submitted" in r, f"Missing patch_submitted in {tf.name}"
            assert "turns_used" in r, f"Missing turns_used in {tf.name}"
            assert "total_reward" in (r.get("grade") or {}), f"Missing grade.total_reward in {tf.name}"


def test_sampled_token_ids_preserves_assistant_turn_header_and_prefix_mutation_fallback():
    """Verifies `N1` & `N2`:
      1. When `sampled_token_ids` are attached to an assistant turn, the `<|turn>model\n` turn header
         is emitted with `loss_mask = 0.0` before `sampled_token_ids` (`loss_mask = 1.0`), and the
         trailing `<turn|>\n` delimiter is emitted with `loss_mask = 0.0`.
      2. When a chat template mutates earlier tokens on later turns (e.g. stripping earlier `<|thought>`
         blocks, violating `cur_ids[:len(prev_ids)] == prev_ids`), `_compute_multi_turn_token_ids_and_mask`
         detects the prefix mismatch and falls back to the turn-by-turn formatter with exact mask alignment.
    """
    class CharTemplateTokenizer:
        bos_token = "<bos>"
        bos_token_id = 1
        eos_token_id = 2

        def apply_chat_template(self, messages, tools=None, tokenize=False, add_generation_prompt=False):
            parts = ["<bos>"]
            for m in messages:
                parts.append(format_gemma4_message_turn(m))
            if add_generation_prompt:
                parts.append("<|turn>model\n")
            return "".join(parts)

        def encode(self, text, add_special_tokens=True, truncation=False):
            raw = [ord(c) for c in text]
            return ([self.bos_token_id] + raw) if add_special_tokens else raw

    tok = CharTemplateTokenizer()
    sampled_str = '<|thought>\nChecking line 1.\n<thought|><|tool_call>call:submit_patch{}<tool_call|>'
    sampled_ids = [ord(c) for c in sampled_str]

    msgs = [
        {"role": "system", "content": "Rules."},
        {"role": "user", "content": "Fix bug."},
        {
            "role": "assistant",
            "content": "",
            "raw_content": sampled_str,
            "sampled_token_ids": sampled_ids,
        },
    ]
    token_ids, loss_mask, first_asst_idx = _compute_multi_turn_token_ids_and_mask(tok, msgs)
    full_decoded = "".join(chr(t) for t in token_ids if t > 2)
    unmasked_decoded = "".join(chr(t) for t, m in zip(token_ids, loss_mask) if m == 1.0)
    masked_decoded = "".join(chr(t) for t, m in zip(token_ids, loss_mask) if m == 0.0 and t > 2)

    # N1 check: `<|turn>model\n` MUST be present in full sequence & masked out (0.0), while
    # `sampled_token_ids` MUST be unmasked (1.0) immediately after `<|turn>model\n`!
    assert "<|turn>model\n" + sampled_str in full_decoded, (
        f"Assistant header `<|turn>model\\n` missing before sampled_token_ids in: {repr(full_decoded)}"
    )
    assert unmasked_decoded == sampled_str, (
        f"Expected unmasked tokens to equal sampled_str, got: {repr(unmasked_decoded)}"
    )
    assert "<|turn>model\n" in masked_decoded
    assert full_decoded.endswith("<turn|>\n")

    # N2 check: Chat template that strips `<|thought>...<thought|>` on prior assistant turns
    # (simulating real Gemma 4 Jinja template behavior across multi-turn conversations)
    import re as _re

    class ThoughtStrippingTemplateTokenizer(CharTemplateTokenizer):
        def apply_chat_template(self, messages, tools=None, tokenize=False, add_generation_prompt=False):
            parts = ["<bos>"]
            for idx, m in enumerate(messages):
                is_last = idx == len(messages) - 1
                if m.get("role") == "assistant" and not is_last:
                    # Mutate earlier assistant turn by stripping `<|thought>...<thought|>` on later turns
                    m_copy = dict(m)
                    m_copy["content"] = _re.sub(
                        r"<\|thought>.*?<thought\|>", "", m_copy.get("content", ""), flags=_re.DOTALL
                    )
                    parts.append(format_gemma4_message_turn(m_copy))
                else:
                    parts.append(format_gemma4_message_turn(m))
            if add_generation_prompt:
                parts.append("<|turn>model\n")
            return "".join(parts)

    mutating_tok = ThoughtStrippingTemplateTokenizer()
    multi_turn_msgs = [
        {"role": "system", "content": "Rules."},
        {"role": "user", "content": "Fix bug."},
        {
            "role": "assistant",
            "content": "<|thought>\nSecret thought.\n<thought|>Read file.",
            "raw_content": "<|thought>\nSecret thought.\n<thought|>Read file.",
        },
        {"role": "tool", "name": "read_file", "content": '{"status": "ok"}'},
        {"role": "assistant", "content": "Done."},
    ]
    t_ids2, mask2, first_idx2 = _compute_multi_turn_token_ids_and_mask(mutating_tok, multi_turn_msgs)
    assert len(t_ids2) == len(mask2) and first_idx2 is not None
    unmasked2 = "".join(chr(t) for t, m in zip(t_ids2, mask2) if m == 1.0)
    masked2 = "".join(chr(t) for t, m in zip(t_ids2, mask2) if m == 0.0 and t > 2)
    assert "Secret thought." in unmasked2 and "Done." in unmasked2
    assert '{"status": "ok"}' in masked2 and '{"status": "ok"}' not in unmasked2


def test_periodic_checkpoint_isolation_and_submission_adapter_sanitization():
    """Verifies `N3` & `N4`:
      1. `_save_periodic_checkpoint` writes into `config.checkpoint_dir` (`results/checkpoints/`),
         never into `submission/adapters/`, and saves/restores `optimizer.pt` + `trainer_state.json`.
      2. `_sanitize_submission_adapter_dir` strips non-whitelisted tokenizer/trainer files from
         `submission/adapters/main_lora/` so `adk-submission` validation succeeds.
    """
    tmp_root = Path(tempfile.mkdtemp(prefix="test_ckpt_n3_"))
    try:
        sub_adapters_dir = tmp_root / "submission" / "adapters"
        main_lora_dir = sub_adapters_dir / "main_lora"
        ckpt_root_dir = tmp_root / "results" / "checkpoints"

        cfg = PatchGRPOConfig(checkpoint_dir=ckpt_root_dir)

        class DummyPeftModel:
            def __init__(self):
                if torch is not None and nn is not None:
                    self.w = nn.Parameter(torch.tensor([1.0, 2.0]))

            def parameters(self):
                return [self.w] if hasattr(self, "w") else []

            def save_pretrained(self, save_dir):
                Path(save_dir).mkdir(parents=True, exist_ok=True)
                (Path(save_dir) / "adapter_config.json").write_text('{"r": 16}', encoding="utf-8")
                (Path(save_dir) / "adapter_model.safetensors").write_bytes(b"SAFELORA")

        class DummyTokenizer:
            def save_pretrained(self, save_dir):
                Path(save_dir).mkdir(parents=True, exist_ok=True)
                (Path(save_dir) / "tokenizer.model").write_bytes(b"SPM")

        dummy_model = DummyPeftModel()
        dummy_tok = DummyTokenizer()
        opt = None
        if torch is not None and dummy_model.parameters():
            opt = torch.optim.AdamW(dummy_model.parameters(), lr=1e-4)
            loss = (dummy_model.w ** 2).sum()
            loss.backward()
            opt.step()

        ckpt_path = _save_periodic_checkpoint(
            model=dummy_model,
            optimizer=opt,
            output_dir=main_lora_dir,
            checkpoint_root=cfg.checkpoint_dir,
            step_label="ep1_step2",
            opt_steps=2,
            epoch=1,
            completed_tasks=["fastapi_14786", "requests_7315"],
        )
        assert ckpt_path == ckpt_root_dir / "main_lora_ep1_step2"
        assert ckpt_path.exists()
        assert (ckpt_path / "trainer_state.json").exists()
        if torch is not None:
            assert (ckpt_path / "optimizer.pt").exists()
        # Verify NO extra checkpoint directories were created inside `submission/adapters/`!
        assert [p.name for p in sub_adapters_dir.iterdir()] == ["main_lora"]
        assert {p.name for p in main_lora_dir.iterdir()} <= {"adapter_config.json", "adapter_model.safetensors", "README.md"}

        # Verify `_maybe_load_optimizer_and_step_state` and `_load_completed_tasks_from_checkpoint` restore state
        opt2 = torch.optim.AdamW(dummy_model.parameters(), lr=1e-4) if (torch is not None and dummy_model.parameters()) else None
        opt_steps_restored, epoch_restored = _maybe_load_optimizer_and_step_state(opt2, ckpt_path)
        assert opt_steps_restored == 2 and epoch_restored == 1
        assert _load_completed_tasks_from_checkpoint(ckpt_path) == ["fastapi_14786", "requests_7315"]

        # Now test `_sanitize_submission_adapter_dir` on `main_lora_dir` polluted with extra files
        dummy_model.save_pretrained(main_lora_dir)
        dummy_tok.save_pretrained(main_lora_dir)
        (main_lora_dir / "optimizer.pt").write_bytes(b"OPT")
        (main_lora_dir / "special_tokens_map.json").write_text("{}", encoding="utf-8")
        (main_lora_dir / "README.md").write_text("# LoRA", encoding="utf-8")

        removed = _sanitize_submission_adapter_dir(main_lora_dir)
        assert set(removed) == {"tokenizer.model", "optimizer.pt", "special_tokens_map.json"}
        remaining = {p.name for p in main_lora_dir.iterdir()}
        assert remaining == {"adapter_config.json", "adapter_model.safetensors", "README.md"}
    finally:
        _force_rmtree(tmp_root)


def test_compute_patchgrpo_loss_and_logprobs_numerics_on_cpu_toy_model():
    """Verifies `N5` & `N7`:
      - `_resolve_logits_keep_kwarg` detects `logits_to_keep` vs `num_logits_to_keep`.
      - At step 0 (when `old_log_probs` are computed from the current policy), the PPO importance
        ratio is 1.0, the Schulman KL approximation is >= 0.0, and gradients flow strictly to
        trainable LoRA parameters (`requires_grad=True`) while base parameters have `grad is None`.
    """
    class ModernForwardModel:
        def forward(self, input_ids, logits_to_keep=None):
            pass

    class LegacyForwardModel:
        def forward(self, input_ids, num_logits_to_keep=None):
            pass

    assert _resolve_logits_keep_kwarg(ModernForwardModel(), 5) == {"logits_to_keep": 5}
    assert _resolve_logits_keep_kwarg(LegacyForwardModel(), 5) == {"num_logits_to_keep": 5}

    if torch is None or nn is None:
        return

    class ToyLoRACausalLM(nn.Module):
        def __init__(self, vocab_size: int = 64, hidden_size: int = 16):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, hidden_size)
            self.embed.weight.requires_grad_(False)
            self.base_proj = nn.Linear(hidden_size, vocab_size, bias=False)
            self.base_proj.weight.requires_grad_(False)
            # Simulated LoRA adapter weights (non-zero so policy != reference when enabled)
            self.lora_A = nn.Parameter(torch.randn(4, hidden_size) * 0.1)
            self.lora_B = nn.Parameter(torch.randn(vocab_size, 4) * 0.1)
            self._adapter_disabled = False
            self.last_logits_to_keep = None

        def get_input_embeddings(self):
            return self.embed

        @contextlib.contextmanager
        def disable_adapter(self):
            prev = self._adapter_disabled
            self._adapter_disabled = True
            try:
                yield
            finally:
                self._adapter_disabled = prev

        def forward(self, input_ids, logits_to_keep: int | None = None):
            self.last_logits_to_keep = logits_to_keep
            h = self.embed(input_ids)
            if logits_to_keep is not None and logits_to_keep < h.shape[1]:
                h = h[:, -logits_to_keep:, :]
            logits = self.base_proj(h)
            if not self._adapter_disabled:
                logits = logits + (h @ self.lora_A.t()) @ self.lora_B.t()
            return type("Out", (), {"logits": logits})()

    class ToyCharTokenizer:
        bos_token = "<bos>"
        bos_token_id = 1
        eos_token_id = 2

        def apply_chat_template(self, messages, tools=None, tokenize=False, add_generation_prompt=False):
            parts = ["<bos>"]
            for m in messages:
                parts.append(format_gemma4_message_turn(m))
            if add_generation_prompt:
                parts.append("<|turn>model\n")
            return "".join(parts)

        def encode(self, text, add_special_tokens=True, truncation=False, max_length=None):
            raw = [(ord(c) % 60) + 3 for c in text]
            res = ([self.bos_token_id] + raw) if add_special_tokens else raw
            return res[:max_length] if max_length is not None else res

    torch.manual_seed(42)
    toy_model = ToyLoRACausalLM()
    toy_tok = ToyCharTokenizer()
    cfg = PatchGRPOConfig(group_size=2, logprob_chunk_size=16)

    # N5: verify `_resolve_logits_keep_kwarg` detects modern `logits_to_keep`
    assert _resolve_logits_keep_kwarg(toy_model, 7) == {"logits_to_keep": 7}

    prompt = "<bos><|turn>user\nFix bug.<turn|>\n<|turn>model\n"
    patches = ["--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-a=1\n+a=2\n", "--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-a=1\n+a=999\n"]
    rewards = [5.0, 1.0]

    old_lps = compute_group_old_log_probs(toy_model, toy_tok, prompt, patches, cfg)
    assert len(old_lps) == 2 and all(lp is not None for lp in old_lps)
    assert toy_model.last_logits_to_keep is not None, "Expected logits_to_keep to be passed to model.forward"

    toy_model.zero_grad()
    loss, stats = compute_patchgrpo_loss(
        model=toy_model,
        tokenizer=toy_tok,
        prompt=prompt,
        candidate_responses=patches,
        rewards=rewards,
        config=cfg,
        old_log_probs=old_lps,
        call_backward=True,
    )
    assert math.isfinite(loss.item())
    assert abs(stats["mean_ratio"] - 1.0) < 1e-5, f"Expected step-0 importance ratio == 1.0, got {stats['mean_ratio']}"
    # Because lora_A and lora_B are non-zero, policy != base reference so mean_kl > 0
    assert stats["mean_kl"] > 0.0, f"Expected strictly positive Schulman KL, got {stats['mean_kl']}"
    # Gradients must be non-zero on LoRA parameters and None on frozen base parameters
    assert toy_model.lora_A.grad is not None and toy_model.lora_A.grad.norm().item() > 0.0
    assert toy_model.lora_B.grad is not None and toy_model.lora_B.grad.norm().item() > 0.0
    assert toy_model.base_proj.weight.grad is None
    assert toy_model.embed.weight.grad is None


def test_backend_and_config_max_new_tokens_synchronized():
    """Verifies `N8`, `N9`, & `N10`:
      - `DirectGemmaTransformersBackend.__init__` defaults to `max_new_tokens=1024`, matching `PatchGRPOConfig.max_new_tokens`.
      - `extract_training_target_from_record` extracts assistant actions cleanly even when `patch` is empty (`N9`).
    """
    import inspect

    sig = inspect.signature(DirectGemmaTransformersBackend.__init__)
    assert sig.parameters["max_new_tokens"].default == PatchGRPOConfig().max_new_tokens == 1024

    rec_empty_patch = {
        "patch": "",
        "messages": [
            {"role": "system", "content": "Sys"},
            {"role": "user", "content": "Usr"},
            {
                "role": "assistant",
                "content": "Explored file.",
                "raw_content": '<|thought>\nExplored file.\n<thought|><|tool_call>call:read_file{filepath: "x.py"}<tool_call|>',
            },
        ],
    }
    target_text = extract_training_target_from_record(rec_empty_patch)
    assert isinstance(target_text, str) and "Explored file." in target_text


def test_gemma4_clippable_linear_unwrapping_and_target_module_filtering():
    """Verifies that:
      1. `_unwrap_clippable_linears` detects any `Gemma4ClippableLinear` wrapper and swaps it
         with its inner `.linear` layer (`torch.nn.Linear`).
      2. `_filter_text_target_modules` filters out multimodal vision/audio encoder towers
         (`vision_tower`, `audio_tower`) so PEFT targets ONLY text decoder layers.
    """
    class MockInnerLinear:
        def __init__(self, in_features, out_features):
            self.in_features = in_features
            self.out_features = out_features

    class Gemma4ClippableLinear:
        def __init__(self, in_features, out_features):
            self.linear = MockInnerLinear(in_features, out_features)

    class MockSelfAttn:
        def __init__(self, is_vision: bool = False):
            if is_vision:
                self.q_proj = Gemma4ClippableLinear(1152, 1152)
            else:
                self.q_proj = MockInnerLinear(4096, 4096)
            self.k_proj = MockInnerLinear(1152 if is_vision else 4096, 1152 if is_vision else 4096)

    class MockLayer:
        def __init__(self, is_vision: bool = False):
            self.self_attn = MockSelfAttn(is_vision=is_vision)

    class MockMultimodalModel:
        def __init__(self):
            self.vision_tower = type("VisionTower", (), {"layers": [MockLayer(is_vision=True)]})()
            self.language_model = type("LangModel", (), {"layers": [MockLayer(is_vision=False)]})()

        def named_modules(self):
            return [
                ("", self),
                ("vision_tower", self.vision_tower),
                ("vision_tower.layers", self.vision_tower.layers),
                ("vision_tower.layers.0", self.vision_tower.layers[0]),
                ("vision_tower.layers.0.self_attn", self.vision_tower.layers[0].self_attn),
                ("vision_tower.layers.0.self_attn.q_proj", self.vision_tower.layers[0].self_attn.q_proj),
                ("vision_tower.layers.0.self_attn.k_proj", self.vision_tower.layers[0].self_attn.k_proj),
                ("language_model", self.language_model),
                ("language_model.layers", self.language_model.layers),
                ("language_model.layers.0", self.language_model.layers[0]),
                ("language_model.layers.0.self_attn", self.language_model.layers[0].self_attn),
                ("language_model.layers.0.self_attn.q_proj", self.language_model.layers[0].self_attn.q_proj),
                ("language_model.layers.0.self_attn.k_proj", self.language_model.layers[0].self_attn.k_proj),
            ]

    model = MockMultimodalModel()
    assert model.vision_tower.layers[0].self_attn.q_proj.__class__.__name__ == "Gemma4ClippableLinear"

    # 1. Verify target module filtering excludes vision tower
    targets = ["q_proj", "k_proj"]
    filtered_targets = _filter_text_target_modules(model, targets)
    assert any("language_model" in t for t in filtered_targets)
    assert not any("vision_tower" in t for t in filtered_targets)

    # 2. Verify unwrapping converts Gemma4ClippableLinear to MockInnerLinear
    unwrapped_count = _unwrap_clippable_linears(model)
    assert unwrapped_count == 1
    assert model.vision_tower.layers[0].self_attn.q_proj.__class__.__name__ == "MockInnerLinear"


def test_compressed_tensors_peft_weight_proxy_and_safe_features():
    """Verifies that `_ensure_module_has_weight_attr` registers `compute_dtype` for packed INT4
    `compressed-tensors` layers, that `_patch_peft_for_compressed_tensors` allows PEFT's feature
    inspectors to resolve dimensions without raising `AttributeError: 'Linear' object has no attribute 'weight'`,
    and crucially that `nn.Linear` remains uncorrupted so adapter weights initialize without `KeyError`.
    """
    if torch is None:
        return

    class MockCTLinear(nn.Module):
        def __init__(self, in_f: int, out_f: int):
            super().__init__()
            self.in_features = in_f
            self.out_features = out_f
            # Compressed tensors store weight_packed instead of weight!
            self.weight_packed = torch.zeros((out_f, in_f // 8), dtype=torch.int32)

    class MockCTModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = MockCTLinear(64, 128)
            self.v_proj = MockCTLinear(64, 128)

    model = MockCTModel()
    assert not hasattr(model.q_proj, "weight")
    assert not hasattr(model.q_proj, "compute_dtype")

    # Step 1: Ensure compute_dtype is registered on module instances
    patched_count = _ensure_module_has_weight_attr(model)
    assert patched_count == 2
    assert hasattr(model.q_proj, "compute_dtype")

    # Step 2: Ensure nn.Linear class is NOT corrupted with any class property
    assert not hasattr(nn.Linear, "weight"), "nn.Linear class must not have a 'weight' descriptor!"
    lora_a = nn.Linear(64, 16, bias=False)
    assert lora_a.weight.shape == torch.Size([16, 64])

    # Step 3: Test monkey-patched PEFT feature inspector
    _patch_peft_for_compressed_tensors()
    try:
        import peft.tuners.tuners_utils as tu
        in_f, out_f = tu._get_in_out_features(model.q_proj)
        assert in_f == 64 and out_f == 128
    except ImportError:
        pass  # peft not installed locally on this machine


def test_ensure_model_decompressed_warmup_hook():
    """Verifies that `_ensure_model_decompressed` triggers a dummy forward pass when a model
    contains modules with `weight_packed` lacking `.weight`, and no-ops when weights are present.
    """
    if torch is None:
        return

    hook_fired = False

    class MockQuantizedLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight_packed = torch.zeros((8, 8), dtype=torch.int32)
            # Initially no `.weight`!

    class MockQuantizedRootModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = MockQuantizedLayer()
            self.get_input_embeddings = lambda: nn.Embedding(10, 16)

        def forward(self, input_ids, **kwargs):
            nonlocal hook_fired
            hook_fired = True
            # Simulate decompression hook unpacking into .weight
            self.layer.weight = torch.randn((8, 8), dtype=torch.float32)
            delattr(self.layer, "weight_packed")
            return type("Out", (), {"logits": torch.zeros((1, 1, 10))})()

    m = MockQuantizedRootModel()
    assert not hasattr(m.layer, "weight")
    assert hasattr(m.layer, "weight_packed")

    _ensure_model_decompressed(m)
    assert hook_fired is True
    assert hasattr(m.layer, "weight")
    assert not hasattr(m.layer, "weight_packed")

    # Second call should no-op
    hook_fired = False
    _ensure_model_decompressed(m)
    assert hook_fired is False


if __name__ == "__main__":
    test_workspace_tools_init_signatures()
    print("[PASS] 1. test_workspace_tools_init_signatures (Blocker 1)")
    test_extract_unified_diff_cases()
    print("[PASS] 2. test_extract_unified_diff_cases (R2-9)")
    test_unified_reward_scale_and_grade_agent_patch()
    print("[PASS] 3. test_unified_reward_scale_and_grade_agent_patch (Blocker 2 & unified [0, 5] scale)")
    test_evaluate_candidate_patch_on_toy_snapshot()
    print("[PASS] 4. test_evaluate_candidate_patch_on_toy_snapshot (R2-9 toy snapshot apply + rollback)")
    test_gemma4_chat_formatting_and_native_kv_syntax()
    print("[PASS] 5. test_gemma4_chat_formatting_and_native_kv_syntax (R2-1, Blockers 3 & 4)")
    test_tie_breaking_without_char_noise()
    print("[PASS] 6. test_tie_breaking_without_char_noise (Training design fixes)")
    test_multi_turn_prefix_mask_alignment()
    print("[PASS] 7. test_multi_turn_prefix_mask_alignment (R2-1, R2-9)")
    test_migrated_trajectories_consistency()
    print("[PASS] 8. test_migrated_trajectories_consistency (R2-7 trajectory JSONL consistency)")
    test_sampled_token_ids_preserves_assistant_turn_header_and_prefix_mutation_fallback()
    print("[PASS] 9. test_sampled_token_ids_preserves_assistant_turn_header_and_prefix_mutation_fallback (N1, N2)")
    test_periodic_checkpoint_isolation_and_submission_adapter_sanitization()
    print("[PASS] 10. test_periodic_checkpoint_isolation_and_submission_adapter_sanitization (N3, N4)")
    test_compute_patchgrpo_loss_and_logprobs_numerics_on_cpu_toy_model()
    print("[PASS] 11. test_compute_patchgrpo_loss_and_logprobs_numerics_on_cpu_toy_model (N5, N7)")
    test_backend_and_config_max_new_tokens_synchronized()
    print("[PASS] 12. test_backend_and_config_max_new_tokens_synchronized (N8, N9, N10)")
    test_gemma4_clippable_linear_unwrapping_and_target_module_filtering()
    print("[PASS] 13. test_gemma4_clippable_linear_unwrapping_and_target_module_filtering (Gemma4ClippableLinear Fix)")
    test_compressed_tensors_peft_weight_proxy_and_safe_features()
    print("[PASS] 14. test_compressed_tensors_peft_weight_proxy_and_safe_features (Compressed-Tensors PEFT Fix)")
    test_ensure_model_decompressed_warmup_hook()
    print("[PASS] 15. test_ensure_model_decompressed_warmup_hook (Decompression Hook Fix)")
