"""End-to-End Local Task Runner and Evaluation Harness for Gemma 4 Developer Agent.

This script executes the complete autonomous agent lifecycle on a single task:
1. Unpacks task snapshot from snapshots/<instance_id>.tgz into workspace/<instance_id>/
2. Initializes Git baseline commit ('eval_baseline') to isolate changes
3. Spins up the agent loop with all 9 bound tools matching swegemma:
   - read_file, edit_file (3-tier resilient matcher), write_file, run_command, submit_patch, get_status
   - get_code_neighbors (4-tier symbol resolution), search_similar_code, get_code_subgraph
4. Invokes the model (Gemma 4 via llama.cpp/Ollama or cloud proxy via LiteLLM)
5. Executes agent tool calls until submit_patch() is called or budget is exhausted
6. Extracts the generated git diff (agent_patch)
7. Evaluates the patch against the ground truth:
   - Compares with golden reference 'patch' from tasks.jsonl (file match, diff metrics)
   - Applies hidden 'test_patch' and runs targeted pytest to grade the fix!
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tarfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# Configure default paths
REPO_ROOT = Path(__file__).resolve().parent.parent
TASKS_FILE = REPO_ROOT / "gemma-4-developer-agent" / "tasks.jsonl"
SNAPSHOTS_DIR = REPO_ROOT / "gemma-4-developer-agent" / "snapshots"
GRAPHS_DIR = REPO_ROOT / "gemma-4-developer-agent" / "graphs"
EMBEDDINGS_DIR = REPO_ROOT / "gemma-4-developer-agent" / "embeddings"
WORKSPACE_BASE = REPO_ROOT / "workspace"
RESULTS_DIR = REPO_ROOT / "results" / "e2e_runs"


# ==============================================================================
# 1. Windows-Resilient Workspace Management
# ==============================================================================

def force_rmtree(path: Path | str) -> None:
    """Recursively removes a directory on Windows, clearing read-only attributes on Git packs."""
    p = Path(path)
    if not p.exists():
        return

    def _remove_readonly(func, pth, exc_info):
        try:
            os.chmod(pth, stat.S_IWRITE | stat.S_IREAD)
            func(pth)
        except Exception:
            pass

    try:
        shutil.rmtree(p, onerror=_remove_readonly)
    except Exception as e:
        print(f"[WARN] Failed to cleanly remove {p}: {e}")


def setup_task_workspace(instance_id: str, clean: bool = True) -> Path:
    """Unpacks the task snapshot into an isolated workspace and initializes Git baseline."""
    task_workdir = WORKSPACE_BASE / instance_id
    snapshot_tar = SNAPSHOTS_DIR / f"{instance_id}.tgz"

    if not snapshot_tar.exists():
        raise FileNotFoundError(f"Snapshot not found: {snapshot_tar}")

    if clean and task_workdir.exists():
        print(f"[*] Cleaning previous workspace: {task_workdir}")
        force_rmtree(task_workdir)

    if not task_workdir.exists():
        task_workdir.mkdir(parents=True, exist_ok=True)
        print(f"[*] Extracting {snapshot_tar.name} into {task_workdir}...")
        with tarfile.open(snapshot_tar, "r:gz") as tar:
            try:
                tar.extractall(path=task_workdir, filter="data")
            except TypeError:
                tar.extractall(path=task_workdir)

    # Initialize git repo if not already initialized
    git_dir = task_workdir / ".git"
    if not git_dir.exists():
        print("[*] Initializing baseline git repository...")
        subprocess.run(["git", "init", "-q"], cwd=task_workdir, check=True)
        subprocess.run(["git", "config", "core.fileMode", "false"], cwd=task_workdir, check=True)
        subprocess.run(["git", "config", "user.email", "agent@eval"], cwd=task_workdir, check=True)
        subprocess.run(["git", "config", "user.name", "Agent"], cwd=task_workdir, check=True)
        subprocess.run(["git", "add", "-A"], cwd=task_workdir, check=True)
        subprocess.run(["git", "commit", "-m", "eval_baseline", "--allow-empty", "-q"], cwd=task_workdir, check=True)

    return task_workdir


# ==============================================================================
# 2. 3-Tier Resilient Matching Engine (adk-eval-core compliant)
# ==============================================================================

def apply_3tier_replacement(content: str, old_string: str, new_string: str, allow_multiple: bool = False) -> tuple[str, str]:
    """
    Replaces old_string with new_string using adk-eval-core's 3-tier matching engine:
      Tier 1: Exact substring match (normalized newlines and unescaped variants)
      Tier 2: Flexible whitespace match with automatic baseline indentation preservation
      Tier 3: Regex tokenized delimiter match across whitespace
    """
    content_norm = content.replace("\r\n", "\n")
    old_norm = old_string.replace("\r\n", "\n")
    new_norm = new_string.replace("\r\n", "\n")

    # Defensive cleaning of trailing stray model backticks/quotes at boundary of new_string
    # e.g. when model generates: new_string: ... except (ValueError, OSError):`
    if new_norm.endswith("`") and not new_norm.startswith("`"):
        new_norm = new_norm.rstrip("`")
    if new_norm.endswith("'") and not new_norm.startswith("'"):
        new_norm = new_norm.rstrip("'")
    if new_norm.endswith('"') and not new_norm.startswith('"'):
        new_norm = new_norm.rstrip('"')

    # Candidate variants to handle escaping discrepancies from LLM tool argument generation
    candidates = [(old_norm, new_norm)]
    
    # Variant 1: unescape escaped newlines and tabs
    if "\\n" in old_norm or "\\t" in old_norm:
        candidates.append((
            old_norm.replace("\\n", "\n").replace("\\t", "\t"),
            new_norm.replace("\\n", "\n").replace("\\t", "\t")
        ))
        
    # Variant 2: unescape escaped quotes (\" and \')
    if '\\"' in old_norm or "\\'" in old_norm:
        candidates.append((
            old_norm.replace('\\"', '"').replace("\\'", "'"),
            new_norm.replace('\\"', '"').replace("\\'", "'")
        ))
        
    # Variant 3: unescape both quotes and newlines
    if ("\\n" in old_norm or "\\t" in old_norm) or ('\\"' in old_norm or "\\'" in old_norm):
        candidates.append((
            old_norm.replace("\\n", "\n").replace("\\t", "\t").replace('\\"', '"').replace("\\'", "'"),
            new_norm.replace("\\n", "\n").replace("\\t", "\t").replace('\\"', '"').replace("\\'", "'")
        ))

    # Deduplicate candidates while preserving order
    seen_cand = set()
    unique_candidates = []
    for o_cand, n_cand in candidates:
        if (o_cand, n_cand) not in seen_cand:
            seen_cand.add((o_cand, n_cand))
            unique_candidates.append((o_cand, n_cand))

    # --- Tier 1: Exact Match ---
    for target_old, target_new in unique_candidates:
        if target_old in content_norm:
            count = content_norm.count(target_old)
            if count > 1 and not allow_multiple:
                raise ValueError(f"Ambiguous match: old_string matches {count} occurrences (allow_multiple=False).")
            return content_norm.replace(target_old, target_new, 1 if not allow_multiple else -1), "exact"

    # --- Tier 2: Flexible Whitespace Match & Auto Indentation ---
    for target_old, target_new in unique_candidates:
        content_lines = content_norm.split("\n")
        old_lines = target_old.strip("\r\n").split("\n")
        old_stripped = [l.strip() for l in old_lines if l.strip()]

        if old_stripped:
            matches = []
            for i in range(len(content_lines) - len(old_lines) + 1):
                sub_lines = content_lines[i : i + len(old_lines)]
                sub_stripped = [l.strip() for l in sub_lines if l.strip()]
                if sub_stripped == old_stripped:
                    matches.append(i)

            if len(matches) > 1 and not allow_multiple:
                raise ValueError(f"Ambiguous match: flexible matching found {len(matches)} occurrences.")
            if len(matches) == 1:
                match_idx = matches[0]
                first_line = content_lines[match_idx]
                indent = first_line[: len(first_line) - len(first_line.lstrip())]

                new_lines = target_new.split("\n")
                if new_lines:
                    non_empty = [l for l in new_lines if l.strip()]
                    base_new_indent = min(len(l) - len(l.lstrip()) for l in non_empty) if non_empty else 0
                    reindented_new = [
                        indent + l[base_new_indent:] if l.strip() else ""
                        for l in new_lines
                    ]
                else:
                    reindented_new = []

                content_lines[match_idx : match_idx + len(old_lines)] = reindented_new
                return "\n".join(content_lines), "flexible"

    # --- Tier 3: Regex Delimiter Tokenization ---
    for target_old, target_new in unique_candidates:
        stripped_old = target_old.strip()
        if not stripped_old:
            continue
        tokens = re.split(r'([():\[\]{}>=<,;."\'])', stripped_old)
        tokens_clean = [t.strip() for t in tokens if t.strip()]
        if not tokens_clean:
            continue
        regex_pattern = r'\s*'.join(re.escape(t) for t in tokens_clean)
        regex_pattern = re.sub(r'(\\s\*)+', r'\\s*', regex_pattern)

        try:
            match_objs = list(re.finditer(regex_pattern, content_norm))
            if len(match_objs) > 1 and not allow_multiple:
                raise ValueError(f"Ambiguous match: regex matching found {len(match_objs)} occurrences.")
            if len(match_objs) == 1:
                m = match_objs[0]
                replaced = content_norm[: m.start()] + target_new + content_norm[m.end() :]
                return replaced, "regex"
        except Exception:
            pass

    raise ValueError("Failed to match old_string across exact, flexible, and regex tiers.")


# ==============================================================================
# 3. Complete Swegemma Workspace & Graph Tools (All 9 Tools Bound)
# ==============================================================================

def _build_sandbox_env(workdir: Path) -> dict[str, str]:
    """Constructs an isolated subprocess environment that places `workdir` and `workdir/src`
    at the front of `PYTHONPATH` so `pytest` always imports the checked-out repository snapshot
    rather than pre-installed system/Colab packages.
    """
    env = os.environ.copy()
    py_paths: list[str] = []
    src_dir = workdir / "src"
    if src_dir.exists() and src_dir.is_dir():
        py_paths.append(str(src_dir.resolve()))
    py_paths.append(str(workdir.resolve()))
    existing_pp = env.get("PYTHONPATH", "")
    if existing_pp:
        py_paths.append(existing_pp)
    env["PYTHONPATH"] = os.pathsep.join(py_paths)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


class WorkspaceTools:
    """Implements all 9 swegemma tools bound to the unpacked repository and graph data."""

    def __init__(self, workdir: Path, repo: str = "", base_commit: str = "HEAD", max_turns: int = 15):
        self.workdir = workdir
        self.repo = repo
        self.base_commit = base_commit
        self.max_turns = max_turns
        self.patch_submitted = False
        self.tool_call_count = 0

        # Load graph and embedding data if available
        self.graph_nodes: dict[str, dict[str, Any]] = {}
        self.adj_out: dict[str, list[tuple[str, str]]] = {}
        self.adj_in: dict[str, list[tuple[str, str]]] = {}
        self.embeddings: dict[str, Any] = {}
        self._load_graph_data()

    def _load_graph_data(self) -> None:
        stem = self.repo.replace("/", "_")
        candidates = [
            GRAPHS_DIR / f"{stem}_{self.base_commit}.json",
            GRAPHS_DIR / f"{self.repo.split('/')[-1]}_{self.base_commit}.json",
            GRAPHS_DIR / f"{stem}.json",
        ]
        for c in candidates:
            if c.exists():
                try:
                    with open(c, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    for node in data.get("nodes", []):
                        nid = node.get("id") or node.get("name")
                        if nid:
                            self.graph_nodes[nid] = node
                    for edge in data.get("edges", []):
                        src = edge.get("source")
                        tgt = edge.get("target")
                        etype = edge.get("type", "calls")
                        if src and tgt:
                            self.adj_out.setdefault(src, []).append((tgt, etype))
                            self.adj_in.setdefault(tgt, []).append((src, etype))
                    print(f"[*] Loaded repository call graph: {len(self.graph_nodes)} nodes, {len(data.get('edges', []))} edges.")
                    break
                except Exception as e:
                    print(f"[WARN] Failed to load graph {c}: {e}")

        # Load embeddings
        emb_candidates = [
            EMBEDDINGS_DIR / f"{stem}_{self.base_commit}.npz",
            EMBEDDINGS_DIR / f"{self.repo.split('/')[-1]}_{self.base_commit}.npz",
            EMBEDDINGS_DIR / f"{stem}.npz",
        ]
        for ec in emb_candidates:
            if ec.exists():
                try:
                    import numpy as np
                    self.embeddings = np.load(ec)
                    print(f"[*] Loaded node embeddings: {len(self.embeddings.files)} symbols.")
                    break
                except Exception as e:
                    print(f"[WARN] Failed to load embeddings {ec}: {e}")

    def _resolve(self, filepath: str) -> Path:
        filepath = filepath.strip().strip('`"')
        # Rule 4: Scratch reproduction scripts in /tmp outside workspace
        is_tmp = (
            filepath.startswith("/tmp") or
            filepath.startswith("\\tmp") or
            filepath.startswith("tmp/") or
            filepath.startswith("tmp\\")
        )
        if is_tmp:
            rel_name = filepath.removeprefix("/tmp").removeprefix("\\tmp").removeprefix("tmp/").removeprefix("tmp\\").lstrip("/\\")
            if not rel_name:
                rel_name = "repro.py"
            if os.name == "nt":
                try:
                    win_tmp = Path("C:/tmp")
                    win_tmp.mkdir(parents=True, exist_ok=True)
                    return (win_tmp / rel_name).resolve()
                except Exception:
                    tmp_dir = Path(tempfile.gettempdir()) / "sandbox_tmp"
                    tmp_dir.mkdir(parents=True, exist_ok=True)
                    return (tmp_dir / rel_name).resolve()
            else:
                tmp_dir = Path("/tmp")
                tmp_dir.mkdir(parents=True, exist_ok=True)
                return (tmp_dir / rel_name).resolve()

        clean = filepath.lstrip("/\\").removeprefix("workspace/").removeprefix("workspace\\")
        p = (self.workdir / clean).resolve()
        if not str(p).startswith(str(self.workdir.resolve())):
            raise PermissionError(f"Access denied: path outside workspace: {filepath}")
        return p

    def _resolve_symbol_4tier(self, query: str) -> str | None:
        """4-tier symbol resolution: exact -> suffix -> case-insensitive -> substring."""
        if not self.graph_nodes:
            return None
        all_keys = list(self.graph_nodes.keys())
        # Tier 1: exact
        if query in self.graph_nodes:
            return query
        # Tier 2: suffix match (e.g. .HTTPBasic or /HTTPBasic)
        for k in all_keys:
            if k.endswith("." + query) or k.endswith("/" + query):
                return k
        # Tier 3: case-insensitive
        q_lower = query.lower()
        for k in all_keys:
            if k.lower() == q_lower:
                return k
        # Tier 4: substring
        for k in all_keys:
            if q_lower in k.lower():
                return k
        return None

    # --- Tool Implementations ---

    def read_file(self, filepath: str, start_line: int | None = None, end_line: int | None = None) -> str:
        self.tool_call_count += 1
        path = self._resolve(filepath)
        if not path.exists():
            stem = Path(filepath).stem.lower()
            candidates = list(self.workdir.rglob(f"*{stem}*.py"))
            hint = ""
            if candidates:
                rel_candidates = [str(c.relative_to(self.workdir)).replace("\\", "/") for c in candidates[:3]]
                hint = f" (Did you mean: {', '.join(rel_candidates)}?)"
            return json.dumps({
                "status": "error",
                "error_type": "FileNotFound",
                "error_message": f"File not found: {filepath}.{hint}"
            })

        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        start = max(1, start_line or 1)
        end = min(len(lines), end_line or len(lines))
        sliced = lines[start - 1 : end][:150]
        content = "\n".join(sliced)
        return json.dumps({
            "status": "ok",
            "filepath": filepath,
            "start_line": start,
            "end_line": start + len(sliced) - 1,
            "total_lines": len(lines),
            "content": content[:10000],
            "is_truncated": len(lines) > (start - 1 + len(sliced)),
        })

    def edit_file(self, filepath: str, old_string: str, new_string: str, allow_multiple: bool = False) -> str:
        self.tool_call_count += 1
        path = self._resolve(filepath)
        if not path.exists():
            return json.dumps({"status": "error", "error_type": "FileEditError", "error_message": f"File not found: {filepath}"})

        content = path.read_text(encoding="utf-8", errors="replace")
        try:
            new_content, strategy = apply_3tier_replacement(content, old_string, new_string, allow_multiple)
            path.write_text(new_content, encoding="utf-8")
            return json.dumps({
                "status": "ok",
                "filepath": filepath,
                "strategy": strategy,
                "message": f"File edited successfully using {strategy} match strategy.",
            })
        except Exception as e:
            return json.dumps({"status": "error", "error_type": "FileEditError", "error_message": str(e)})

    def write_file(self, filepath: str, content: str) -> str:
        self.tool_call_count += 1
        path = self._resolve(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return json.dumps({"status": "ok", "filepath": filepath, "size": len(content)})

    def run_command(self, command: str) -> str:
        self.tool_call_count += 1
        cmd = command
        if os.name == "nt":
            if cmd.startswith("python3 "):
                cmd = f'"{sys.executable}" ' + cmd[8:]
            elif cmd.startswith("python "):
                cmd = f'"{sys.executable}" ' + cmd[7:]
        try:
            res = subprocess.run(
                cmd,
                cwd=self.workdir,
                env=_build_sandbox_env(self.workdir),
                shell=True,
                capture_output=True,
                text=True,
                timeout=120,
            )
            return json.dumps({
                "status": "ok" if res.returncode == 0 else "error",
                "exit_code": res.returncode,
                "stdout": res.stdout[:5000],
                "stderr": res.stderr[:5000],
            })
        except subprocess.TimeoutExpired:
            return json.dumps({"status": "error", "error_type": "TimeoutExceeded", "error_message": "Command timed out after 120s"})

    def get_status(self) -> str:
        # get_status is completely FREE (matches swegemma.tools.get_status schema)
        return json.dumps({
            "status": "ok",
            "tool_calls_used": self.tool_call_count,
            "tool_calls_remaining": max(0, self.max_turns - self.tool_call_count),
            "patch_submitted": self.patch_submitted,
            "max_turns": self.max_turns,
        })

    def submit_patch(self) -> str:
        # submit_patch is completely FREE (matches swegemma.tools.submit_patch schema)
        self.patch_submitted = True
        diff = self.extract_diff()
        return json.dumps({
            "status": "ok",
            "patch_submitted": True,
            "patch_size": len(diff),
            "files_changed": len([l for l in diff.splitlines() if l.startswith("+++ b/")]),
            "diff_preview": diff[:1000],
        })

    def rollback_patch(self) -> str:
        """Rolls back all uncommitted workspace edits to baseline HEAD commit."""
        self.tool_call_count += 1
        subprocess.run(["git", "checkout", "HEAD", "--", "."], cwd=self.workdir, capture_output=True, text=True)
        subprocess.run(["git", "clean", "-fd"], cwd=self.workdir, capture_output=True)
        return json.dumps({
            "status": "ok",
            "message": "All workspace modifications have been rolled back to baseline HEAD. You have a clean workspace.",
        })

    def get_code_neighbors(
        self,
        symbol: str | None = None,
        node: str | None = None,
        direction: str = "both",
        edge_type: str | None = None,
        edge_types: list[str] | None = None,
        max_neighbors: int = 50,
    ) -> str:
        self.tool_call_count += 1
        query_sym = symbol or node or ""
        resolved = self._resolve_symbol_4tier(query_sym)
        if not resolved:
            return json.dumps({"status": "error", "error_message": f"Symbol not found in graph: {query_sym}"})

        filter_types = [e.lower() for e in (edge_types or ([edge_type] if edge_type else [])) if e]
        neighbors = []
        if direction in ("both", "outgoing", "out"):
            for target, etype in self.adj_out.get(resolved, []):
                if not filter_types or any(ft in etype.lower() for ft in filter_types):
                    neighbors.append({"node": target, "relation": f"outgoing:{etype}"})
        if direction in ("both", "incoming", "in"):
            for source, etype in self.adj_in.get(resolved, []):
                if not filter_types or any(ft in etype.lower() for ft in filter_types):
                    neighbors.append({"node": source, "relation": f"incoming:{etype}"})

        return json.dumps({
            "status": "ok",
            "symbol": resolved,
            "node": resolved,
            "neighbors": neighbors[:max_neighbors],
            "count": min(len(neighbors), max_neighbors),
        })

    def search_similar_code(self, query: str, k: int = 10) -> str:
        self.tool_call_count += 1
        if not self.embeddings:
            return json.dumps({"status": "error", "error_message": "No embeddings archive available for this repo."})

        import numpy as np
        all_keys = list(self.embeddings.files)
        # Find closest query key in embeddings
        q_key = None
        for key in all_keys:
            if key == query or key.endswith("." + query) or query.lower() in key.lower():
                q_key = key
                break

        if not q_key:
            return json.dumps({"status": "error", "error_message": f"Query symbol not found in embedding dictionary: {query}"})

        q_vec = self.embeddings[q_key]
        q_norm = np.linalg.norm(q_vec) + 1e-9

        scores = []
        for key in all_keys:
            if key == q_key:
                continue
            vec = self.embeddings[key]
            sim = float(np.dot(q_vec, vec) / (q_norm * (np.linalg.norm(vec) + 1e-9)))
            scores.append({"node_name": key, "similarity": round(sim, 4)})

        scores.sort(key=lambda x: x["similarity"], reverse=True)
        return json.dumps({
            "status": "ok",
            "query": query,
            "matched_symbol": q_key,
            "results": scores[:k],
            "count": min(len(scores), k),
        })

    def get_code_subgraph(self, nodes: list[str]) -> str:
        self.tool_call_count += 1
        resolved_set = set()
        for n in nodes:
            res = self._resolve_symbol_4tier(n)
            if res:
                resolved_set.add(res)

        sub_edges = []
        for src in resolved_set:
            for tgt, etype in self.adj_out.get(src, []):
                if tgt in resolved_set:
                    sub_edges.append({"from": src, "to": tgt, "type": etype})

        return json.dumps({
            "status": "ok",
            "nodes": list(resolved_set),
            "edges": sub_edges,
            "node_count": len(resolved_set),
            "edge_count": len(sub_edges),
        })

    def extract_diff(self) -> str:
        # Rule 4: Defensively clean up any accidental scratch files in workspace
        for scratch_name in ["tmp", "repro.py", "test_repro.py", ".repro.py"]:
            sp = self.workdir / scratch_name
            if sp.exists():
                if sp.is_dir():
                    shutil.rmtree(sp, ignore_errors=True)
                else:
                    sp.unlink(missing_ok=True)

        subprocess.run(["git", "add", "-N", "."], cwd=self.workdir, capture_output=True)
        res = subprocess.run(
            ["git", "diff", "--binary", "HEAD"],
            cwd=self.workdir,
            capture_output=True,
            text=True,
        )
        return res.stdout


# ==============================================================================
# 4. Standardized OpenAI Function Tool Declarations
# ==============================================================================

TOOL_DECLARATIONS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Reads a file from the repository with inclusive line range (capped at 150 lines).",
            "parameters": {
                "type": "object",
                "properties": {
                    "filepath": {"type": "string", "description": "Relative path to file"},
                    "start_line": {"type": "integer", "description": "1-indexed start line"},
                    "end_line": {"type": "integer", "description": "1-indexed end line"},
                },
                "required": ["filepath"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Replaces old_string with new_string using 3-tier resilient matching (exact -> flexible whitespace -> regex delimiters).",
            "parameters": {
                "type": "object",
                "properties": {
                    "filepath": {"type": "string", "description": "Relative path to file"},
                    "old_string": {"type": "string", "description": "Exact text block to replace"},
                    "new_string": {"type": "string", "description": "Replacement text block"},
                    "allow_multiple": {"type": "boolean", "description": "Allow multiple replacements"},
                },
                "required": ["filepath", "old_string", "new_string"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Creates or overwrites a file at the given filepath.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filepath": {"type": "string", "description": "Relative path to file"},
                    "content": {"type": "string", "description": "Content to write"},
                },
                "required": ["filepath", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Runs a targeted shell command (e.g. pytest tests/test_target.py -k test_name). Temporary reproduction scripts MUST be placed in /tmp/repro.py.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command to execute"},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_status",
            "description": "Returns budget consumption and remaining turns (FREE tool, does not count toward budget).",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_patch",
            "description": "Submits your verified code fix and terminates the session. FREE tool; call this as your final action.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_code_neighbors",
            "description": "Finds incoming and outgoing callers/callees for a symbol in the repository call graph using 4-tier resolution.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Symbol name (e.g. HTTPBasic)"},
                    "direction": {"type": "string", "description": "Traversal direction: incoming, outgoing, or both (default both)"},
                    "edge_types": {"type": "array", "items": {"type": "string"}, "description": "Optional edge type filters (calls, imports, defined_in)"},
                    "max_neighbors": {"type": "integer", "description": "Max neighbors to return (default 50)"},
                },
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_similar_code",
            "description": "Finds top-k semantically similar functions/classes to query symbol from pre-computed embeddings.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Symbol name (e.g. HTTPBearer)"},
                    "k": {"type": "integer", "description": "Number of results to return (default 10)"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_code_subgraph",
            "description": "Extracts the induced subgraph (nodes and connecting edges) for a list of symbol names.",
            "parameters": {
                "type": "object",
                "properties": {
                    "nodes": {"type": "array", "items": {"type": "string"}, "description": "List of symbol names"},
                },
                "required": ["nodes"],
            },
        },
    },
]


# ==============================================================================
# 5. Direct In-Process Transformers Backend (No Server / No Localhost)
# ==============================================================================

class DirectGemmaTransformersBackend:
    """In-process model runner using Hugging Face Transformers.
    
    Eliminates localhost HTTP server dependencies in single-GPU / Google Colab environments.
    """

    def __init__(
        self,
        model_id: str = "google/gemma-4-31B-it-qat-w4a16-ct",
        device_map: str = "auto",
        model: Any = None,
        tokenizer: Any = None,
        max_new_tokens: int = 1024,
    ):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.max_new_tokens = max_new_tokens
        if model is not None and tokenizer is not None:
            print("[*] Reusing pre-loaded model and tokenizer in DirectGemmaTransformersBackend...")
            self.model = model
            self.tokenizer = tokenizer
        else:
            print(f"[*] Initializing Direct In-Process Transformers backend: {model_id}...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                device_map=device_map,
                trust_remote_code=True,
            )
            print("[+] Direct In-Process Model successfully loaded into VRAM!")

    def __call__(self, messages: list[dict], tools: list[dict], temperature: float = 0.2) -> Any:
        import torch
        # Prepare messages for HuggingFace chat_template:
        # tool_calls[].function.arguments MUST be a Python dictionary (mapping), NOT a string!
        hf_messages = []
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
                        except Exception:
                            args = self._parse_gemma_args(args, fn_name=fn_name)
                    func["arguments"] = args
                    tc_copy["function"] = func
                    clean_tcs.append(tc_copy)
                msg_copy["tool_calls"] = clean_tcs
            hf_messages.append(msg_copy)

        try:
            prompt = self.tokenizer.apply_chat_template(
                hf_messages,
                tools=tools,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            # Resilient Gemma 4 turn formatting (<|turn>...<turn|>) if Jinja template fails
            lines = []
            for m in hf_messages:
                r = m.get("role")
                c = m.get("content", "")
                if r == "system":
                    lines.append(f"<|turn>system\n{c}<turn|>")
                elif r == "user":
                    lines.append(f"<|turn>user\n{c}<turn|>")
                elif r == "assistant":
                    tcs = m.get("tool_calls", [])
                    tc_str = "".join(
                        f"<|tool_call>call:{tc['function']['name']}{json.dumps(tc['function']['arguments'])}<tool_call|>"
                        for tc in tcs
                    )
                    asst_body = f"{c}\n{tc_str}".strip() if c else tc_str
                    lines.append(f"<|turn>model\n{asst_body}<turn|>")
                elif r == "tool":
                    t_name = m.get("name", "tool")
                    lines.append(f"<|turn>user\n<|tool_response>response:{t_name}{c}<tool_response|><turn|>")
            lines.append("<|turn>model\n")
            prompt = "\n".join(lines)

        bos_tok = getattr(self.tokenizer, "bos_token", None) or "<bos>"
        has_bos = prompt.lstrip().startswith(bos_tok)
        input_device = (
            self.model.get_input_embeddings().weight.device
            if hasattr(self.model, "get_input_embeddings") and self.model.get_input_embeddings() is not None
            else self.model.device
        )
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=(not has_bos),
        ).to(input_device)
        prompt_len = inputs.input_ids.shape[1]

        was_training = getattr(self.model, "training", False)
        prev_use_cache = getattr(getattr(self.model, "config", None), "use_cache", True)
        if was_training:
            self.model.eval()
        if hasattr(self.model, "config"):
            self.model.config.use_cache = True
        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=max(temperature, 0.01),
                    do_sample=(temperature > 0),
                )
        finally:
            if hasattr(self.model, "config"):
                self.model.config.use_cache = prev_use_cache
            if was_training:
                self.model.train()

        gen_tokens = outputs[0][prompt_len:]
        out_text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
        raw_out_text = self.tokenizer.decode(gen_tokens, skip_special_tokens=False).strip()
        sampled_ids_list = [int(t) for t in gen_tokens.tolist()]

        tool_calls = self._extract_tool_calls(out_text)

        class FakeToolCallFunc:
            def __init__(self, name, args_str):
                self.name = name
                self.arguments = args_str

        class FakeToolCall:
            def __init__(self, tc_id, name, args_str):
                self.id = tc_id
                self.type = "function"
                self.function = FakeToolCallFunc(name, args_str)

        class FakeMessage:
            def __init__(self, content, tc_list, raw_content=None, sampled_token_ids=None):
                self.role = "assistant"
                self.content = content
                self.tool_calls = tc_list
                self.raw_content = raw_content
                self.sampled_token_ids = sampled_token_ids

        class FakeChoice:
            def __init__(self, msg):
                self.message = msg

        class FakeUsage:
            def __init__(self, prompt_tokens, comp_tokens):
                self.prompt_tokens = prompt_tokens
                self.completion_tokens = comp_tokens
                self.total_tokens = prompt_tokens + comp_tokens

        class FakeResponse:
            def __init__(self, choice, usage):
                self.choices = [choice]
                self.usage = usage

        tc_objects = [FakeToolCall(f"call_{i}", tc["name"], json.dumps(tc["arguments"])) for i, tc in enumerate(tool_calls)]
        fake_msg = FakeMessage(
            out_text,
            tc_objects if tc_objects else None,
            raw_content=raw_out_text,
            sampled_token_ids=sampled_ids_list,
        )
        return FakeResponse(FakeChoice(fake_msg), FakeUsage(prompt_len, len(gen_tokens)))

    @staticmethod
    def _extract_balanced_payload(text: str, start_brace_idx: int) -> tuple[str, int]:
        """Extracts the balanced JSON/dict payload starting at start_brace_idx ('{')."""
        depth = 0
        in_str = False
        quote_char = None
        escape = False
        i = start_brace_idx
        n = len(text)
        while i < n:
            ch = text[i]
            if not escape:
                if in_str:
                    if ch == quote_char:
                        in_str = False
                        quote_char = None
                else:
                    if ch in ('"', "'"):
                        if i + 2 < n and text[i:i+3] in ('"""', "'''"):
                            quote_char = text[i:i+3]
                            in_str = True
                            i += 2
                        else:
                            quote_char = ch
                            in_str = True
                    elif ch == '{':
                        depth += 1
                    elif ch == '}':
                        depth -= 1
                        if depth == 0:
                            return text[start_brace_idx : i + 1], i + 1
            if ch == '\\' and not escape:
                escape = True
            else:
                escape = False
            i += 1
        return text[start_brace_idx:], n

    @classmethod
    def _parse_gemma_args(cls, raw_args: str, fn_name: str | None = None) -> dict:
        raw_args = raw_args.strip()
        if not raw_args:
            return {}

        # 1. Standard JSON
        try:
            data = json.loads(raw_args)
            if isinstance(data, dict):
                return data
        except Exception:
            pass

        # 2. Python ast literal_eval (handles single-quoted dicts and multiline strings)
        try:
            import ast
            data = ast.literal_eval(raw_args)
            if isinstance(data, dict):
                return data
        except Exception:
            pass

        # 3. Strip outer braces
        inner = raw_args
        if inner.startswith("{") and inner.endswith("}"):
            inner = inner[1:-1].strip()

        if not inner:
            return {}

        def strip_enclosing_quotes(s: str) -> str:
            s = s.strip().strip('`')
            if (s.startswith('"""') and s.endswith('"""')) or (s.startswith("'''") and s.endswith("'''")):
                return s[3:-3].strip().strip('`')
            if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
                return s[1:-1].strip().strip('`')
            return s.strip('`')

        # 4. Tool-specific deterministic parsing
        if fn_name == "run_command":
            if re.match(r'^command\s*:\s*', inner, re.IGNORECASE):
                cmd = re.sub(r'^command\s*:\s*', '', inner, flags=re.IGNORECASE).strip()
                return {"command": strip_enclosing_quotes(cmd)}
            return {"command": strip_enclosing_quotes(inner)}

        if fn_name == "read_file":
            if ":" not in inner:
                return {"filepath": strip_enclosing_quotes(inner)}
            res = {}
            m_fp = re.search(r'filepath\s*:\s*(?:"""(.*?)"""|\'\'\'(.*?)\'\'\'|"([^"]*)"|\'([^\']*)\'|([^,\s}]+))', inner, re.DOTALL)
            if m_fp:
                val = next(g for g in m_fp.groups() if g is not None).strip()
                res["filepath"] = strip_enclosing_quotes(val)
            m_sl = re.search(r'start_line\s*:\s*(\d+)', inner)
            if m_sl:
                res["start_line"] = int(m_sl.group(1))
            m_el = re.search(r'end_line\s*:\s*(\d+)', inner)
            if m_el:
                res["end_line"] = int(m_el.group(1))
            return res

        if fn_name == "write_file":
            idx_fp = inner.find("filepath:")
            idx_content = inner.find("content:")
            if idx_fp != -1 and idx_content != -1:
                if idx_fp < idx_content:
                    fp_part = inner[idx_fp + len("filepath:"):idx_content].strip().rstrip(",").strip()
                    content_part = inner[idx_content + len("content:"):].strip()
                else:
                    content_part = inner[idx_content + len("content:"):idx_fp].strip().rstrip(",").strip()
                    fp_part = inner[idx_fp + len("filepath:"):].strip()
                fp = fp_part.strip(' "\'').rstrip(",").strip(' "\'')
                cnt = content_part
                if (cnt.startswith('"""') and cnt.endswith('"""')) or (cnt.startswith("'''") and cnt.endswith("'''")):
                    cnt = cnt[3:-3]
                elif (cnt.startswith('"') and cnt.endswith('"')) or (cnt.startswith("'") and cnt.endswith("'")):
                    cnt = cnt[1:-1]
                return {"filepath": fp, "content": cnt}
            elif idx_content != -1:
                cnt = inner[idx_content + len("content:"):].strip()
                if (cnt.startswith('"""') and cnt.endswith('"""')) or (cnt.startswith("'''") and cnt.endswith("'''")):
                    cnt = cnt[3:-3]
                elif (cnt.startswith('"') and cnt.endswith('"')) or (cnt.startswith("'") and cnt.endswith("'")):
                    cnt = cnt[1:-1]
                return {"content": cnt}
            elif idx_fp != -1:
                fp = inner[idx_fp + len("filepath:"):].strip().strip(' "\'')
                return {"filepath": fp}

        if fn_name == "edit_file":
            keys = ["filepath:", "old_string:", "new_string:", "allow_multiple:"]
            positions = []
            for k in keys:
                pos = inner.find(k)
                if pos != -1:
                    positions.append((pos, k))
            positions.sort()
            res = {}
            for idx, (pos, k) in enumerate(positions):
                val_start = pos + len(k)
                val_end = positions[idx + 1][0] if idx + 1 < len(positions) else len(inner)
                raw_val = inner[val_start:val_end].strip().rstrip(",").strip()
                clean_k = k.rstrip(":")
                if (raw_val.startswith('"""') and raw_val.endswith('"""')) or (raw_val.startswith("'''") and raw_val.endswith("'''")):
                    val = raw_val[3:-3]
                elif (raw_val.startswith('"') and raw_val.endswith('"')) or (raw_val.startswith("'") and raw_val.endswith("'")):
                    val = raw_val[1:-1]
                elif clean_k == "allow_multiple":
                    val = raw_val.lower() == "true"
                else:
                    val = raw_val
                res[clean_k] = val
            return res

        # 5. Generic regex key-value extraction fallback
        result = {}
        pattern = r'([a-zA-Z0-9_]+)\s*:\s*(?:"""(.*?)"""|\'\'\'(.*?)\'\'\'|"((?:\\.|[^"\\])*)"|\'((?:\\.|[^\'\\])*)\'|([^,{}]+))'
        matches = list(re.finditer(pattern, inner, re.DOTALL))
        if matches:
            for m in matches:
                k = m.group(1).strip()
                v = next((g for g in m.groups()[1:] if g is not None), None)
                if v is not None:
                    v = v.strip()
                    if v.isdigit():
                        v = int(v)
                    elif v.lower() == "true":
                        v = True
                    elif v.lower() == "false":
                        v = False
                    result[k] = v
            return result

        if inner:
            return {"command": inner}
        return {}

    def _extract_tool_calls(self, text: str) -> list[dict]:
        tool_calls = []
        call_iter = re.finditer(r'call:([a-zA-Z0-9_]+)\s*\{', text)
        for m in call_iter:
            fn_name = m.group(1)
            start_brace_idx = m.end() - 1
            payload, _ = self._extract_balanced_payload(text, start_brace_idx)
            parsed_args = self._parse_gemma_args(payload, fn_name=fn_name)
            tool_calls.append({"name": fn_name, "arguments": parsed_args})

        if not tool_calls:
            json_matches = re.finditer(r'```(?:json)?\s*(\{[^`]*?\})\s*```', text, re.DOTALL)
            for jm in json_matches:
                try:
                    data = json.loads(jm.group(1))
                    if "name" in data and "arguments" in data:
                        tool_calls.append(data)
                except Exception:
                    pass
        return tool_calls


# ==============================================================================
# 6. Model Execution Loop
# ==============================================================================

def dispatch_tool_call(tools: WorkspaceTools, func_name: str, func_args: Any) -> str:
    """Safely dispatches a tool call to WorkspaceTools with parameter filtering, argument aliases, and crash protection."""
    import inspect
    if not isinstance(func_args, dict):
        if isinstance(func_args, str):
            try:
                func_args = json.loads(func_args)
            except Exception:
                func_args = DirectGemmaTransformersBackend._parse_gemma_args(func_args, fn_name=func_name)
        else:
            func_args = {}

    method = getattr(tools, func_name, None)
    if method is None:
        return json.dumps({
            "status": "error",
            "error_type": "UnknownToolError",
            "error_message": f"Unknown tool '{func_name}'. Available tools: read_file, edit_file, write_file, run_command, get_status, submit_patch, rollback_patch, get_code_neighbors, search_similar_code, get_code_subgraph."
        })

    # Resilient argument aliasing
    if "filepath" not in func_args:
        for alias in ("path", "file", "filename"):
            if alias in func_args:
                func_args["filepath"] = func_args[alias]
                break
    if "command" not in func_args and "cmd" in func_args:
        func_args["command"] = func_args["cmd"]

    try:
        sig = inspect.signature(method)
        valid_params = set(sig.parameters.keys())
        # Filter out unexpected keyword arguments (e.g. spurious keys from unquoted code colons)
        filtered_args = {k: v for k, v in func_args.items() if k in valid_params}

        # Check for missing required arguments (arguments without default values)
        missing = [
            p.name for p in sig.parameters.values()
            if p.default == inspect.Parameter.empty and p.name not in filtered_args
        ]
        if missing:
            return json.dumps({
                "status": "error",
                "error_type": "MissingArgumentError",
                "error_message": f"Tool '{func_name}' is missing required argument(s): {', '.join(missing)}. "
                                 f"Accepted parameters: {list(valid_params)}. "
                                 f"Example: call:{func_name}{{{', '.join(f'{k}: ...' for k in valid_params)}}}"
            })

        return method(**filtered_args)
    except Exception as exc:
        return json.dumps({
            "status": "error",
            "error_type": type(exc).__name__,
            "error_message": f"Tool '{func_name}' execution failed: {str(exc)}"
        })


def run_agent_loop(
    problem_statement: str,
    tools: WorkspaceTools,
    model: str = "openai/google/gemma-4-E4B-it",
    api_base: str = "http://localhost:8080",
    api_key: str = "EMPTY",
    max_turns: int = 15,
    mock: bool = False,
    direct_generator: Any = None,
    temperature: float = 0.2,
    strategy_hint: str | None = None,
) -> tuple[str, dict[str, Any]]:
    """Executes the multi-turn agent interaction loop with Gemma 4, returning (agent_patch, loop_stats)."""
    import time
    start_time = time.time()
    turns_used = 0
    total_tokens = 0
    tool_counts: dict[str, int] = {}

    if mock:
        print("[*] Running in Mock/Dry-Run mode (no LLM call). Simulating direct inspection.")
        tools.read_file("fastapi/security/utils.py", 1, 30)
        tools.submit_patch()
        return tools.extract_diff(), {
            "turns_used": 1,
            "total_tokens": 120,
            "duration_sec": round(time.time() - start_time, 2),
            "tool_counts": {"read_file": 1, "submit_patch": 1},
        }

    import litellm
    litellm.drop_params = True

    submission_sys_md = REPO_ROOT / "gemma-4-developer-agent" / "submission" / "prompts" / "system.md"
    if submission_sys_md.exists():
        system_content = submission_sys_md.read_text(encoding="utf-8").strip()
    else:
        system_content = (
            "You are an expert autonomous software engineer assigned to resolve an issue in this repository.\n"
            "Workflow:\n"
            "1. Locate and read the relevant source file using read_file or search_similar_code.\n"
            "2. Apply a surgical, minimal fix to the repository source file using edit_file.\n"
            "3. (Optional) Run targeted tests to verify your fix.\n"
            "4. Always call submit_patch as your final action to submit your fix.\n"
        )
    if strategy_hint:
        system_content += f"\n\nStrategic Exploration Directive: {strategy_hint}"

    messages = [
        {
            "role": "system",
            "content": system_content,
        },
        {
            "role": "user",
            "content": f"Problem Statement:\n{problem_statement}\n\nPlease inspect the repository, implement the fix, and call submit_patch.",
        },
    ]

    mode_desc = "Direct In-Process Transformers" if direct_generator else f"'{model}' at '{api_base}'"
    print(f"[*] Starting agent session with {mode_desc} (temp={temperature}, max turns: {max_turns})...")

    for turn in range(1, max_turns + 1):
        if tools.patch_submitted:
            print(f"[*] Patch submitted on turn {turn - 1}. Exiting agent loop.")
            break

        print(f"\n--- [Turn {turn}/{max_turns}] Invoking Model (temp={temperature}) ---")

        # Turn-budget alert: If running low on turns and no files have been modified, warn the model
        remaining_turns = max_turns - turn + 1
        current_diff = tools.extract_diff()
        if remaining_turns <= 3 and not current_diff:
            if remaining_turns == 1:
                alert_msg = (
                    f"[FINAL TURN WARNING]: You have only 1 turn left and have NOT modified any repository files yet! "
                    f"You MUST call call:edit_file{{filepath: ..., old_string: ..., new_string: ...}} NOW to apply your fix. "
                    f"Do NOT call submit_patch without editing first, as an empty patch will score 0!"
                )
            else:
                alert_msg = (
                    f"[URGENT TURN-BUDGET ALERT]: You have only {remaining_turns} turn(s) left before termination and have NOT edited any repository files yet! "
                    f"Stop browsing files now. Identify the target file and apply your fix immediately using call:edit_file{{filepath: ..., old_string: ..., new_string: ...}}."
                )
            print(f"[*] {alert_msg}")
            messages.append({"role": "user", "content": alert_msg})
        elif remaining_turns <= 2 and current_diff and not tools.patch_submitted:
            alert_msg = f"[FINAL TURN ALERT]: You have modified repository files ({len(current_diff)} chars) and have only {remaining_turns} turn(s) left. Please call call:submit_patch{{}} immediately to finalize and submit your solution."
            print(f"[*] {alert_msg}")
            messages.append({"role": "user", "content": alert_msg})

        # Dynamically prune history to prevent token blowout and tool-call truncation
        from eval.context_pruner import prune_conversation_history
        active_messages = prune_conversation_history(messages, max_history_tokens=6000, keep_recent_turns=2)

        try:
            if direct_generator is not None:
                response = direct_generator(messages=active_messages, tools=TOOL_DECLARATIONS, temperature=temperature)
            else:
                response = litellm.completion(
                    model=model,
                    messages=active_messages,
                    tools=TOOL_DECLARATIONS,
                    api_base=api_base,
                    api_key=api_key,
                    temperature=temperature,
                    max_tokens=4096,
                )
        except Exception as e:
            print(f"[ERROR] LLM completion failed: {e}")
            break

        turns_used = turn
        if hasattr(response, "usage") and response.usage:
            total_tokens += getattr(response.usage, "total_tokens", 0) or 0

        choice = response.choices[0]
        message = choice.message
        
        # Serialize assistant message cleanly for subsequent turns
        if hasattr(message, "model_dump"):
            dumped = message.model_dump(exclude_unset=True)
            if "role" not in dumped:
                dumped["role"] = "assistant"
            messages.append(dumped)
        elif isinstance(message, dict):
            messages.append(message)
        else:
            asst_dict = {"role": "assistant", "content": getattr(message, "content", "") or ""}
            if getattr(message, "raw_content", None):
                asst_dict["raw_content"] = message.raw_content
            if getattr(message, "sampled_token_ids", None):
                asst_dict["sampled_token_ids"] = message.sampled_token_ids
            if hasattr(message, "tool_calls") and message.tool_calls:
                asst_dict["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments if isinstance(tc.function.arguments, str) else json.dumps(tc.function.arguments),
                        },
                    }
                    for tc in message.tool_calls
                ]
            messages.append(asst_dict)

        if getattr(message, "content", None):
            print(f"[Model Thought/Message]: {message.content[:300]}...")

        # Process tool calls
        if hasattr(message, "tool_calls") and message.tool_calls:
            for tc in message.tool_calls:
                func_name = tc.function.name
                raw_args = tc.function.arguments
                if isinstance(raw_args, str):
                    try:
                        func_args = json.loads(raw_args)
                    except Exception:
                        func_args = DirectGemmaTransformersBackend._parse_gemma_args(raw_args, fn_name=func_name)
                elif isinstance(raw_args, dict):
                    func_args = raw_args
                else:
                    func_args = {}

                tool_counts[func_name] = tool_counts.get(func_name, 0) + 1
                print(f"[Tool Call]: {func_name}({func_args})")

                res_str = dispatch_tool_call(tools, func_name, func_args)
                print(f"[Tool Result]: {res_str[:250]}...")

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": func_name,
                    "content": res_str,
                })
        else:
            print("[*] No tool call emitted. Prompting model to continue...")
            messages.append({
                "role": "user",
                "content": "Please continue by calling read_file, edit_file, or submit_patch.",
            })

    duration_sec = time.time() - start_time
    loop_stats = {
        "turns_used": turns_used,
        "total_tokens": total_tokens,
        "duration_sec": round(duration_sec, 2),
        "tool_counts": tool_counts,
        "patch_submitted": bool(tools.patch_submitted),
        "messages": messages,
    }
    return tools.extract_diff(), loop_stats


# ==============================================================================
# 6. Evaluation & Verification Against Golden References
# ==============================================================================

def compute_patchgrpo_reward(
    grade_res: dict[str, Any],
    task: dict[str, Any],
    agent_patch: str,
) -> float:
    """Canonical unified PatchGRPO reward function on [0.0, 5.0] scale shared by
    `grade_agent_patch`, `trajectory_harvester.py`, and `colab_patchgrpo_pipeline.py`.

    Breakdown:
      - Empty or unapplied patch: 0.0
      - R_syntax (+1.0): Modified Python files parse cleanly via `ast.parse`
      - R_loc (+1.5 or +0.2): +1.5 if target ground-truth file matched, +0.2 if non-target file edited
      - R_test (+2.0): Targeted pytest passes with `test_patch` applied
      - R_concise (+0.5 / -0.5): +0.5 if diff size <= 2x ground-truth (or <= 25 lines), -0.5 if > 5x (or > 80 lines)
    """
    agent_files = grade_res.get("agent_files", [])
    patch_size = int(grade_res.get("patch_size", 0))
    if not agent_files or patch_size == 0 or not (agent_patch or "").strip():
        return 0.0

    reward = 0.0
    if grade_res.get("syntax_valid"):
        reward += 1.0

    if grade_res.get("target_matched"):
        reward += 1.5
    else:
        reward += 0.2

    if grade_res.get("pytest_resolved"):
        reward += 2.0

    gt_patch = task.get("patch", "") if isinstance(task, dict) else ""
    gt_lines = len([
        l for l in gt_patch.splitlines()
        if (l.startswith("+") or l.startswith("-")) and not l.startswith("+++") and not l.startswith("---")
    ]) or len(gt_patch.splitlines())

    if gt_lines > 0:
        ratio = patch_size / max(gt_lines, 1)
        if ratio <= 2.0:
            reward += 0.5
        elif ratio > 5.0:
            reward -= 0.5
    else:
        if 0 < patch_size <= 25:
            reward += 0.5
        elif patch_size > 80:
            reward -= 0.5

    return round(max(0.0, reward), 3)


def grade_agent_patch(
    task: dict,
    agent_patch: str,
    workdir: Path,
    run_pytest: bool = True,
) -> dict:
    """Grades agent_patch against ground truth patch and test_patch from tasks.jsonl."""
    print("\n=======================================================")
    print(" Phase 2 Verification & Ground Truth Scoring")
    print("=======================================================")

    def extract_files(diff_text: str) -> list[str]:
        files = []
        for line in diff_text.splitlines():
            if line.startswith("--- a/") or line.startswith("+++ b/"):
                f = line[6:].strip()
                if f not in files and f != "/dev/null":
                    files.append(f)
        return files

    gt_files = extract_files(task["patch"])
    agent_files = extract_files(agent_patch)
    matched_files = set(gt_files).intersection(set(agent_files))

    print(f"[*] Ground Truth Files:  {gt_files}")
    print(f"[*] Agent Modified Files: {agent_files}")
    print(f"[*] Target Localization:  {'MATCHED' if matched_files else 'MISSED'}")

    syntax_ok = True
    added_lines = [l[1:] for l in agent_patch.splitlines() if l.startswith("+") and not l.startswith("+++")]
    removed_lines = [l[1:] for l in agent_patch.splitlines() if l.startswith("-") and not l.startswith("---")]
    total_diff_lines = len(added_lines) + len(removed_lines)

    # Check AST syntax on actual modified repository files
    if agent_files:
        for af in agent_files:
            fp = workdir / af
            if fp.exists() and fp.suffix == ".py":
                try:
                    ast.parse(fp.read_text(encoding="utf-8", errors="ignore"))
                except SyntaxError:
                    syntax_ok = False
                    break
    print(f"[*] AST Syntax Validity: {'VALID' if syntax_ok else 'SYNTAX_ERROR'}")

    pytest_passed = None
    if run_pytest and task.get("test_patch"):
        print("[*] Applying hidden test_patch for Phase 2 verification...")
        test_patch_file = workdir / "_eval_test.patch"
        test_patch_file.write_text(task["test_patch"], encoding="utf-8")
        subprocess.run(["git", "apply", "--ignore-whitespace", str(test_patch_file)], cwd=workdir, capture_output=True)

        test_files = extract_files(task["test_patch"])
        if test_files:
            test_target = test_files[0]
            print(f"[*] Running targeted pytest on {test_target}...")
            res = subprocess.run(
                [sys.executable, "-m", "pytest", test_target, "-q"],
                cwd=workdir,
                env=_build_sandbox_env(workdir),
                capture_output=True,
                text=True,
                timeout=120,
            )
            pytest_passed = (res.returncode == 0)
            print(f"[*] Pytest Result: {'PASSED (Score = 1.0)' if pytest_passed else 'FAILED (Score = 0.0)'} (exit: {res.returncode})")

    grade_res = {
        "instance_id": task["instance_id"],
        "target_matched": bool(matched_files),
        "ground_truth_files": gt_files,
        "agent_files": agent_files,
        "syntax_valid": syntax_ok,
        "patch_size": total_diff_lines,
        "patch_size_bytes": len(agent_patch.encode("utf-8")),
        "pytest_resolved": pytest_passed,
    }
    total_reward = compute_patchgrpo_reward(grade_res, task, agent_patch)
    grade_res["total_reward"] = total_reward
    grade_res["reward"] = total_reward
    return grade_res


# ==============================================================================
# 7. CLI Entrypoint
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="End-to-End Task Runner with Gemma 4 Invocation and Grading.")
    parser.add_argument("--task-id", default="fastapi_14786", help="Instance ID from tasks.jsonl (e.g. fastapi_14786)")
    parser.add_argument("--model", default="openai/google/gemma-4-E4B-it", help="Model identifier")
    parser.add_argument("--api-base", default="http://localhost:8080", help="API base URL (default: http://localhost:8080)")
    parser.add_argument("--api-key", default="EMPTY", help="API key")
    parser.add_argument("--max-turns", type=int, default=10, help="Maximum agent reasoning turns")
    parser.add_argument("--mock", action="store_true", help="Run without an LLM to verify workspace & grading end-to-end")
    parser.add_argument("--no-clean", action="store_true", help="Preserve existing unpacked workspace")

    args = parser.parse_args()

    # Load task
    with open(TASKS_FILE, "r", encoding="utf-8") as f:
        tasks = {json.loads(line)["instance_id"]: json.loads(line) for line in f if line.strip()}

    if args.task_id not in tasks:
        print(f"[ERROR] Task '{args.task_id}' not found in {TASKS_FILE}")
        sys.exit(1)

    task = tasks[args.task_id]
    print(f"\n=======================================================")
    print(f" End-to-End Task Evaluation: {task['instance_id']} ({task['repo']})")
    print(f" Base Commit: {task['base_commit']}")
    clean_summary = task["problem_statement"].splitlines()[0].encode("ascii", "replace").decode("ascii")
    print(f" Problem: {clean_summary}")
    print(f"=======================================================")

    # 1. Unpack snapshot into workspace with Windows permission protection
    workdir = setup_task_workspace(task["instance_id"], clean=not args.no_clean)

    # 2. Bind all 9 tools including graph and 3-tier matcher
    tools = WorkspaceTools(workdir, repo=task["repo"], base_commit=task["base_commit"], max_turns=args.max_turns)

    # 3. Run Gemma 4 agent loop
    agent_patch, loop_stats = run_agent_loop(
        problem_statement=task["problem_statement"],
        tools=tools,
        model=args.model,
        api_base=args.api_base,
        api_key=args.api_key,
        max_turns=args.max_turns,
        mock=args.mock,
    )

    # 4. Save extracted patch
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    patch_out = RESULTS_DIR / f"{task['instance_id']}.patch"
    patch_out.write_text(agent_patch, encoding="utf-8")
    print(f"\n[*] Extracted agent patch saved to: {patch_out} ({len(agent_patch)} bytes)")

    # 5. Grade against golden reference
    score = grade_agent_patch(task, agent_patch, workdir, run_pytest=True)
    score.update(loop_stats)

    # Save summary
    summary_out = RESULTS_DIR / f"{task['instance_id']}_summary.json"
    summary_out.write_text(json.dumps(score, indent=2), encoding="utf-8")
    print(f"[*] Complete evaluation summary saved to: {summary_out}")


if __name__ == "__main__":
    main()
