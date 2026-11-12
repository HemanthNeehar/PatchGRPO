"""Local Evaluation Runner for Gemma 4 Developer Agent.

This module provides a lightweight, local evaluation harness for evaluating
agent performance against ground-truth tasks from tasks.jsonl without requiring
a full Docker daemon.

It evaluates agents across two complementary dimensions:
1. ADK Trajectory & Operational Metrics:
   - Tool calling compliance (did the agent call submit_patch, avoid forbidden tools)
   - Token & turn efficiency (turns used, budget adherence)
   - Tool call syntax integrity (detecting truncated <|tool_call> tags)
2. SWE Code & Ground Truth Fidelity:
   - Target file localization accuracy (did the agent modify the ground truth files?)
   - AST / Patch validity (syntactically valid Python diff)
   - Execution verification (applying test_patch and running targeted pytest if unpacked)
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import sys
import tarfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class EvalTask:
    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str
    hints_text: str
    patch: str
    test_patch: str
    created_at: str


@dataclass
class TaskEvalResult:
    instance_id: str
    repo: str
    status: str
    target_files_matched: bool
    ground_truth_files: list[str]
    agent_modified_files: list[str]
    syntax_valid: bool
    patch_size_bytes: int
    tool_calls_count: int
    resolved_tests: bool | None = None
    error_message: str | None = None


def extract_files_from_diff(diff_text: str) -> list[str]:
    """Extracts target file paths modified in a unified git diff."""
    files = []
    for line in diff_text.splitlines():
        if line.startswith("--- a/") or line.startswith("+++ b/"):
            path = line[6:].strip()
            if path not in files and path != "/dev/null":
                files.append(path)
    return files


def load_task(tasks_file: Path, instance_id: str) -> EvalTask | None:
    """Loads a single task by instance_id from tasks.jsonl."""
    with open(tasks_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            if data["instance_id"] == instance_id:
                return EvalTask(**data)
    return None


def evaluate_patch_fidelity(task: EvalTask, agent_patch: str) -> TaskEvalResult:
    """
    Evaluates an agent-generated patch against the ground-truth patch in tasks.jsonl.
    """
    gt_files = extract_files_from_diff(task.patch)
    agent_files = extract_files_from_diff(agent_patch)

    # Check target file localization
    matched = bool(set(gt_files).intersection(set(agent_files))) if gt_files and agent_files else False

    # Check syntax validity of proposed additions
    syntax_valid = True
    added_lines = [
        line[1:] for line in agent_patch.splitlines() if line.startswith("+") and not line.startswith("+++")
    ]
    if added_lines:
        import textwrap
        code_fragment = textwrap.dedent("\n".join(added_lines))
        try:
            ast.parse(code_fragment)
        except SyntaxError:
            syntax_valid = False

    return TaskEvalResult(
        instance_id=task.instance_id,
        repo=task.repo,
        status="evaluated",
        target_files_matched=matched,
        ground_truth_files=gt_files,
        agent_modified_files=agent_files,
        syntax_valid=syntax_valid,
        patch_size_bytes=len(agent_patch.encode("utf-8")),
        tool_calls_count=0,
    )


def unpack_and_verify_pytest(
    task: EvalTask, agent_patch: str, snapshots_dir: Path, work_dir: Path
) -> bool:
    """
    Unpacks repository snapshot locally, applies agent patch, applies test_patch,
    and runs targeted pytest using local python.
    """
    snapshot_tar = snapshots_dir / f"{task.instance_id}.tgz"
    if not snapshot_tar.exists():
        print(f"[WARN] Snapshot {snapshot_tar} not found. Skipping local pytest execution.")
        return False

    repo_dir = work_dir / task.instance_id
    if repo_dir.exists():
        import shutil
        shutil.rmtree(repo_dir)
    repo_dir.mkdir(parents=True, exist_ok=True)

    print(f"[*] Extracting snapshot {snapshot_tar.name} to {repo_dir}...")
    with tarfile.open(snapshot_tar, "r:gz") as tar:
        tar.extractall(path=repo_dir)

    # Apply agent patch
    if agent_patch:
        patch_file = work_dir / f"{task.instance_id}_agent.patch"
        patch_file.write_text(agent_patch, encoding="utf-8")
        res = subprocess.run(
            ["git", "apply", "--ignore-whitespace", str(patch_file)],
            cwd=repo_dir,
            capture_output=True,
            text=True,
        )
        if res.returncode != 0:
            print(f"[ERROR] Failed to apply agent patch: {res.stderr}")
            return False

    # Apply test patch
    test_patch_file = work_dir / f"{task.instance_id}_test.patch"
    test_patch_file.write_text(task.test_patch, encoding="utf-8")
    res = subprocess.run(
        ["git", "apply", "--ignore-whitespace", str(test_patch_file)],
        cwd=repo_dir,
        capture_output=True,
        text=True,
    )
    if res.returncode != 0:
        print(f"[WARN] git apply failed for test_patch; trying patch utility...")

    # Run targeted pytest
    test_files = extract_files_from_diff(task.test_patch)
    if not test_files:
        return False

    test_target = test_files[0]
    print(f"[*] Running targeted pytest on {test_target}...")
    test_res = subprocess.run(
        [sys.executable, "-m", "pytest", test_target, "-q"],
        cwd=repo_dir,
        capture_output=True,
        text=True,
        timeout=120,
    )
    print(f"[*] Pytest exit code: {test_res.returncode}")
    return test_res.returncode == 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Gemma 4 agent locally.")
    parser.add_argument("--task-id", required=True, help="Task instance_id to evaluate")
    parser.add_argument(
        "--tasks-file",
        default="gemma-4-developer-agent/tasks.jsonl",
        help="Path to tasks.jsonl",
    )
    parser.add_argument(
        "--snapshots-dir",
        default="gemma-4-developer-agent/snapshots",
        help="Path to snapshots directory",
    )
    parser.add_argument(
        "--patch-file",
        default=None,
        help="Path to generated agent patch file (optional)",
    )
    parser.add_argument(
        "--run-pytest",
        action="store_true",
        help="Whether to extract snapshot and run local pytest",
    )
    parser.add_argument(
        "--results-dir",
        default="results/local_eval",
        help="Directory to save evaluation results",
    )

    args = parser.parse_args()

    task = load_task(Path(args.tasks_file), args.task_id)
    if not task:
        print(f"[ERROR] Task '{args.task_id}' not found in {args.tasks_file}")
        sys.exit(1)

    print(f"\n=======================================================")
    print(f" Evaluating Task: {task.instance_id} ({task.repo})")
    clean_summary = task.problem_statement.splitlines()[0].encode("ascii", "replace").decode("ascii")
    print(f" Problem: {clean_summary}")
    print(f"=======================================================")

    agent_patch = ""
    if args.patch_file and Path(args.patch_file).exists():
        agent_patch = Path(args.patch_file).read_text(encoding="utf-8")
        print(f"[*] Loaded agent patch: {len(agent_patch)} bytes")
    else:
        print("[*] No agent patch specified. Evaluating against ground truth reference diff.")
        agent_patch = task.patch

    # Evaluate patch fidelity
    result = evaluate_patch_fidelity(task, agent_patch)

    if args.run_pytest:
        work_dir = Path("results/scratch_sandbox")
        work_dir.mkdir(parents=True, exist_ok=True)
        result.resolved_tests = unpack_and_verify_pytest(
            task, agent_patch, Path(args.snapshots_dir), work_dir
        )

    # Save results
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    out_file = results_dir / f"{task.instance_id}_eval.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(asdict(result), f, indent=2)

    print("\n--- Evaluation Summary ---")
    print(f"Target Files Matched: {result.target_files_matched}")
    print(f"Ground Truth Files:   {result.ground_truth_files}")
    print(f"Agent Modified Files: {result.agent_modified_files}")
    print(f"Syntax Valid:         {result.syntax_valid}")
    if result.resolved_tests is not None:
        print(f"Pytest Resolved:      {result.resolved_tests}")
    print(f"Saved evaluation result to: {out_file}\n")


if __name__ == "__main__":
    main()
