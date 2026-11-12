"""Trajectory Harvester and Rejection Sampler for PatchGRPO.

Collects multi-turn agent interaction trajectories across tasks in tasks.jsonl,
evaluates candidate patches against AST syntax, localization, and test execution,
computes deterministic PatchGRPO rewards, and classifies trajectories into
'chosen', 'localized_unverified', and 'rejected' cohorts for policy training.
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from eval.run_task_e2e import (
    setup_task_workspace,
    WorkspaceTools,
    run_agent_loop,
    grade_agent_patch,
    compute_patchgrpo_reward,
    DirectGemmaTransformersBackend,
)
from eval.benchmark_suite import load_all_tasks

TRAJECTORY_DIR = REPO_ROOT / "results" / "trajectories"
TRAJECTORY_DIR.mkdir(parents=True, exist_ok=True)


# Multi-Axis Exploration Schedule for Group Diversity in GRPO
DIVERSITY_SCHEDULE = [
    {
        "temperature": 0.6,
        "strategy": "Surgical root-cause focus: Locate the exact failure line and apply a minimal, localized 1-3 line fix.",
    },
    {
        "temperature": 0.8,
        "strategy": "Defensive invariant focus: Inspect caller arguments, trace invariants, and add robust input validation or fallback handling.",
    },
    {
        "temperature": 0.95,
        "strategy": "Graph-guided architectural exploration: Use get_code_neighbors and search_similar_code to inspect callers and related classes before modifying code.",
    },
    {
        "temperature": 0.75,
        "strategy": "Test-driven reproduction: Focus on the exact assertion that failed in the issue description and repair the return condition.",
    },
]


def harvest_task_trajectories(
    instance_id: str,
    group_size: int = 4,
    backend: str = "vllm",
    model: str = "openai/google/gemma-4-E4B-it",
    api_base: str = "http://localhost:8080",
    api_key: str = "EMPTY",
    max_turns: int = 8,
    mock: bool = False,
) -> list[dict[str, Any]]:
    """Runs G candidate rollouts for a single task and scores their rewards."""
    all_tasks = load_all_tasks()
    if instance_id not in all_tasks:
        raise ValueError(f"Task {instance_id} not found in tasks.jsonl")

    direct_gen = None
    if backend == "direct" and not mock:
        raw_model = model.removeprefix("openai/")
        direct_gen = DirectGemmaTransformersBackend(model_id=raw_model)

    task_data = all_tasks[instance_id]
    problem_statement = task_data.get("problem_statement", "")

    rollouts = []
    print(f"\n=======================================================")
    print(f" [*] Harvester: Sampling G={group_size} Diverse Rollouts for {instance_id}")
    print(f"=======================================================")

    for g_idx in range(1, group_size + 1):
        sched = DIVERSITY_SCHEDULE[(g_idx - 1) % len(DIVERSITY_SCHEDULE)]
        temp = sched["temperature"]
        strat = sched["strategy"]

        print(f"\n >>> Sampling Rollout {g_idx}/{group_size} <<<")
        print(f" [*] Strategy: temp={temp} | '{strat[:50]}...'")

        # Setup clean isolated workspace
        workdir = setup_task_workspace(instance_id, clean=True)

        tools = WorkspaceTools(
            workdir=workdir,
            repo=task_data["repo"],
            base_commit=task_data.get("base_commit", "HEAD"),
            max_turns=max_turns,
        )

        agent_patch, loop_stats = run_agent_loop(
            problem_statement=problem_statement,
            tools=tools,
            model=model,
            api_base=api_base,
            api_key=api_key,
            max_turns=max_turns,
            mock=mock,
            direct_generator=direct_gen,
            temperature=temp,
            strategy_hint=strat,
        )

        # Grade patch against ground truth (includes unified compute_patchgrpo_reward)
        grade_res = grade_agent_patch(task_data, agent_patch, workdir, run_pytest=True)
        patch_size = grade_res.get("patch_size", 0)
        reward = float(grade_res.get("total_reward", compute_patchgrpo_reward(grade_res, task_data, agent_patch)))

        # Cohort classification - strictly requires target localization and non-empty patch for "chosen"
        if grade_res.get("pytest_resolved") and patch_size > 0 and grade_res.get("target_matched"):
            cohort = "chosen"
        elif grade_res.get("target_matched") and grade_res.get("syntax_valid") and patch_size > 0:
            cohort = "localized_unverified"
        else:
            cohort = "rejected"

        # Separate messages from loop_stats to keep summary clean while retaining full trajectory
        msgs = loop_stats.get("messages", [])
        clean_stats = {k: v for k, v in loop_stats.items() if k != "messages"}

        rollout_record = {
            "rollout_id": f"{instance_id}_g{g_idx}",
            "instance_id": instance_id,
            "group_index": g_idx,
            "cohort": cohort,
            "reward": round(reward, 3),
            "patch_submitted": bool(tools.patch_submitted),
            "turns_used": int(loop_stats.get("turns_used", 0)),
            "grade": grade_res,
            "loop_stats": clean_stats,
            "messages": msgs,
            "patch": agent_patch,
            "patch_preview": agent_patch[:500] if agent_patch else "",
            "timestamp": datetime.datetime.now().isoformat(),
        }
        rollouts.append(rollout_record)
        print(f" [*] Rollout {g_idx} Result: Cohort={cohort} | Reward={reward:.2f}")

    # Compute Group Diversity Metrics for PatchGRPO
    rewards = [r["reward"] for r in rollouts]
    mean_r = sum(rewards) / max(len(rewards), 1)
    variance_r = sum((r - mean_r) ** 2 for r in rewards) / max(len(rewards), 1)
    std_r = variance_r ** 0.5
    unique_patches = len(set(r.get("patch_preview", "") for r in rollouts if r.get("patch_preview")))

    print("\n-------------------------------------------------------")
    print(f" [GRPO Group Metrics] Mean Reward: {mean_r:.2f} | Std(R): {std_r:.3f}")
    print(f" [GRPO Group Diversity] Unique Patches: {unique_patches}/{len(rollouts)}")
    if std_r < 1e-4:
        print(" [!] WARN: Zero reward variance in group. Try higher temperature.")
    else:
        print(" [+] Healthy advantage contrast established for PatchGRPO policy update.")
    print("-------------------------------------------------------")

    # Save to disk
    out_file = TRAJECTORY_DIR / f"{instance_id}_trajectories.jsonl"
    with open(out_file, "w", encoding="utf-8") as f:
        for r in rollouts:
            f.write(json.dumps(r) + "\n")

    print(f"\n[+] Saved {len(rollouts)} rollouts to {out_file}")
    return rollouts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Harvest agent trajectories for PatchGRPO")
    parser.add_argument("--task-id", type=str, default="fastapi_14786", help="Task ID from tasks.jsonl")
    parser.add_argument("--group-size", type=int, default=4, help="Number of rollouts per task (default: 4)")
    parser.add_argument(
        "--backend",
        choices=["vllm", "direct"],
        default="vllm",
        help="Backend mode: 'direct' (Transformers in-process) or 'vllm' (HTTP)",
    )
    parser.add_argument("--model", type=str, default="openai/google/gemma-4-E4B-it")
    parser.add_argument("--api-base", type=str, default="http://localhost:8080")
    parser.add_argument("--api-key", type=str, default="EMPTY")
    parser.add_argument("--max-turns", type=int, default=8, help="Maximum agent reasoning turns per rollout (default: 8)")
    parser.add_argument("--mock", action="store_true", help="Simulate rollouts without calling LLM")
    args = parser.parse_args()

    harvest_task_trajectories(
        instance_id=args.task_id,
        group_size=args.group_size,
        backend=args.backend,
        model=args.model,
        api_base=args.api_base,
        api_key=args.api_key,
        max_turns=args.max_turns,
        mock=args.mock,
    )
