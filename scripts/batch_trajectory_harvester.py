#!/usr/bin/env python3
"""Batch Trajectory Harvester for 40 Stratified SWE-bench Tasks.

Collects multi-turn agent interaction trajectories across the 40 stratified tasks,
evaluates candidate patches with pytest verification in the sandbox, computes
deterministic PatchGRPO rewards, and saves trajectory groups for policy training.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from eval.trajectory_harvester import harvest_task_trajectories

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("batch_harvester")


def run_batch_harvest(
    tasks_jsonl: Path,
    output_dir: Path,
    group_size: int = 4,
    backend: str = "vllm",
    model: str = "openai/google/gemma-4-31B-it-qat-w4a16-ct",
    api_base: str = "http://localhost:8000/v1",
    api_key: str = "EMPTY",
    max_turns: int = 8,
    skip_existing: bool = True,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(tasks_jsonl, encoding="utf-8") as f:
        tasks = [json.loads(line) for line in f if line.strip()]

    logger.info(f"Loaded {len(tasks)} tasks from {tasks_jsonl} for trajectory harvesting.")

    total_tasks = len(tasks)
    completed = 0
    skipped = 0

    for idx, task in enumerate(tasks, 1):
        instance_id = task["instance_id"]
        out_file = output_dir / f"{instance_id}_trajectories.jsonl"

        if skip_existing and out_file.exists() and out_file.stat().st_size > 0:
            logger.info(f"[{idx}/{total_tasks}] Skipping already harvested task: {instance_id}")
            skipped += 1
            continue

        logger.info(f"[{idx}/{total_tasks}] Harvesting trajectories for {instance_id} ({task['repo']})...")
        try:
            rollouts = harvest_task_trajectories(
                instance_id=instance_id,
                group_size=group_size,
                backend=backend,
                model=model,
                api_base=api_base,
                api_key=api_key,
                max_turns=max_turns,
            )
            # Ensure copied to output_dir
            default_out = REPO_ROOT / "results" / "trajectories" / f"{instance_id}_trajectories.jsonl"
            if default_out.exists() and default_out != out_file:
                out_file.write_text(default_out.read_text(encoding="utf-8"), encoding="utf-8")

            completed += 1
            logger.info(f"[{idx}/{total_tasks}] Finished {instance_id} ({len(rollouts)} rollouts).")
        except Exception as exc:
            logger.error(f"[{idx}/{total_tasks}] Error harvesting {instance_id}: {exc}", exc_info=True)

    logger.info(
        f"Batch harvesting complete: {completed} new tasks harvested, "
        f"{skipped} skipped, total {completed + skipped}/{total_tasks} available in {output_dir}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch trajectory harvester for PatchGRPO")
    parser.add_argument(
        "--tasks-jsonl",
        type=Path,
        default=REPO_ROOT / "gemma-4-developer-agent" / "tasks_40_stratified.jsonl",
        help="Path to stratified tasks JSONL",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "trajectories" / "trajectories_colab_40",
        help="Directory to save harvested trajectories",
    )
    parser.add_argument("--group-size", type=int, default=4, help="Number of rollouts per task")
    parser.add_argument("--backend", choices=["vllm", "direct"], default="vllm")
    parser.add_argument("--model", type=str, default="openai/google/gemma-4-31B-it-qat-w4a16-ct")
    parser.add_argument("--api-base", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--max-turns", type=int, default=8)
    parser.add_argument("--no-skip", action="store_true", help="Re-harvest existing tasks")
    args = parser.parse_args()

    run_batch_harvest(
        tasks_jsonl=args.tasks_jsonl,
        output_dir=args.output_dir,
        group_size=args.group_size,
        backend=args.backend,
        model=args.model,
        api_base=args.api_base,
        api_key="EMPTY",
        max_turns=args.max_turns,
        skip_existing=not args.no_skip,
    )
