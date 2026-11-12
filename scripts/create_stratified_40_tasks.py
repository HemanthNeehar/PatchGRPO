#!/usr/bin/env python3
"""Generates a stratified 40-task benchmark dataset for PatchGRPO training.
Balances across all 4 repositories and complexity tiers (surgical, medium, multi-file).
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TASKS_FILE = REPO_ROOT / "gemma-4-developer-agent" / "tasks.jsonl"
OUT_FILE = REPO_ROOT / "gemma-4-developer-agent" / "tasks_40_stratified.jsonl"


def select_stratified_tasks(tasks_path: Path, out_path: Path, target_count: int = 40):
    with open(tasks_path, encoding="utf-8") as f:
        all_tasks = [json.loads(line) for line in f if line.strip()]

    by_repo = defaultdict(list)
    for t in all_tasks:
        patch_lines = len(t.get("patch", "").splitlines())
        t["_patch_lines"] = patch_lines
        # Exclude extreme outlier renames / macro auto-generated diffs (> 300 lines)
        if patch_lines <= 300:
            by_repo[t["repo"]].append(t)

    # Quotas for balanced 40 tasks:
    # fastapi: 20
    # rich: 13
    # requests: 6
    # httpx: 1
    quotas = {
        "fastapi/fastapi": 20,
        "Textualize/rich": 13,
        "psf/requests": 6,
        "encode/httpx": 1,
    }

    selected = []
    for repo, quota in quotas.items():
        rtasks = by_repo.get(repo, [])
        rtasks.sort(key=lambda x: x["_patch_lines"])
        if len(rtasks) <= quota:
            chosen = rtasks
        else:
            # Sample evenly across the complexity spectrum
            step = len(rtasks) / float(quota)
            chosen = [rtasks[int(i * step)] for i in range(quota)]
        selected.extend(chosen)

    # Clean temporary sorting key
    for t in selected:
        t.pop("_patch_lines", None)

    with open(out_path, "w", encoding="utf-8") as f:
        for t in selected:
            f.write(json.dumps(t) + "\n")

    print(f"Successfully generated {len(selected)} stratified tasks at: {out_path}")
    counts = defaultdict(int)
    for t in selected:
        counts[t["repo"]] += 1
    for r, c in counts.items():
        print(f"  - {r}: {c} tasks")


if __name__ == "__main__":
    select_stratified_tasks(TASKS_FILE, OUT_FILE, 40)
