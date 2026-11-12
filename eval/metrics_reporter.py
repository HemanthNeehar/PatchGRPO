"""Metrics and Cost Reporter for Gemma 4 Autonomous Developer Agent.

Aggregates benchmark run outputs, computes financial cost-per-resolved-issue,
and exports publication-grade Markdown and LaTeX comparison tables for the ArXiv paper.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
from pathlib import Path
from typing import Any, List, Dict

REPO_ROOT = Path(__file__).resolve().parent.parent
BENCHMARK_DIR = REPO_ROOT / "results" / "benchmark_runs"
PAPERS_DIR = REPO_ROOT / "papers" / "tables"
PAPERS_DIR.mkdir(parents=True, exist_ok=True)

# Standardized Pricing Model for Gemma 4 31B (Cloud Inference Estimates per 1M Tokens)
INPUT_COST_PER_M = 0.20   # $0.20 / 1M prompt tokens
OUTPUT_COST_PER_M = 0.60  # $0.60 / 1M completion tokens


def compute_run_metrics(run_path: Path) -> dict[str, Any]:
    with open(run_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    metrics = data.get("metrics", {})
    total_tasks = metrics.get("total_tasks", 0)
    resolved_count = metrics.get("resolved_count", 0)
    avg_tokens = metrics.get("avg_tokens", 0)
    total_tokens = avg_tokens * total_tasks

    # Estimate 75% prompt tokens, 25% generation tokens
    est_prompt_tokens = total_tokens * 0.75
    est_gen_tokens = total_tokens * 0.25
    total_cost_usd = (est_prompt_tokens / 1e6 * INPUT_COST_PER_M) + (est_gen_tokens / 1e6 * OUTPUT_COST_PER_M)
    cost_per_task = total_cost_usd / max(total_tasks, 1)
    cost_per_resolved = total_cost_usd / max(resolved_count, 1) if resolved_count > 0 else 0.0

    return {
        "run_id": data.get("run_id", run_path.stem),
        "timestamp": data.get("timestamp", ""),
        "model": data.get("model", ""),
        "environment": data.get("environment", ""),
        "total_tasks": total_tasks,
        "pass_rate_pct": metrics.get("pass_rate_pct", 0.0),
        "localization_rate_pct": metrics.get("localization_rate_pct", 0.0),
        "syntax_valid_pct": metrics.get("syntax_valid_pct", 0.0),
        "avg_turns": metrics.get("avg_turns", 0.0),
        "avg_tokens": avg_tokens,
        "avg_latency_sec": metrics.get("avg_latency_sec", 0.0),
        "total_cost_usd": round(total_cost_usd, 4),
        "cost_per_task_usd": round(cost_per_task, 4),
        "cost_per_resolved_usd": round(cost_per_resolved, 4),
    }


def generate_latex_table(runs: list[dict[str, Any]]) -> str:
    """Generates LaTeX tabular code ready for inclusion in the ArXiv research paper."""
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\caption{Ablation and Weekly Benchmark Progression across SWE Complexity Tiers on Gemma 4.}",
        r"\label{tab:benchmark_progression}",
        r"\begin{tabular}{lccccccr}",
        r"\toprule",
        r"\textbf{Run ID} & \textbf{Model / Setup} & \textbf{Pass@1 (\%)} & \textbf{Loc. (\%)} & \textbf{AST (\%) } & \textbf{Avg Turns} & \textbf{Avg Tokens} & \textbf{Cost / Resolved (\$) } \\",
        r"\midrule",
    ]

    for r in runs:
        row = (
            f"{r['run_id']} & {r['environment']} & "
            f"{r['pass_rate_pct']}\\% & {r['localization_rate_pct']}\\% & {r['syntax_valid_pct']}\\% & "
            f"{r['avg_turns']} & {r['avg_tokens']:,} & "
            f"\\${r['cost_per_resolved_usd']:.3f} \\\\"
        )
        lines.append(row)

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ])
    return "\n".join(lines)


def generate_markdown_table(runs: list[dict[str, Any]]) -> str:
    """Generates a Markdown table for dashboards or GitHub READMEs."""
    header = (
        "| Run ID | Model / Environment | Pass@1 (%) | Localization (%) | AST Valid (%) | Avg Turns | Avg Tokens | Cost / Resolved ($) |\n"
        "|---|---|---|---|---|---|---|---|\n"
    )
    rows = []
    for r in runs:
        rows.append(
            f"| **{r['run_id']}** | {r['environment']} | {r['pass_rate_pct']}% | "
            f"{r['localization_rate_pct']}% | {r['syntax_valid_pct']}% | {r['avg_turns']} | "
            f"{r['avg_tokens']:,} | ${r['cost_per_resolved_usd']:.3f} |"
        )
    return header + "\n".join(rows)


def main():
    runs = []
    for file in sorted(BENCHMARK_DIR.glob("*.json")):
        try:
            m = compute_run_metrics(file)
            runs.append(m)
        except Exception as e:
            print(f"[WARN] Failed to parse {file}: {e}")

    if not runs:
        print("No benchmark JSON runs found in results/benchmark_runs/")
        return

    print("\n=======================================================")
    print("  WEEKLY BENCHMARK & COST SUMMARY REPORT")
    print("=======================================================")
    md_table = generate_markdown_table(runs)
    print(md_table)

    latex_table = generate_latex_table(runs)
    latex_out = PAPERS_DIR / "benchmark_comparison.tex"
    with open(latex_out, "w", encoding="utf-8") as f:
        f.write(latex_table)
    print(f"\n[+] Exported publication LaTeX table to: {latex_out}")


if __name__ == "__main__":
    main()
