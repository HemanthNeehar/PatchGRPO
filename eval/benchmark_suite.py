"""Benchmark Suite Orchestrator for Gemma 4 Developer Agent.

Executes a standardized 5-task benchmark spanning distinct complexity tiers:
- Tier 1: fastapi_14786 (Surgical Fix, 8 lines, Single-Function)
- Tier 2: requests_7315 (Targeted Protocol Fix, 11 lines, URL Parsing)
- Tier 3: rich_4077 (Stream I/O & Delegation, 9 lines, Terminal Console)
- Tier 4: fastapi_14873 (Routing & Lifecycle, 37 lines, Multi-Function)
- Tier 5: fastapi_15661 (Macro Architectural Stress-Test, 219 lines, Multi-Module)

Tracks Pass@1, Localization Accuracy, AST Syntax Validity, Turn Counts,
Token Burn, and Latency across weekly iterative milestones.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, List, Dict

REPO_ROOT = Path(__file__).resolve().parent.parent
TASKS_FILE = REPO_ROOT / "gemma-4-developer-agent" / "tasks.jsonl"
BENCHMARK_DIR = REPO_ROOT / "results" / "benchmark_runs"
DASHBOARD_DIR = REPO_ROOT / "dashboard"

BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)

# Import end-to-end task runner machinery
sys.path.insert(0, str(REPO_ROOT))
from eval.run_task_e2e import setup_task_workspace, WorkspaceTools, run_agent_loop, grade_agent_patch

# 5 Canonical Complexity-Tiered Benchmark Tasks
CANONICAL_TASKS = [
    {
        "instance_id": "fastapi_14786",
        "repo": "fastapi/fastapi",
        "tier": "Tier 1: Surgical Fix",
        "complexity_level": "Low (8 lines)",
        "domain": "Security / Header Parsing",
        "description": "Strip whitespaces from Authorization header credentials in fastapi/security/utils.py",
        "target_file": "fastapi/security/utils.py",
        "ground_truth_lines": 8,
    },
    {
        "instance_id": "requests_7315",
        "repo": "psf/requests",
        "tier": "Tier 2: Targeted Protocol Fix",
        "complexity_level": "Low-Med (11 lines)",
        "domain": "HTTP / URL Path Parsing",
        "description": "Preserve leading slashes in request path_url in src/requests/models.py",
        "target_file": "src/requests/models.py",
        "ground_truth_lines": 11,
    },
    {
        "instance_id": "rich_4077",
        "repo": "Textualize/rich",
        "tier": "Tier 3: Stream I/O & Delegation",
        "complexity_level": "Medium (9 lines)",
        "domain": "Terminal / Console Stream",
        "description": "Proxy isatty method on terminal console wrapper in rich/console.py",
        "target_file": "rich/console.py",
        "ground_truth_lines": 9,
    },
    {
        "instance_id": "fastapi_14873",
        "repo": "fastapi/fastapi",
        "tier": "Tier 4: Routing & Lifecycle",
        "complexity_level": "Med-High (37 lines)",
        "domain": "Routing / Lifecycle Hooks",
        "description": "Propagate on_startup and on_shutdown parameters of APIRouter in fastapi/routing.py",
        "target_file": "fastapi/routing.py",
        "ground_truth_lines": 37,
    },
    {
        "instance_id": "fastapi_15661",
        "repo": "fastapi/fastapi",
        "tier": "Tier 5: Macro Architectural Stress-Test",
        "complexity_level": "High (219 lines)",
        "domain": "Multi-Module Architecture",
        "description": "Automate release preparation workflow and multi-module release coordination",
        "target_file": "scripts/prepare_release.py",
        "ground_truth_lines": 219,
    },
]


def load_all_tasks() -> dict[str, dict]:
    with open(TASKS_FILE, "r", encoding="utf-8") as f:
        return {json.loads(line)["instance_id"]: json.loads(line) for line in f if line.strip()}


def run_benchmark_task(
    task_def: dict,
    full_task: dict,
    model: str = "openai/google/gemma-4-E4B-it",
    api_base: str = "http://localhost:8080",
    api_key: str = "EMPTY",
    max_turns: int = 5,
    mock: bool = False,
    direct_generator: Any = None,
) -> dict[str, Any]:
    instance_id = task_def["instance_id"]
    print(f"\n=======================================================")
    print(f" [*] Benchmark Task: {instance_id} [{task_def['tier']}]")
    print(f"     Repo: {task_def['repo']} | Complexity: {task_def['complexity_level']}")
    print(f"     Description: {task_def['description']}")
    print(f"=======================================================")

    t0 = time.time()
    try:
        # 1. Setup workspace with Windows file-lock resilience
        workdir = setup_task_workspace(instance_id, clean=True)

        # 2. Bind workspace tools
        tools = WorkspaceTools(workdir, repo=full_task["repo"], base_commit=full_task["base_commit"], max_turns=max_turns)

        # 3. Execute agent loop (Live or Mock or Direct In-Process)
        agent_patch, loop_stats = run_agent_loop(
            problem_statement=full_task["problem_statement"],
            tools=tools,
            model=model,
            api_base=api_base,
            api_key=api_key,
            max_turns=max_turns,
            mock=mock,
            direct_generator=direct_generator,
        )

        # 4. Save patch
        patch_file = BENCHMARK_DIR / f"{instance_id}.patch"
        patch_file.write_text(agent_patch, encoding="utf-8")

        # 5. Grade against ground truth
        score = grade_agent_patch(full_task, agent_patch, workdir, run_pytest=True)
        score.update(loop_stats)
        score.update({
            "tier": task_def["tier"],
            "complexity_level": task_def["complexity_level"],
            "domain": task_def["domain"],
            "ground_truth_lines": task_def["ground_truth_lines"],
            "status": "PASS" if score.get("pytest_resolved") else ("LOCALIZED" if score.get("target_matched") else "ATTEMPTED"),
        })
        return score

    except Exception as e:
        print(f"[ERROR] Task {instance_id} failed with error: {e}")
        return {
            "instance_id": instance_id,
            "tier": task_def["tier"],
            "complexity_level": task_def["complexity_level"],
            "domain": task_def["domain"],
            "ground_truth_lines": task_def["ground_truth_lines"],
            "target_matched": False,
            "syntax_valid": False,
            "pytest_resolved": False,
            "patch_size_bytes": 0,
            "turns_used": 0,
            "total_tokens": 0,
            "duration_sec": round(time.time() - t0, 2),
            "tool_counts": {},
            "status": "ERROR",
            "error_message": str(e),
        }


def generate_html_dashboard(all_runs: list[dict], output_html: Path):
    """Renders a self-contained HTML visual dashboard with Tailwind and charts."""
    dashboard_template = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SWE-Gemma Autonomous Developer Agent - Benchmark Dashboard</title>
<script src="https://cdn.tailwindcss.com"></script>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background-color: #0f172a; color: #f8fafc; }
  .badge-pass { background-color: #10b981; color: #ffffff; }
  .badge-localized { background-color: #3b82f6; color: #ffffff; }
  .badge-attempted { background-color: #f59e0b; color: #ffffff; }
  .badge-error { background-color: #ef4444; color: #ffffff; }
</style>
</head>
<body class="p-6 md:p-10">
<div class="max-w-7xl mx-auto space-y-8">

  <!-- Header -->
  <div class="flex flex-col md:flex-row md:items-center justify-between pb-6 border-b border-slate-700">
    <div>
      <h1 class="text-3xl font-extrabold tracking-tight text-white flex items-center gap-3">
        <span>🚀 Gemma 4 Developer Agent</span>
        <span class="text-xs uppercase tracking-wider font-semibold px-2.5 py-1 rounded bg-indigo-600 text-white">Weekly Eval Suite</span>
      </h1>
      <p class="text-sm text-slate-400 mt-1">Multi-Tier Benchmark Dashboard across 5 Complexity Spectrum Tasks</p>
    </div>
    <div class="mt-4 md:mt-0 text-right">
      <span class="text-xs text-slate-400">Competition Target:</span>
      <span class="block text-sm font-bold text-indigo-400">Google - The Gemma 4 Developer Agent Challenge (2026)</span>
    </div>
  </div>

  <!-- Run Selector Cards / High Level Overview -->
  <div id="runs-container" class="grid grid-cols-1 md:grid-cols-4 gap-4">
    <!-- Populated by JS -->
  </div>

  <!-- Visual Charts Section -->
  <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
    <div class="bg-slate-800/80 p-5 rounded-xl border border-slate-700 shadow-lg">
      <h3 class="text-base font-semibold text-slate-200 mb-4 flex items-center justify-between">
        <span>Resolution Rate & Localization (%)</span>
        <span class="text-xs text-slate-400">Week-over-Week</span>
      </h3>
      <div class="h-64">
        <canvas id="accuracyChart"></canvas>
      </div>
    </div>

    <div class="bg-slate-800/80 p-5 rounded-xl border border-slate-700 shadow-lg">
      <h3 class="text-base font-semibold text-slate-200 mb-4 flex items-center justify-between">
        <span>Efficiency Metrics (Avg Turns & Tokens)</span>
        <span class="text-xs text-slate-400">Context Optimization</span>
      </h3>
      <div class="h-64">
        <canvas id="efficiencyChart"></canvas>
      </div>
    </div>
  </div>

  <!-- Detailed Tasks Table for Latest Run -->
  <div class="bg-slate-800/80 rounded-xl border border-slate-700 shadow-lg overflow-hidden">
    <div class="p-5 border-b border-slate-700 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
      <div>
        <h3 class="text-lg font-bold text-white flex items-center gap-2">
          <span>Task Breakdown by Complexity Tier</span>
          <span id="active-run-badge" class="text-xs font-normal px-2 py-0.5 rounded bg-slate-700 text-slate-300"></span>
        </h3>
        <p class="text-xs text-slate-400 mt-0.5">Evaluation across distinct functional layers and code size tiers</p>
      </div>
    </div>

    <div class="overflow-x-auto">
      <table class="w-full text-left text-sm text-slate-300">
        <thead class="text-xs uppercase bg-slate-900/60 text-slate-400 border-b border-slate-700">
          <tr>
            <th class="px-5 py-3">Task ID & Repository</th>
            <th class="px-4 py-3">Complexity Tier</th>
            <th class="px-4 py-3">Target Match</th>
            <th class="px-4 py-3">AST Syntax</th>
            <th class="px-4 py-3">Pytest Resolved</th>
            <th class="px-4 py-3">Turns</th>
            <th class="px-4 py-3">Tokens</th>
            <th class="px-4 py-3">Latency</th>
            <th class="px-4 py-3 text-right">Status</th>
          </tr>
        </thead>
        <tbody id="tasks-table-body" class="divide-y divide-slate-700/60 font-mono text-xs">
          <!-- Populated by JS -->
        </tbody>
      </table>
    </div>
  </div>

</div>

<script>
  const BENCHMARK_DATA = __BENCHMARK_DATA_PLACEHOLDER__;

  function renderDashboard() {
    if (!BENCHMARK_DATA || BENCHMARK_DATA.length === 0) return;
    const latest = BENCHMARK_DATA[BENCHMARK_DATA.length - 1];

    // 1. Render Summary Cards
    const cardsContainer = document.getElementById("runs-container");
    cardsContainer.innerHTML = `
      <div class="bg-slate-800 p-5 rounded-xl border border-slate-700">
        <span class="text-xs text-slate-400 uppercase tracking-wider font-semibold">Active Run ID</span>
        <div class="text-xl font-bold text-indigo-400 mt-1">${latest.run_id}</div>
        <div class="text-xs text-slate-500 mt-1">${latest.timestamp} | ${latest.model}</div>
      </div>
      <div class="bg-slate-800 p-5 rounded-xl border border-slate-700">
        <span class="text-xs text-slate-400 uppercase tracking-wider font-semibold">Pass@1 Resolution Rate</span>
        <div class="text-2xl font-bold text-emerald-400 mt-1">${latest.metrics.pass_rate_pct.toFixed(1)}%</div>
        <div class="text-xs text-slate-400 mt-1">${latest.metrics.resolved_count} of ${latest.metrics.total_tasks} tasks passed</div>
      </div>
      <div class="bg-slate-800 p-5 rounded-xl border border-slate-700">
        <span class="text-xs text-slate-400 uppercase tracking-wider font-semibold">Localization Accuracy</span>
        <div class="text-2xl font-bold text-blue-400 mt-1">${latest.metrics.localization_rate_pct.toFixed(1)}%</div>
        <div class="text-xs text-slate-400 mt-1">${latest.metrics.localized_count} of ${latest.metrics.total_tasks} files matched</div>
      </div>
      <div class="bg-slate-800 p-5 rounded-xl border border-slate-700">
        <span class="text-xs text-slate-400 uppercase tracking-wider font-semibold">AST Syntax Validity</span>
        <div class="text-2xl font-bold text-purple-400 mt-1">${latest.metrics.syntax_valid_pct.toFixed(1)}%</div>
        <div class="text-xs text-slate-400 mt-1">Avg ${latest.metrics.avg_tokens.toLocaleString()} tokens/task</div>
      </div>
    `;

    document.getElementById("active-run-badge").innerText = `${latest.run_id} (${latest.model})`;

    // 2. Render Table Rows
    const tbody = document.getElementById("tasks-table-body");
    tbody.innerHTML = latest.tasks.map(t => {
      let badgeClass = "badge-attempted";
      if (t.status === "PASS") badgeClass = "badge-pass";
      else if (t.status === "LOCALIZED") badgeClass = "badge-localized";
      else if (t.status === "ERROR") badgeClass = "badge-error";

      return `
        <tr class="hover:bg-slate-750/50 transition">
          <td class="px-5 py-3 font-semibold text-slate-200">
            <div>${t.instance_id}</div>
            <div class="text-[10px] text-slate-400 font-normal">${t.domain}</div>
          </td>
          <td class="px-4 py-3 text-slate-300">
            <div>${t.tier}</div>
            <div class="text-[10px] text-slate-500">${t.complexity_level}</div>
          </td>
          <td class="px-4 py-3">${t.target_matched ? '<span class="text-emerald-400 font-bold">✓ YES</span>' : '<span class="text-slate-500">✗ NO</span>'}</td>
          <td class="px-4 py-3">${t.syntax_valid ? '<span class="text-emerald-400 font-bold">✓ VALID</span>' : '<span class="text-red-400 font-bold">✗ ERROR</span>'}</td>
          <td class="px-4 py-3">${t.pytest_resolved ? '<span class="text-emerald-400 font-bold">✓ 1.0</span>' : '<span class="text-slate-500">0.0</span>'}</td>
          <td class="px-4 py-3 text-slate-300">${t.turns_used || '-'}</td>
          <td class="px-4 py-3 text-slate-300">${(t.total_tokens || 0).toLocaleString()}</td>
          <td class="px-4 py-3 text-slate-300">${t.duration_sec || 0}s</td>
          <td class="px-4 py-3 text-right">
            <span class="px-2.5 py-1 rounded text-[11px] font-bold ${badgeClass}">${t.status}</span>
          </td>
        </tr>
      `;
    }).join("");

    // 3. Render Charts
    const runLabels = BENCHMARK_DATA.map(r => r.run_id);
    const passRates = BENCHMARK_DATA.map(r => r.metrics.pass_rate_pct);
    const locRates = BENCHMARK_DATA.map(r => r.metrics.localization_rate_pct);
    const avgTurns = BENCHMARK_DATA.map(r => r.metrics.avg_turns);
    const avgTokens = BENCHMARK_DATA.map(r => r.metrics.avg_tokens);

    new Chart(document.getElementById("accuracyChart"), {
      type: "bar",
      data: {
        labels: runLabels,
        datasets: [
          { label: "Pass@1 (%)", data: passRates, backgroundColor: "rgba(16, 185, 129, 0.85)" },
          { label: "Localization Accuracy (%)", data: locRates, backgroundColor: "rgba(59, 130, 246, 0.85)" },
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          y: { beginAtZero: true, max: 100, grid: { color: "#334155" } },
          x: { grid: { display: false } }
        },
        plugins: { legend: { labels: { color: "#cbd5e1" } } }
      }
    });

    new Chart(document.getElementById("efficiencyChart"), {
      type: "line",
      data: {
        labels: runLabels,
        datasets: [
          { label: "Avg Turns", data: avgTurns, yAxisID: 'yTurns', borderColor: "#f59e0b", backgroundColor: "#f59e0b", tension: 0.2 },
          { label: "Avg Tokens", data: avgTokens, yAxisID: 'yTokens', borderColor: "#a855f7", backgroundColor: "#a855f7", tension: 0.2 },
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          yTurns: { type: 'linear', position: 'left', beginAtZero: true, grid: { color: "#334155" }, ticks: { color: "#f59e0b" } },
          yTokens: { type: 'linear', position: 'right', beginAtZero: true, grid: { display: false }, ticks: { color: "#a855f7" } },
          x: { grid: { display: false } }
        },
        plugins: { legend: { labels: { color: "#cbd5e1" } } }
      }
    });
  }

  document.addEventListener("DOMContentLoaded", renderDashboard);
</script>
</body>
</html>
"""
    json_str = json.dumps(all_runs, indent=2)
    rendered = dashboard_template.replace("__BENCHMARK_DATA_PLACEHOLDER__", json_str)
    output_html.write_text(rendered, encoding="utf-8")
    print(f"[*] HTML Dashboard updated: {output_html}")


def run_benchmark_suite(
    run_id: str = "week1_baseline",
    model: str = "openai/google/gemma-4-E4B-it",
    api_base: str = "http://localhost:8080/v1",
    api_key: str = "EMPTY",
    max_turns: int = 3,
    mock: bool = False,
    task_filter: list[str] | None = None,
) -> dict:
    all_tasks = load_all_tasks()
    selected_tasks = CANONICAL_TASKS
    if task_filter:
        selected_tasks = [t for t in CANONICAL_TASKS if t["instance_id"] in task_filter]

    print(f"\n=======================================================")
    print(f"  LAUNCHING BENCHMARK SUITE: {run_id}")
    print(f"  Model: {model} | Base: {api_base}")
    print(f"  Tasks: {len(selected_tasks)} across 5 complexity tiers")
    print(f"  Timestamp: {datetime.datetime.now().isoformat()}")
    print(f"=======================================================")

    results = []
    for t_def in selected_tasks:
        cid = t_def["instance_id"]
        if cid not in all_tasks:
            print(f"[WARN] Task {cid} not in tasks.jsonl, skipping.")
            continue
        task_res = run_benchmark_task(
            task_def=t_def,
            full_task=all_tasks[cid],
            model=model,
            api_base=api_base,
            api_key=api_key,
            max_turns=max_turns,
            mock=mock,
        )
        results.append(task_res)

    # Calculate Aggregate Metrics
    total = len(results)
    resolved = sum(1 for r in results if r.get("pytest_resolved"))
    localized = sum(1 for r in results if r.get("target_matched"))
    syntax_ok = sum(1 for r in results if r.get("syntax_valid"))
    avg_turns = round(sum(r.get("turns_used", 0) for r in results) / max(total, 1), 2)
    avg_tokens = int(sum(r.get("total_tokens", 0) for r in results) / max(total, 1))
    avg_latency = round(sum(r.get("duration_sec", 0.0) for r in results) / max(total, 1), 2)

    run_summary = {
        "run_id": run_id,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "model": model,
        "api_base": api_base,
        "metrics": {
            "total_tasks": total,
            "resolved_count": resolved,
            "pass_rate_pct": round((resolved / max(total, 1)) * 100, 1),
            "localized_count": localized,
            "localization_rate_pct": round((localized / max(total, 1)) * 100, 1),
            "syntax_valid_pct": round((syntax_ok / max(total, 1)) * 100, 1),
            "avg_turns": avg_turns,
            "avg_tokens": avg_tokens,
            "avg_latency_sec": avg_latency,
        },
        "tasks": results,
    }

    # Save Run JSON
    run_json_path = BENCHMARK_DIR / f"{run_id}.json"
    run_json_path.write_text(json.dumps(run_summary, indent=2), encoding="utf-8")
    print(f"\n[*] Benchmark run serialized to: {run_json_path}")

    # Load all historical runs to update dashboard
    all_runs = []
    for f in sorted(BENCHMARK_DIR.glob("*.json")):
        try:
            all_runs.append(json.loads(f.read_text(encoding="utf-8")))
        except Exception:
            pass

    generate_html_dashboard(all_runs, DASHBOARD_DIR / "index.html")

    # Print Summary Table
    print(f"\n=======================================================")
    print(f" BENCHMARK RUN COMPLETE: {run_id}")
    print(f" Pass@1 Resolution Rate: {run_summary['metrics']['pass_rate_pct']}% ({resolved}/{total})")
    print(f" Localization Accuracy:  {run_summary['metrics']['localization_rate_pct']}% ({localized}/{total})")
    print(f" AST Syntax Validity:    {run_summary['metrics']['syntax_valid_pct']}%")
    print(f" Avg Turns / Task:       {avg_turns}")
    print(f" Avg Tokens / Task:      {avg_tokens}")
    print(f"=======================================================\n")

    return run_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run 5-task complexity benchmark suite.")
    parser.add_argument("--run-id", default="week1_baseline", help="Identifier for this benchmark run (e.g. week1_baseline)")
    parser.add_argument("--model", default="openai/google/gemma-4-E4B-it", help="Model name")
    parser.add_argument("--api-base", default="http://localhost:8080/v1", help="API base URL")
    parser.add_argument("--api-key", default="EMPTY", help="API Key")
    parser.add_argument("--max-turns", type=int, default=3, help="Max agent turns per task")
    parser.add_argument("--mock", action="store_true", help="Run in mock/dry-run mode")
    parser.add_argument("--tasks", nargs="*", default=None, help="Filter tasks by instance_id")

    args = parser.parse_args()
    run_benchmark_suite(
        run_id=args.run_id,
        model=args.model,
        api_base=args.api_base,
        api_key=args.api_key,
        max_turns=args.max_turns,
        mock=args.mock,
        task_filter=args.tasks,
    )
