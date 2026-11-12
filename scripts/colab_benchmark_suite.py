"""Google Colab 5-Task Benchmark Suite for Gemma 4 31B (A100 GPU).

Supports two execution backends:
1. Direct Mode (--backend direct): [RECOMMENDED FOR COLAB]
   Loads google/gemma-4-31B-it-qat-w4a16-ct directly into memory via Hugging Face Transformers.
   Zero network dependencies, no localhost, no background daemon, zero connection errors.
2. Server Mode (--backend vllm):
   Connects to an OpenAI-compatible vLLM inference server daemon at http://127.0.0.1:8000/v1.

Tasks evaluated across 5 complexity tiers:
- Tier 1: fastapi_14786 (Surgical Fix, 8 lines)
- Tier 2: requests_7315 (Targeted Protocol Fix, 11 lines)
- Tier 3: rich_4077 (Stream I/O & Delegation, 9 lines)
- Tier 4: fastapi_14873 (Routing & Lifecycle, 37 lines)
- Tier 5: fastapi_15661 (Macro Architectural Stress-Test, 219 lines)
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from eval.benchmark_suite import (
    CANONICAL_TASKS,
    load_all_tasks,
    run_benchmark_task,
    generate_html_dashboard,
    BENCHMARK_DIR,
    DASHBOARD_DIR,
)
from eval.run_task_e2e import DirectGemmaTransformersBackend


def check_server_health(api_base: str) -> bool:
    import urllib.request
    check_url = api_base.rstrip("/") + "/models"
    try:
        req = urllib.request.Request(check_url, headers={"User-Agent": "HealthCheck"})
        with urllib.request.urlopen(req, timeout=3) as resp:
            return resp.status == 200
    except Exception:
        return False


def run_colab_benchmark(
    run_id: str = "colab_31b_baseline",
    backend: str = "direct",
    model: str = "google/gemma-4-31B-it-qat-w4a16-ct",
    api_base: str = "http://127.0.0.1:8000/v1",
    api_key: str = "EMPTY",
    max_turns: int = 5,
    mock: bool = False,
):
    print(f"\n=======================================================")
    print(f"  GOOGLE COLAB BENCHMARK RUN: {run_id}")
    print(f"  Execution Backend: {backend.upper()}")
    print(f"  Target Model: {model}")
    if backend == "vllm":
        print(f"  Server API URL: {api_base}")
    print(f"  Tasks: 5 Complexity Spectrum Tasks")
    print(f"=======================================================")

    direct_gen = None

    if not mock:
        if backend == "direct":
            print("\n[*] Initializing Direct In-Process Transformers Backend (No localhost/server needed)...")
            raw_model_id = model.removeprefix("openai/")
            direct_gen = DirectGemmaTransformersBackend(model_id=raw_model_id)
        elif backend == "vllm":
            if not check_server_health(api_base):
                # Try fallback to 127.0.0.1 if localhost was passed
                fallback_url = api_base.replace("localhost", "127.0.0.1")
                if check_server_health(fallback_url):
                    api_base = fallback_url
                    print(f" [+] Connected via fallback IP: {api_base}")
                else:
                    print(f"\n[ERROR] Could not connect to vLLM server at '{api_base}'.")
                    print("\nTroubleshooting vLLM in Colab:")
                    print("1. Check if vLLM crashed on startup (e.g. missing HF token, CUDA OOM):")
                    print("   !cat /tmp/vllm.log | tail -n 25")
                    print("2. Check if the process is still running:")
                    print("   !ps aux | grep vllm")
                    print("3. Alternatively, switch to Direct In-Process mode (no server needed):")
                    print("   !python scripts/colab_benchmark_suite.py --backend direct\n")
                    raise ConnectionError(f"Inference server at {api_base} is not reachable.")

    all_tasks = load_all_tasks()
    results = []

    for t_def in CANONICAL_TASKS:
        cid = t_def["instance_id"]
        if cid not in all_tasks:
            print(f"[WARN] Task {cid} not in tasks.jsonl, skipping.")
            continue

        res = run_benchmark_task(
            task_def=t_def,
            full_task=all_tasks[cid],
            model=model if model.startswith("openai/") else f"openai/{model}",
            api_base=api_base,
            api_key=api_key,
            max_turns=max_turns,
            mock=mock,
            direct_generator=direct_gen,
        )
        results.append(res)

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
        "backend": backend,
        "model": model,
        "api_base": api_base if backend == "vllm" else "in-process",
        "environment": "Google Colab (A100 80GB)",
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

    out_file = BENCHMARK_DIR / f"{run_id}.json"
    out_file.write_text(json.dumps(run_summary, indent=2), encoding="utf-8")
    print(f"\n[+] Colab benchmark report saved: {out_file}")

    all_runs = []
    for f in sorted(BENCHMARK_DIR.glob("*.json")):
        try:
            all_runs.append(json.loads(f.read_text(encoding="utf-8")))
        except Exception:
            pass

    generate_html_dashboard(all_runs, DASHBOARD_DIR / "index.html")
    print(f"[+] Dashboard synchronized: {DASHBOARD_DIR / 'index.html'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Colab Benchmark Suite for Gemma 4 31B")
    parser.add_argument("--run-id", default="colab_31b_baseline", help="Benchmark Run ID")
    parser.add_argument(
        "--backend",
        choices=["direct", "vllm"],
        default="direct",
        help="Inference backend: 'direct' (Hugging Face Transformers in-process) or 'vllm' (HTTP server)",
    )
    parser.add_argument("--model", default="google/gemma-4-31B-it-qat-w4a16-ct", help="Model name or Hugging Face repo")
    parser.add_argument("--api-base", default="http://127.0.0.1:8000/v1", help="vLLM server endpoint")
    parser.add_argument("--api-key", default="EMPTY", help="API key")
    parser.add_argument("--max-turns", type=int, default=5, help="Max turns per task")
    parser.add_argument("--mock", action="store_true", help="Run in mock/dry-run mode")
    args = parser.parse_args()

    run_colab_benchmark(
        run_id=args.run_id,
        backend=args.backend,
        model=args.model,
        api_base=args.api_base,
        api_key=args.api_key,
        max_turns=args.max_turns,
        mock=args.mock,
    )
