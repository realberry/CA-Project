#!/usr/bin/env python3
"""Single entry point for checking, presenting, or reproducing the project."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRACE = ROOT / "data" / "day3" / "c4_results.first24MiB.jsonl.part"

REQUIRED = (
    "README.md",
    "configs/endsem_experiments.json",
    "results/day2/summary.csv",
    "results/day3/summary.json",
    "results/final_endsem/hbm_policies/hbm_policy_summary.csv",
    "results/final_endsem/sensitivity/endsem_sensitivity.csv",
    "results/final_endsem/hardware_matrix_summary/hardware_matrix_summary.csv",
    "results/final_endsem/batch_context_prefetch/batch_context_prefetch.csv",
)

STAGES = {
    "mid": (
        ("scripts/analyze_day2.py",),
        ("scripts/analyze_day3.py",),
    ),
    "primary": (
        ("scripts/analyze_single_gpu.py",),
        ("scripts/evaluate_hbm_policies.py",),
    ),
    "matrix": (
        ("scripts/analyze_endsem_matrix.py",),
        ("scripts/analyze_endsem_batch_context.py",),
    ),
    "sensitivity": (("scripts/analyze_endsem_sensitivity.py",),),
}


def read_csv(path: str) -> list[dict[str, str]]:
    with (ROOT / path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def percent(value: str | float) -> str:
    return f"{100.0 * float(value):.2f}%"


def check() -> bool:
    missing = [path for path in REQUIRED if not (ROOT / path).is_file()]
    for path in REQUIRED:
        state = "OK" if (ROOT / path).is_file() else "MISSING"
        print(f"[{state:7s}] {path}")
    if TRACE.is_file():
        print(f"[OK     ] {TRACE.relative_to(ROOT)} (external trace present)")
    else:
        print("[INFO   ] external trace absent; run scripts/fetch_mixtral_trace.sh before regeneration")
    return not missing


def summary() -> None:
    day2 = read_csv("results/day2/summary.csv")
    day2_by_metric = {row["metric"]: row for row in day2}
    day3 = json.loads((ROOT / "results/day3/summary.json").read_text(encoding="utf-8"))
    primary = {
        row["scenario"]: row
        for row in read_csv("results/final_endsem/hbm_policies/hbm_policy_summary.csv")
    }
    batch_rows = [
        row
        for row in read_csv(
            "results/final_endsem/batch_context_prefetch/batch_context_prefetch.csv"
        )
        if row["scenario"] in ("complete_car", "adaptive_complete_car")
    ]
    cct_complete = primary["complete_car"]
    adaptive_complete = primary["adaptive_complete_car"]

    print("\nMID-SEMESTER EVIDENCE")
    print(
        "  Memory wall: "
        f'{float(day2_by_metric["offchip_expert_loading"]["mean_ms"]):.3f} ms loading '
        f'vs {float(day2_by_metric["offchip_expert_execution"]["mean_ms"]):.3f} ms execution'
    )
    print(
        "  Real-route prediction: "
        f'{100*float(day3["spatial_recall"]):.2f}% CCT recall; '
        f'{100*float(day3["temporal_recall"]):.2f}% previous-token recall'
    )

    print("\nEND-SEMESTER PRIMARY RESULT")
    print(
        f'  Baseline: {float(primary["baseline_lru"]["mean_token_latency_ms"]):.3f} ms'
    )
    print(
        f'  CCT+CAR: {float(cct_complete["mean_token_latency_ms"]):.3f} ms, '
        f'{float(cct_complete["speedup_vs_baseline"]):.3f}x, '
        f'{percent(cct_complete["prediction_recall"])} recall'
    )
    print(
        f'  Adaptive HT-CCT+CAR: {float(adaptive_complete["mean_token_latency_ms"]):.3f} ms, '
        f'{float(adaptive_complete["speedup_vs_baseline"]):.3f}x, '
        f'{percent(adaptive_complete["prediction_recall"])} recall'
    )

    print("\nBATCH / CONTEXT SENSITIVITY (COMPLETE DESIGN)")
    for batch in sorted({int(row["batch_size"]) for row in batch_rows}):
        cct = [
            row for row in batch_rows
            if int(row["batch_size"]) == batch and row["scenario"] == "complete_car"
        ]
        adaptive = [
            row for row in batch_rows
            if int(row["batch_size"]) == batch
            and row["scenario"] == "adaptive_complete_car"
        ]
        cct_speedups = [float(row["speedup_vs_baseline"]) for row in cct]
        adaptive_speedups = [float(row["speedup_vs_baseline"]) for row in adaptive]
        print(
            f'  batch={batch}: CCT {min(cct_speedups):.3f}-{max(cct_speedups):.3f}x; '
            f'adaptive {min(adaptive_speedups):.3f}-{max(adaptive_speedups):.3f}x'
        )


def run_stage(stage: str) -> None:
    stages = tuple(STAGES) if stage == "all" else (stage,)
    if any(name in ("mid", "primary", "sensitivity", "matrix") for name in stages) and not TRACE.is_file():
        # Matrix hardware aggregation alone does not need the trace, but its
        # combined prefetch evaluation does.
        raise SystemExit(
            "External trace is missing. Run `bash scripts/fetch_mixtral_trace.sh` first."
        )
    for name in stages:
        for command in STAGES[name]:
            shown = " ".join(command)
            print(f"\n== {name}: {shown} ==", flush=True)
            subprocess.run([sys.executable, *command], cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("check", help="verify canonical demo artifacts")
    subparsers.add_parser("summary", help="print the concise demo results")
    run_parser = subparsers.add_parser("run", help="regenerate one analysis stage")
    run_parser.add_argument("stage", choices=(*STAGES, "all"))
    args = parser.parse_args()
    if args.command == "check":
        raise SystemExit(0 if check() else 1)
    if args.command == "summary":
        summary()
        return
    run_stage(args.stage)


if __name__ == "__main__":
    main()
