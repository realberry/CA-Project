#!/usr/bin/env python3
"""Combine VM batch/context calibration with real-trace prefetch evaluation."""

from __future__ import annotations

import csv
import json
from copy import deepcopy
from pathlib import Path

from evaluate_hbm_policies import ROOT, csv_rows, evaluate


EXPERIMENTS = ROOT / "configs" / "endsem_experiments.json"
HARDWARE = (
    ROOT
    / "results"
    / "final_endsem"
    / "hardware_matrix_summary"
    / "hardware_matrix_summary.csv"
)
OUTPUT = ROOT / "results" / "final_endsem" / "batch_context_prefetch"


def run() -> tuple[list[dict], dict]:
    experiment = json.loads(EXPERIMENTS.read_text(encoding="utf-8"))
    base = json.loads(
        (ROOT / experiment["base_prefetch_config"]).read_text(encoding="utf-8")
    )
    base["evaluation_tokens"] = int(experiment["sensitivity"]["evaluation_tokens"])
    primary = experiment["primary"]
    base["hbm_capacity_experts"] = int(primary["hbm_capacity_experts"])
    base["candidate_budget"] = int(primary["candidate_budget"])
    base["overlap_multiplier"] = float(primary["overlap_multiplier"])

    with HARDWARE.open(newline="", encoding="utf-8") as handle:
        hardware_rows = list(csv.DictReader(handle))
    output_rows: list[dict] = []
    run_metadata: dict[str, dict] = {}
    for hardware in hardware_rows:
        batch = int(hardware["batch_size"])
        context = int(hardware["context_length"])
        config = deepcopy(base)
        config["routing_batch_size"] = batch
        config["measured_offchip_expert_load_ms"] = float(hardware["expert_load_ms"])
        config["measured_offchip_expert_execution_ms"] = float(
            hardware["offchip_expert_execution_ms"]
        )
        config["measured_offchip_decoder_ms"] = float(
            hardware["no_prefetch_decode_step_ms"]
        )
        rows, metadata = evaluate(config)
        key = f"batch={batch},context={context}"
        run_metadata[key] = metadata
        for row in csv_rows(rows):
            output_rows.append(
                {
                    "batch_size": batch,
                    "context_length": context,
                    **row,
                }
            )
        print(f"completed {key}")
    return output_rows, {
        "hardware_summary": str(HARDWARE.relative_to(ROOT)),
        "routing_tokens_per_case": int(base["evaluation_tokens"]),
        "cases": run_metadata,
    }


def write_svg(rows: list[dict], destination: Path) -> None:
    selected = [row for row in rows if row["scenario"] == "complete_car"]
    batches = sorted({int(row["batch_size"]) for row in selected})
    contexts = sorted({int(row["context_length"]) for row in selected})
    width, height = 1000, 560
    left, top, plot_w, plot_h = 100, 100, 820, 350
    ymin, ymax = 1.0, max(float(row["speedup_vs_baseline"]) for row in selected) * 1.08
    colors = {1: "#e45756", 4: "#2f6fed", 8: "#13a89e"}
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fff"/>',
        '<style>text{font-family:Segoe UI,Arial,sans-serif;fill:#17233c}.title{font-size:25px;font-weight:700}.sub{font-size:14px;fill:#596579}.tick{font-size:12px;fill:#6b7280}.legend{font-size:14px;font-weight:600}</style>',
        '<text x="32" y="42" class="title">Complete prefetch design across batch and context sizes</text>',
        '<text x="32" y="68" class="sub">Real Mixtral routes · matching LLMSimulator calibration · CCT predictor + CAR eviction</text>',
    ]
    for index in range(6):
        value = ymin + (ymax - ymin) * index / 5
        y = top + plot_h * (ymax - value) / (ymax - ymin)
        svg.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left+plot_w}" y2="{y:.1f}" stroke="#e2e7ef"/>')
        svg.append(f'<text x="{left-12}" y="{y+4:.1f}" text-anchor="end" class="tick">{value:.3f}×</text>')
    for index, context in enumerate(contexts):
        x = left + plot_w * index / (len(contexts) - 1)
        svg.append(f'<text x="{x:.1f}" y="475" text-anchor="middle" class="tick">{context}</text>')
    for batch in batches:
        series = sorted((row for row in selected if int(row["batch_size"]) == batch), key=lambda row: int(row["context_length"]))
        points = []
        for index, row in enumerate(series):
            x = left + plot_w * index / (len(series) - 1)
            speedup = float(row["speedup_vs_baseline"])
            y = top + plot_h * (ymax - speedup) / (ymax - ymin)
            points.append(f"{x:.1f},{y:.1f}")
            svg.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{colors[batch]}"/>')
        svg.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{colors[batch]}" stroke-width="4"/>')
        lx = 250 + batches.index(batch) * 220
        svg.append(f'<line x1="{lx}" y1="525" x2="{lx+30}" y2="525" stroke="{colors[batch]}" stroke-width="4"/>')
        svg.append(f'<text x="{lx+38}" y="530" class="legend">batch {batch}</text>')
    svg.append(f'<text x="{left+plot_w/2}" y="500" text-anchor="middle" class="sub">Input context length (tokens)</text>')
    svg.append(f'<text x="28" y="{top+plot_h/2}" text-anchor="middle" class="sub" transform="rotate(-90 28 {top+plot_h/2})">Speedup versus matching no-prefetch LRU</text>')
    svg.append('</svg>')
    destination.write_text("\n".join(svg), encoding="utf-8")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows, metadata = run()
    with (OUTPUT / "batch_context_prefetch.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (OUTPUT / "batch_context_prefetch.json").write_text(
        json.dumps({"metadata": metadata, "scenarios": rows}, indent=2) + "\n",
        encoding="utf-8",
    )
    write_svg(rows, OUTPUT / "batch_context_prefetch.svg")
    print(f"rows={len(rows)}")
    print(f"wrote={OUTPUT}")


if __name__ == "__main__":
    main()
