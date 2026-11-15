#!/usr/bin/env python3
"""Run organized HBM-capacity, aggressiveness, and lookahead sensitivity."""

from __future__ import annotations

import csv
import json
from copy import deepcopy
from pathlib import Path

from evaluate_hbm_policies import ROOT, csv_rows, evaluate


EXPERIMENTS = ROOT / "configs" / "endsem_experiments.json"
OUTPUT = ROOT / "results" / "final_endsem" / "sensitivity"


def run() -> tuple[list[dict], dict]:
    experiment = json.loads(EXPERIMENTS.read_text(encoding="utf-8"))
    base_path = ROOT / experiment["base_prefetch_config"]
    base = json.loads(base_path.read_text(encoding="utf-8"))
    settings = experiment["sensitivity"]
    base["evaluation_tokens"] = int(settings["evaluation_tokens"])
    default = experiment["primary"]

    requested: list[tuple[str, float, tuple[int, int, float]]] = []
    for capacity in settings["hbm_capacity_experts"]:
        requested.append(("hbm_capacity_experts", capacity, (int(capacity), int(default["candidate_budget"]), float(default["overlap_multiplier"]))))
    for budget in settings["candidate_budget"]:
        requested.append(("candidate_budget", budget, (int(default["hbm_capacity_experts"]), int(budget), float(default["overlap_multiplier"]))))
    for overlap in settings["overlap_multiplier"]:
        requested.append(("overlap_multiplier", overlap, (int(default["hbm_capacity_experts"]), int(default["candidate_budget"]), float(overlap))))

    cache: dict[tuple[int, int, float], list[dict]] = {}
    metadata_by_key: dict[str, dict] = {}
    output_rows: list[dict] = []
    for dimension, value, key in requested:
        if key not in cache:
            capacity, budget, overlap = key
            config = deepcopy(base)
            config["hbm_capacity_experts"] = capacity
            config["candidate_budget"] = budget
            config["overlap_multiplier"] = overlap
            rows, metadata = evaluate(config)
            cache[key] = csv_rows(rows)
            metadata_by_key[f"capacity={capacity},budget={budget},overlap={overlap:g}"] = metadata
            print(f"completed capacity={capacity} budget={budget} overlap={overlap:g}")
        for row in cache[key]:
            if row["scenario"] not in ("baseline_lru", "prefetch_lru", "complete_car"):
                continue
            output_rows.append(
                {
                    "dimension": dimension,
                    "value": value,
                    **row,
                }
            )
    return output_rows, {
        "experiment_config": str(EXPERIMENTS.relative_to(ROOT)),
        "evaluation_tokens_per_unique_run": int(settings["evaluation_tokens"]),
        "unique_runs": len(cache),
        "runs": metadata_by_key,
    }


def write_svg(rows: list[dict], destination: Path) -> None:
    width, height = 1100, 590
    dimensions = (
        ("hbm_capacity_experts", "HBM capacity (experts)"),
        ("candidate_budget", "Prefetch candidates"),
        ("overlap_multiplier", "Overlap multiplier"),
    )
    colors = {"prefetch_lru": "#2f6fed", "complete_car": "#13a89e"}
    labels = {"prefetch_lru": "CCT + LRU", "complete_car": "CCT + CAR"}
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fff"/>',
        '<style>text{font-family:Segoe UI,Arial,sans-serif;fill:#17233c}.title{font-size:25px;font-weight:700}.sub{font-size:14px;fill:#596579}.panel{font-size:16px;font-weight:700}.tick{font-size:12px;fill:#6b7280}.legend{font-size:14px;font-weight:600}</style>',
        '<text x="32" y="42" class="title">Prefetch sensitivity: capacity, aggressiveness, and overlap</text>',
        '<text x="32" y="68" class="sub">10,000 held-out Mixtral tokens per unique setting · all other parameters held at the primary value</text>',
    ]
    panel_width = 310
    panel_gap = 35
    top, plot_h = 125, 320
    plotted = [
        float(row["speedup_vs_baseline"])
        for row in rows
        if row["scenario"] in ("prefetch_lru", "complete_car")
    ]
    ymin = min(0.98, min(plotted) * 0.98)
    ymax = max(plotted) * 1.05
    for panel_index, (dimension, title) in enumerate(dimensions):
        left = 85 + panel_index * (panel_width + panel_gap)
        values = sorted({float(row["value"]) for row in rows if row["dimension"] == dimension})
        svg.append(f'<text x="{left+panel_width/2}" y="102" text-anchor="middle" class="panel">{title}</text>')
        for tick_index in range(5):
            tick = ymin + (ymax - ymin) * tick_index / 4
            y = top + plot_h * (ymax - tick) / (ymax - ymin)
            svg.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left+panel_width}" y2="{y:.1f}" stroke="#e2e7ef"/>')
            if panel_index == 0:
                svg.append(f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end" class="tick">{tick:.2f}×</text>')
        for index, value in enumerate(values):
            x = left + panel_width * index / max(1, len(values) - 1)
            shown = f"{value:g}×" if dimension == "overlap_multiplier" else f"{value:g}"
            svg.append(f'<text x="{x:.1f}" y="470" text-anchor="middle" class="tick">{shown}</text>')
        for scenario in ("prefetch_lru", "complete_car"):
            series = sorted(
                (row for row in rows if row["dimension"] == dimension and row["scenario"] == scenario),
                key=lambda row: float(row["value"]),
            )
            points = []
            for index, row in enumerate(series):
                x = left + panel_width * index / max(1, len(series) - 1)
                speedup = float(row["speedup_vs_baseline"])
                y = top + plot_h * (ymax - speedup) / (ymax - ymin)
                points.append(f"{x:.1f},{y:.1f}")
                svg.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{colors[scenario]}"/>')
            svg.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{colors[scenario]}" stroke-width="4"/>')
    for index, scenario in enumerate(("prefetch_lru", "complete_car")):
        x = 390 + index * 250
        svg.append(f'<line x1="{x}" y1="535" x2="{x+32}" y2="535" stroke="{colors[scenario]}" stroke-width="4"/>')
        svg.append(f'<text x="{x+42}" y="540" class="legend">{labels[scenario]}</text>')
    svg.append('<text x="30" y="290" text-anchor="middle" class="sub" transform="rotate(-90 30 290)">Speedup versus matching no-prefetch LRU</text>')
    svg.append('</svg>')
    destination.write_text("\n".join(svg), encoding="utf-8")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows, metadata = run()
    with (OUTPUT / "endsem_sensitivity.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (OUTPUT / "endsem_sensitivity.json").write_text(
        json.dumps({"metadata": metadata, "scenarios": rows}, indent=2) + "\n",
        encoding="utf-8",
    )
    write_svg(rows, OUTPUT / "endsem_sensitivity.svg")
    print(f"rows={len(rows)}")
    print(f"wrote={OUTPUT}")


if __name__ == "__main__":
    main()
