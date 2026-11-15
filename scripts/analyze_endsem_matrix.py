#!/usr/bin/env python3
"""Summarize the end-semester LLMSimulator batch/context calibration matrix."""

from __future__ import annotations

import csv
import json
import re
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = ROOT / "results" / "final_endsem" / "hardware_matrix"
OUTPUT = ROOT / "results" / "final_endsem" / "hardware_matrix_summary"
CASE_PATTERN = re.compile(r"batch(?P<batch>\d+)_context(?P<context>\d+)")
COMPUTE_ENERGY_COLUMNS = (
    "fc_dram", "fc_comp", "attn_dram", "attn_comp", "ffn_dram",
    "ffn_comp", "shared_moe_hbm", "shared_moe_comp", "moe_hbm", "moe_comp",
)


def read_one(raw: Path, pattern: str) -> tuple[Path, list[dict[str, str]]]:
    matches = list(raw.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {pattern} in {raw}, found {len(matches)}")
    with matches[0].open(newline="", encoding="utf-8") as handle:
        return matches[0], list(csv.DictReader(handle))


def mean_ms(rows: list[dict[str, str]], field: str) -> float:
    return statistics.fmean(float(row[field]) for row in rows) / 1e6


def analyze_case(case_root: Path) -> dict:
    match = CASE_PATTERN.fullmatch(case_root.name)
    if match is None:
        raise RuntimeError(f"Unexpected case directory: {case_root}")
    batch = int(match.group("batch"))
    context = int(match.group("context"))
    raw = case_root / "raw"
    if not raw.is_dir() and batch == 1 and context == 1024:
        # This matrix point is byte-identical to the canonical final single-GPU
        # calibration, so the artifact stores one copy only.
        raw = ROOT / "results" / "final_day2" / "single_gpu_batch1" / "raw"
    baseline_path, baseline = read_one(raw, "*baseline*GPU_compute.csv")
    compute_path, compute = read_one(raw, "*CPU_memory*GPU_compute.csv")
    load_path, load = read_one(raw, "*offloaded_weight_copy.csv")
    if not (len(baseline) == len(compute) == len(load)):
        raise RuntimeError(f"Mismatched samples in {case_root}")
    samples = len(baseline)
    resident_decoder = mean_ms(baseline, "MoE_decoders")
    offchip_decoder_compute = mean_ms(compute, "MoE_decoders")
    load_ms = mean_ms(load, "expert_ffn")
    decode_step_ms = offchip_decoder_compute + load_ms
    compute_j = statistics.fmean(
        sum(float(row[column]) for column in COMPUTE_ENERGY_COLUMNS)
        for row in compute
    ) / 1e9
    offload_j = statistics.fmean(float(row["moe_offload"]) for row in load) / 1e9
    return {
        "batch_size": batch,
        "context_length": context,
        "samples": samples,
        "resident_expert_execution_ms": mean_ms(baseline, "expert_ffn"),
        "offchip_expert_execution_ms": mean_ms(compute, "expert_ffn"),
        "expert_load_ms": load_ms,
        "resident_decode_step_ms": resident_decoder,
        "no_prefetch_decode_step_ms": decode_step_ms,
        "normalized_latency_ms_per_sequence": decode_step_ms / batch,
        "compute_dynamic_j_per_step": compute_j,
        "offload_dynamic_j_per_step": offload_j,
        "total_dynamic_j_per_step": compute_j + offload_j,
        "dynamic_j_per_sequence": (compute_j + offload_j) / batch,
        "baseline_file": baseline_path.name,
        "compute_file": compute_path.name,
        "load_file": load_path.name,
    }


def write_svg(rows: list[dict], destination: Path) -> None:
    width, height = 1000, 540
    left, top, plot_w, plot_h = 100, 100, 820, 330
    batches = sorted({int(row["batch_size"]) for row in rows})
    contexts = sorted({int(row["context_length"]) for row in rows})
    ymax = max(float(row["normalized_latency_ms_per_sequence"]) for row in rows) * 1.12
    colors = {1: "#e45756", 4: "#2f6fed", 8: "#13a89e"}
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fff"/>',
        '<style>text{font-family:Segoe UI,Arial,sans-serif;fill:#17233c}.title{font-size:25px;font-weight:700}.sub{font-size:14px;fill:#596579}.tick{font-size:12px;fill:#6b7280}.legend{font-size:14px;font-weight:600}</style>',
        '<text x="32" y="42" class="title">Batch and context-length sensitivity</text>',
        '<text x="32" y="68" class="sub">Single H100 · FP16 · unsharded experts · no-prefetch LLMSimulator calibration</text>',
    ]
    for tick_index in range(6):
        value = ymax * tick_index / 5
        y = top + plot_h * (1.0 - value / ymax)
        svg.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left+plot_w}" y2="{y:.1f}" stroke="#e2e7ef"/>')
        svg.append(f'<text x="{left-12}" y="{y+4:.1f}" text-anchor="end" class="tick">{value:.1f}</text>')
    for index, context in enumerate(contexts):
        x = left + plot_w * index / (len(contexts) - 1)
        svg.append(f'<text x="{x:.1f}" y="{top+plot_h+25}" text-anchor="middle" class="tick">{context}</text>')
    for batch in batches:
        series = sorted((row for row in rows if int(row["batch_size"]) == batch), key=lambda row: int(row["context_length"]))
        points = []
        for index, row in enumerate(series):
            x = left + plot_w * index / (len(series) - 1)
            value = float(row["normalized_latency_ms_per_sequence"])
            y = top + plot_h * (1.0 - value / ymax)
            points.append(f"{x:.1f},{y:.1f}")
            svg.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{colors[batch]}"/>')
        svg.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{colors[batch]}" stroke-width="4"/>')
        lx = 250 + batches.index(batch) * 220
        svg.append(f'<line x1="{lx}" y1="490" x2="{lx+30}" y2="490" stroke="{colors[batch]}" stroke-width="4"/>')
        svg.append(f'<text x="{lx+38}" y="495" class="legend">batch {batch}</text>')
    svg.append(f'<text x="{left+plot_w/2}" y="475" text-anchor="middle" class="sub">Input context length (tokens)</text>')
    svg.append(f'<text x="28" y="{top+plot_h/2}" text-anchor="middle" class="sub" transform="rotate(-90 28 {top+plot_h/2})">Normalized decode-step latency per sequence (ms)</text>')
    svg.append('</svg>')
    destination.write_text("\n".join(svg), encoding="utf-8")


def main() -> None:
    cases = sorted(
        (path for path in RAW_ROOT.iterdir() if path.is_dir() and CASE_PATTERN.fullmatch(path.name)),
        key=lambda path: (
            int(CASE_PATTERN.fullmatch(path.name).group("batch")),
            int(CASE_PATTERN.fullmatch(path.name).group("context")),
        ),
    )
    if not cases:
        raise SystemExit(f"No hardware-matrix cases found under {RAW_ROOT}")
    rows = [analyze_case(case) for case in cases]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / "hardware_matrix_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (OUTPUT / "hardware_matrix_summary.json").write_text(
        json.dumps({"cases": rows}, indent=2) + "\n", encoding="utf-8"
    )
    write_svg(rows, OUTPUT / "batch_context_sensitivity.svg")
    print(f"cases={len(rows)}")
    print(f"wrote={OUTPUT}")


if __name__ == "__main__":
    main()
