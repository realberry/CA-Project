#!/usr/bin/env python3
"""Evaluate prefetch buffer, lead-time, stall, traffic, and energy sensitivity."""

from __future__ import annotations

import csv
import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

from simulate_prefetch import (
    build_profile,
    cct_scores,
    iter_sequences,
    top_indices,
    trace_prefix_sha256,
)


ROOT = Path(__file__).resolve().parents[1]
BUFFER_SLOTS = (1, 2, 3, 4)
OVERLAP_MULTIPLIERS = (0.5, 1.0, 2.0, 4.0)
STALL_POWERS_W = (0, 100, 200, 350, 500, 700)
OFFLOAD_BANDWIDTH_BYTES_S = 450e9


@dataclass
class Scenario:
    policy: str
    slots: int
    overlap_multiplier: float
    actual: int = 0
    predicted: int = 0
    hits: int = 0
    useful_ms: float = 0.0
    wasted_ms: float = 0.0
    latency_sum_ms: float = 0.0
    latency_square_sum_ms: float = 0.0


def measured_dynamic_energy_j(config: dict) -> tuple[float, float]:
    raw = ROOT / config["energy_raw_directory"]
    compute_matches = list(raw.glob("*CPU_memory*GPU_compute.csv"))
    load_matches = list(raw.glob("*offloaded_weight_copy.csv"))
    if len(compute_matches) != 1 or len(load_matches) != 1:
        raise RuntimeError(
            f"Expected one compute and one load energy CSV in {raw}; "
            f"found {len(compute_matches)} and {len(load_matches)}"
        )
    compute_path = compute_matches[0]
    load_path = load_matches[0]
    compute_columns = (
        "fc_dram", "fc_comp", "attn_dram", "attn_comp", "ffn_dram",
        "ffn_comp", "shared_moe_hbm", "shared_moe_comp", "moe_hbm", "moe_comp",
    )
    with compute_path.open(newline="", encoding="utf-8") as handle:
        compute_rows = list(csv.DictReader(handle))
    with load_path.open(newline="", encoding="utf-8") as handle:
        load_rows = list(csv.DictReader(handle))
    compute_j = sum(
        sum(float(row[column]) for column in compute_columns)
        for row in compute_rows
    ) / len(compute_rows) / 1e9
    offload_j = sum(float(row["moe_offload"]) for row in load_rows) / len(load_rows) / 1e9
    return compute_j, offload_j


def run_grid(config: dict):
    trace = ROOT / config["trace"]
    profile_tokens = int(config["profile_tokens"])
    evaluation_tokens = int(config["evaluation_tokens"])
    total_tokens = profile_tokens + evaluation_tokens
    layers = int(config["layers"])
    experts = int(config["experts_per_layer"])
    top_k = int(config["top_k"])
    total_load_ms = float(config["measured_offchip_expert_load_ms"])
    total_execution_ms = float(config["measured_offchip_expert_execution_ms"])
    decoder_ms = float(config["measured_offchip_decoder_ms"])
    non_expert_ms = decoder_ms - total_load_ms - total_execution_ms
    load_per_expert_ms = total_load_ms / (layers * top_k)
    execution_per_layer_ms = total_execution_ms / layers

    cct, layer_frequency = build_profile(trace, profile_tokens, layers, experts)
    scenarios = {
        (policy, slots, multiplier): Scenario(policy, slots, multiplier)
        for policy in ("cct", "oracle")
        for slots in BUFFER_SLOTS
        for multiplier in OVERLAP_MULTIPLIERS
    }
    evaluated = 0

    for offset, routes in iter_sequences(trace, total_tokens):
        start = max(0, profile_tokens - offset)
        if start >= len(routes):
            continue
        for position in range(start, len(routes)):
            token = routes[position]
            token_useful = {key: 0.0 for key in scenarios}
            for layer in range(layers):
                actual = {token[0][layer], token[1][layer]}
                scores = cct_scores(cct, layer, token, layer_frequency, experts)
                ranking = top_indices(scores, max(BUFFER_SLOTS))
                for key, scenario in scenarios.items():
                    policy, slots, multiplier = key
                    if policy == "cct":
                        predicted = set(ranking[:slots])
                    else:
                        predicted = set(sorted(actual)[: min(slots, len(actual))])
                    transfer_budget_ms = execution_per_layer_ms * multiplier
                    progress_ms = min(
                        load_per_expert_ms,
                        transfer_budget_ms / len(predicted),
                    )
                    hits = len(actual & predicted)
                    wrong = len(predicted - actual)
                    scenario.actual += len(actual)
                    scenario.predicted += len(predicted)
                    scenario.hits += hits
                    scenario.useful_ms += hits * progress_ms
                    scenario.wasted_ms += wrong * progress_ms
                    token_useful[key] += hits * progress_ms
            for key, scenario in scenarios.items():
                latency_ms = decoder_ms - token_useful[key]
                scenario.latency_sum_ms += latency_ms
                scenario.latency_square_sum_ms += latency_ms * latency_ms
            evaluated += 1

    if evaluated != evaluation_tokens:
        raise RuntimeError(f"Evaluated {evaluated}, expected {evaluation_tokens}")

    compute_dynamic_j, offload_dynamic_j = measured_dynamic_energy_j(config)
    baseline_dynamic_j = compute_dynamic_j + offload_dynamic_j
    rows = []
    energy_rows = []
    for key in sorted(scenarios, key=lambda k: (k[0], k[1], k[2])):
        scenario = scenarios[key]
        mean = scenario.latency_sum_ms / evaluated
        variance = max(0.0, scenario.latency_square_sum_ms / evaluated - mean * mean)
        useful_ms = scenario.useful_ms / evaluated
        wasted_ms = scenario.wasted_ms / evaluated
        exposed_stall_ms = total_load_ms - useful_ms
        wasted_fraction = wasted_ms / total_load_ms
        dynamic_j = compute_dynamic_j + offload_dynamic_j * (1.0 + wasted_fraction)
        latency_saved_s = (decoder_ms - mean) / 1000.0
        break_even_w = (
            (dynamic_j - baseline_dynamic_j) / latency_saved_s
            if latency_saved_s > 0
            else math.inf
        )
        row = {
            "policy": scenario.policy,
            "buffer_slots": scenario.slots,
            "buffer_mib": scenario.slots * load_per_expert_ms / 1000.0 * OFFLOAD_BANDWIDTH_BYTES_S / 2**20,
            "overlap_multiplier": scenario.overlap_multiplier,
            "overlap_budget_ms_per_layer": execution_per_layer_ms * scenario.overlap_multiplier,
            "prediction_recall": scenario.hits / scenario.actual,
            "prediction_precision": scenario.hits / scenario.predicted,
            "exposed_stall_ms": exposed_stall_ms,
            "stall_reduction_pct": 100.0 * useful_ms / total_load_ms,
            "mean_decoder_latency_ms": mean,
            "stdev_decoder_latency_ms": math.sqrt(variance),
            "speedup_vs_no_prefetch": decoder_ms / mean,
            "wasted_transfer_pct_of_demand": 100.0 * wasted_fraction,
            "dynamic_energy_j": dynamic_j,
            "dynamic_energy_change_pct": 100.0 * (dynamic_j / baseline_dynamic_j - 1.0),
            "break_even_stall_power_w": break_even_w,
        }
        rows.append(row)
        for power_w in STALL_POWERS_W:
            baseline_total_j = baseline_dynamic_j + power_w * total_load_ms / 1000.0
            policy_total_j = dynamic_j + power_w * exposed_stall_ms / 1000.0
            energy_rows.append(
                {
                    "policy": scenario.policy,
                    "buffer_slots": scenario.slots,
                    "overlap_multiplier": scenario.overlap_multiplier,
                    "stall_power_w": power_w,
                    "baseline_total_energy_j": baseline_total_j,
                    "policy_total_energy_j": policy_total_j,
                    "energy_saving_pct": 100.0 * (baseline_total_j - policy_total_j) / baseline_total_j,
                }
            )

    metadata = {
        "trace": config["trace"],
        "trace_sha256": trace_prefix_sha256(trace),
        "profile_tokens": profile_tokens,
        "evaluation_tokens": evaluation_tokens,
        "baseline_decoder_ms": decoder_ms,
        "baseline_exposed_stall_ms": total_load_ms,
        "execution_per_layer_ms": execution_per_layer_ms,
        "load_per_expert_per_layer_ms": load_per_expert_ms,
        "expert_size_mib_at_450GBps": load_per_expert_ms / 1000.0 * OFFLOAD_BANDWIDTH_BYTES_S / 2**20,
        "measured_compute_dynamic_energy_j": compute_dynamic_j,
        "measured_offload_dynamic_energy_j": offload_dynamic_j,
        "baseline_dynamic_energy_j": baseline_dynamic_j,
        "energy_scope": (
            "SSD-offloading dynamic energy plus a separately reported stalled-GPU "
            "power sensitivity; no fixed physical idle power is assumed"
        ),
        "residency_policy": (
            "buffer holds only the next layer's candidates and evicts all remaining "
            "entries at that layer boundary"
        ),
    }
    return rows, energy_rows, metadata


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_latency_svg(rows: list[dict], path: Path) -> None:
    selected = [row for row in rows if row["policy"] == "cct"]
    width, height = 1000, 580
    left, top, plot_w, plot_h = 100, 90, 820, 350
    x_values = list(OVERLAP_MULTIPLIERS)
    # Cover the complete 4x-overlap sweep; a fixed 48 ms floor clipped the
    # fastest points and made the apparent trend misleading.
    ymin, ymax = 46.0, 58.0
    colors = {1: "#13a89e", 2: "#2f6fed", 3: "#7b61d1", 4: "#e45756"}
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fff"/>',
        '<style>text{font-family:Segoe UI,Arial,sans-serif;fill:#17233c}.title{font-size:25px;font-weight:700}.sub{font-size:14px;fill:#596579}.tick{font-size:12px;fill:#6b7280}.legend{font-size:14px;font-weight:600}</style>',
        '<text x="32" y="42" class="title">CCT prefetch sensitivity: buffer budget versus overlap</text>',
        '<text x="32" y="68" class="sub">50,000 held-out tokens · measured-calibrated latency · next-layer eviction</text>',
    ]
    for tick in range(46, 59, 2):
        y = top + plot_h * (ymax - tick) / (ymax - ymin)
        svg.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left+plot_w}" y2="{y:.1f}" stroke="#e2e7ef"/>')
        svg.append(f'<text x="{left-12}" y="{y+4:.1f}" text-anchor="end" class="tick">{tick}</text>')
    for index, multiplier in enumerate(x_values):
        x = left + plot_w * index / (len(x_values) - 1)
        svg.append(f'<text x="{x:.1f}" y="{top+plot_h+25}" text-anchor="middle" class="tick">{multiplier:g}×</text>')
    for slots in BUFFER_SLOTS:
        series = sorted(
            (row for row in selected if row["buffer_slots"] == slots),
            key=lambda row: row["overlap_multiplier"],
        )
        points = []
        for index, row in enumerate(series):
            x = left + plot_w * index / (len(series) - 1)
            y = top + plot_h * (ymax - row["mean_decoder_latency_ms"]) / (ymax - ymin)
            points.append(f"{x:.1f},{y:.1f}")
            svg.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{colors[slots]}"/>')
        svg.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{colors[slots]}" stroke-width="4"/>')
        lx = 170 + (slots - 1) * 180
        svg.append(f'<line x1="{lx}" y1="535" x2="{lx+30}" y2="535" stroke="{colors[slots]}" stroke-width="4"/>')
        svg.append(f'<text x="{lx+38}" y="540" class="legend">{slots} slot{"s" if slots != 1 else ""}</text>')
    svg.append(f'<text x="{left+plot_w/2}" y="495" text-anchor="middle" class="sub">Overlap window relative to one layer of expert execution</text>')
    svg.append(f'<text x="28" y="{top+plot_h/2}" text-anchor="middle" class="sub" transform="rotate(-90 28 {top+plot_h/2})">Mean decoder latency (ms)</text>')
    svg.append('</svg>')
    path.write_text("\n".join(svg), encoding="utf-8")


def write_primary_svg(rows: list[dict], metadata: dict, path: Path) -> None:
    cct_one = next(row for row in rows if row["policy"] == "cct" and row["buffer_slots"] == 1 and row["overlap_multiplier"] == 1.0)
    cct_two = next(row for row in rows if row["policy"] == "cct" and row["buffer_slots"] == 2 and row["overlap_multiplier"] == 1.0)
    oracle = next(row for row in rows if row["policy"] == "oracle" and row["buffer_slots"] == 2 and row["overlap_multiplier"] == 1.0)
    items = [
        ("On demand", metadata["baseline_decoder_ms"], "#e45756", "baseline"),
        ("CCT, 1 slot", cct_one["mean_decoder_latency_ms"], "#13a89e", f'{cct_one["speedup_vs_no_prefetch"]:.3f}×'),
        ("CCT, 2 slots", cct_two["mean_decoder_latency_ms"], "#2f6fed", f'{cct_two["speedup_vs_no_prefetch"]:.3f}×'),
        ("Oracle, 2 slots", oracle["mean_decoder_latency_ms"], "#17233c", f'{oracle["speedup_vs_no_prefetch"]:.3f}×'),
    ]
    width, height = 1000, 500
    left, plot_w, scale = 190, 700, 65.0
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fff"/>',
        '<style>text{font-family:Segoe UI,Arial,sans-serif;fill:#17233c}.title{font-size:25px;font-weight:700}.sub{font-size:14px;fill:#596579}.label{font-size:15px;font-weight:600}.value{font-size:14px;font-weight:700}.tick{font-size:12px;fill:#6b7280}</style>',
        '<text x="32" y="42" class="title">Single-GPU expert prefetch reduces exposed stalls</text>',
        '<text x="32" y="68" class="sub">Mixtral · batch 1 · FP16 · 450 GB/s · 50,000 held-out routing tokens</text>',
    ]
    for tick in range(0, 66, 10):
        x = left + plot_w * tick / scale
        svg.append(f'<line x1="{x:.1f}" y1="90" x2="{x:.1f}" y2="350" stroke="#e2e7ef"/>')
        svg.append(f'<text x="{x:.1f}" y="375" text-anchor="middle" class="tick">{tick}</text>')
    for index, (label, value, color, note) in enumerate(items):
        y = 110 + index * 62
        bar_w = plot_w * value / scale
        svg.append(f'<text x="{left-14}" y="{y+25}" text-anchor="end" class="label">{label}</text>')
        svg.append(f'<rect x="{left}" y="{y}" width="{bar_w:.1f}" height="38" rx="5" fill="{color}"/>')
        svg.append(f'<text x="{left+bar_w+9:.1f}" y="{y+25}" class="value">{value:.3f} ms · {note}</text>')
    svg.append(f'<text x="{left+plot_w/2}" y="405" text-anchor="middle" class="sub">Mean modeled decoder latency (ms)</text>')
    svg.append(f'<text x="32" y="452" class="sub">Primary CCT design: 1 × {metadata["expert_size_mib_at_450GBps"]:.0f} MiB next-layer buffer; {cct_one["stall_reduction_pct"]:.2f}% stall reduction; {cct_one["wasted_transfer_pct_of_demand"]:.2f}% extra traffic.</text>')
    svg.append('</svg>')
    path.write_text("\n".join(svg), encoding="utf-8")


def write_energy_svg(energy_rows: list[dict], path: Path) -> None:
    selected = sorted(
        (row for row in energy_rows if row["policy"] == "cct" and row["buffer_slots"] == 1 and row["overlap_multiplier"] == 1.0),
        key=lambda row: float(row["stall_power_w"]),
    )
    width, height = 1000, 500
    left, top, plot_w, plot_h = 100, 95, 820, 300
    ymin, ymax = -7.0, 7.0
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fff"/>',
        '<style>text{font-family:Segoe UI,Arial,sans-serif;fill:#17233c}.title{font-size:25px;font-weight:700}.sub{font-size:14px;fill:#596579}.tick{font-size:12px;fill:#6b7280}.value{font-size:13px;font-weight:600}</style>',
        '<text x="32" y="42" class="title">Energy outcome depends on stalled-GPU power</text>',
        '<text x="32" y="68" class="sub">Measured SSD-offloading dynamic energy plus transparent stall-power sensitivity</text>',
    ]
    for tick in (-6, -3, 0, 3, 6):
        y = top + plot_h * (ymax - tick) / (ymax - ymin)
        svg.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left+plot_w}" y2="{y:.1f}" stroke="{("#697386" if tick == 0 else "#e2e7ef")}"/>')
        svg.append(f'<text x="{left-12}" y="{y+4:.1f}" text-anchor="end" class="tick">{tick}%</text>')
    points = []
    for index, row in enumerate(selected):
        x = left + plot_w * index / (len(selected) - 1)
        saving = float(row["energy_saving_pct"])
        y = top + plot_h * (ymax - saving) / (ymax - ymin)
        points.append(f"{x:.1f},{y:.1f}")
        svg.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="#7b61d1"/>')
        svg.append(f'<text x="{x:.1f}" y="{top+plot_h+25}" text-anchor="middle" class="tick">{row["stall_power_w"]} W</text>')
        svg.append(f'<text x="{x:.1f}" y="{y-10:.1f}" text-anchor="middle" class="value">{saving:.2f}%</text>')
    svg.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="#7b61d1" stroke-width="4"/>')
    svg.append(f'<text x="{left+plot_w/2}" y="455" text-anchor="middle" class="sub">Assumed GPU power while waiting on expert transfers</text>')
    svg.append('</svg>')
    path.write_text("\n".join(svg), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "final_prefetch_single_gpu_batch1.json",
    )
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output = ROOT / config["output_directory"]
    output.mkdir(parents=True, exist_ok=True)
    rows, energy_rows, metadata = run_grid(config)
    write_csv(output / "sensitivity.csv", rows)
    write_csv(output / "energy_sensitivity.csv", energy_rows)
    (output / "sensitivity_summary.json").write_text(
        json.dumps({"metadata": metadata, "scenarios": rows}, indent=2) + "\n",
        encoding="utf-8",
    )
    write_latency_svg(rows, output / "prefetch_sensitivity.svg")
    write_primary_svg(rows, metadata, output / "primary_latency.svg")
    write_energy_svg(energy_rows, output / "energy_sensitivity.svg")

    default = next(
        row for row in rows
        if row["policy"] == "cct"
        and row["buffer_slots"] == 2
        and row["overlap_multiplier"] == 1.0
    )
    best_real = min(
        (row for row in rows if row["policy"] == "cct"),
        key=lambda row: row["mean_decoder_latency_ms"],
    )
    print("default_cct", json.dumps(default, sort_keys=True))
    print("best_cct", json.dumps(best_real, sort_keys=True))
    print("metadata", json.dumps(metadata, sort_keys=True))
    print(f"wrote={output}")


if __name__ == "__main__":
    main()
