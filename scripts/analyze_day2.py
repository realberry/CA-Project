#!/usr/bin/env python3
import csv
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results" / "day2" / "raw"
OUT = ROOT / "results" / "day2" / "summary.csv"
SVG_OUT = ROOT / "results" / "day2" / "day2_breakdown.svg"


def find_csv(fragment: str) -> Path:
    matches = sorted(path for path in RAW.glob("*.csv") if fragment in path.name)
    if len(matches) != 1:
        raise RuntimeError(f"expected one CSV containing {fragment!r}, found {matches}")
    return matches[0]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def values(rows: list[dict[str, str]], column: str) -> list[float]:
    return [float(row[column]) for row in rows]


def stats_ns(series: list[float]) -> tuple[float, float, float, float]:
    mean = statistics.fmean(series)
    stdev = statistics.stdev(series) if len(series) > 1 else 0.0
    return mean, stdev, min(series), max(series)


baseline = read_rows(find_csv("baseline_NVLink4_GPU_compute"))
offchip_compute = read_rows(find_csv("CPU_memory_NVLink4_GPU_compute"))
offchip_load = read_rows(find_csv("CPU_memory_NVLink4_offloaded_weight_copy"))

row_counts = {len(baseline), len(offchip_compute), len(offchip_load)}
if len(row_counts) != 1:
    raise RuntimeError(f"row-count mismatch: {row_counts}")

metrics = {
    "baseline_expert_execution": values(baseline, "expert_ffn"),
    "offchip_expert_execution": values(offchip_compute, "expert_ffn"),
    "offchip_expert_loading": values(offchip_load, "expert_ffn"),
    "baseline_moe_decoder": values(baseline, "MoE_decoders"),
    "offchip_moe_decoder_execution": values(offchip_compute, "MoE_decoders"),
}

metrics["offchip_expert_no_prefetch"] = [
    load + execute
    for load, execute in zip(
        metrics["offchip_expert_loading"], metrics["offchip_expert_execution"]
    )
]
metrics["offchip_moe_decoder_no_prefetch"] = [
    load + execute
    for load, execute in zip(
        metrics["offchip_expert_loading"], metrics["offchip_moe_decoder_execution"]
    )
]

with OUT.open("w", newline="", encoding="utf-8") as stream:
    writer = csv.writer(stream)
    writer.writerow(["metric", "samples", "mean_ms", "stdev_ms", "min_ms", "max_ms"])
    for name, series in metrics.items():
        mean, stdev, minimum, maximum = stats_ns(series)
        writer.writerow(
            [name, len(series), mean / 1e6, stdev / 1e6, minimum / 1e6, maximum / 1e6]
        )

baseline_mean = statistics.fmean(metrics["baseline_expert_execution"])
execute_mean = statistics.fmean(metrics["offchip_expert_execution"])
load_mean = statistics.fmean(metrics["offchip_expert_loading"])
sequential_mean = statistics.fmean(metrics["offchip_expert_no_prefetch"])

print(f"samples={len(baseline)}")
print(f"baseline_expert_execution_ms={baseline_mean / 1e6:.6f}")
print(f"offchip_expert_execution_ms={execute_mean / 1e6:.6f}")
print(f"offchip_expert_loading_ms={load_mean / 1e6:.6f}")
print(f"offchip_expert_no_prefetch_ms={sequential_mean / 1e6:.6f}")
print(f"loading_to_execution_ratio={load_mean / execute_mean:.6f}")
print(f"loading_share_no_prefetch_pct={100.0 * load_mean / sequential_mean:.3f}")
print(f"offchip_execution_overhead_vs_baseline_pct={100.0 * (execute_mean / baseline_mean - 1.0):.3f}")
print(f"wrote={OUT}")

width, height = 1000, 470
plot_x, plot_width = 190, 650
scale_max_ms = 60.0
baseline_ms = baseline_mean / 1e6
execute_ms = execute_mean / 1e6
load_ms = load_mean / 1e6
sequential_ms = sequential_mean / 1e6


def x_for(value_ms: float) -> float:
    return plot_x + plot_width * value_ms / scale_max_ms


svg: list[str] = [
    f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
    '<rect width="100%" height="100%" fill="#ffffff"/>',
    '<style>text{font-family:Segoe UI,Arial,sans-serif;fill:#172033}.title{font-size:25px;font-weight:700}.sub{font-size:14px;fill:#526071}.label{font-size:16px;font-weight:600}.tick{font-size:12px;fill:#657184}.value{font-size:15px;font-weight:700}.legend{font-size:14px}</style>',
    '<text x="32" y="42" class="title">Expert latency breakdown: off-chip DRAM creates the memory wall</text>',
    '<text x="32" y="70" class="sub">Mixtral, batch 4, four H100 devices, expert tensor degree 1, FP16, 450 GB/s off-chip link, n=7 decode samples</text>',
]

for tick in range(0, 61, 10):
    x = x_for(float(tick))
    svg.append(f'<line x1="{x:.1f}" y1="105" x2="{x:.1f}" y2="330" stroke="#dce2ea" stroke-width="1"/>')
    svg.append(f'<text x="{x:.1f}" y="350" text-anchor="middle" class="tick">{tick}</text>')
svg.append('<text x="515" y="378" text-anchor="middle" class="sub">Latency per decode step (ms)</text>')

baseline_y = 135
offchip_y = 235
bar_h = 52
baseline_w = x_for(baseline_ms) - plot_x
load_w = x_for(load_ms) - plot_x
execute_w = x_for(execute_ms) - plot_x
svg.extend(
    [
        f'<text x="170" y="{baseline_y + 32}" text-anchor="end" class="label">GPU-resident</text>',
        f'<rect x="{plot_x}" y="{baseline_y}" width="{baseline_w:.1f}" height="{bar_h}" rx="5" fill="#2f6fed"/>',
        f'<text x="{x_for(baseline_ms) + 10:.1f}" y="{baseline_y + 33}" class="value">{baseline_ms:.2f} ms</text>',
        f'<text x="170" y="{offchip_y + 32}" text-anchor="end" class="label">Off-chip, no prefetch</text>',
        f'<rect x="{plot_x}" y="{offchip_y}" width="{load_w:.1f}" height="{bar_h}" rx="5" fill="#e45756"/>',
        f'<rect x="{x_for(load_ms):.1f}" y="{offchip_y}" width="{execute_w:.1f}" height="{bar_h}" rx="5" fill="#2f6fed"/>',
        f'<text x="{plot_x + load_w / 2:.1f}" y="{offchip_y + 33}" text-anchor="middle" fill="#ffffff" class="value">load {load_ms:.2f} ms</text>',
        f'<text x="{x_for(sequential_ms) + 10:.1f}" y="{offchip_y + 33}" class="value">{sequential_ms:.2f} ms total</text>',
        '<rect x="250" y="408" width="18" height="18" rx="3" fill="#e45756"/>',
        '<text x="278" y="422" class="legend">Expert-weight loading</text>',
        '<rect x="480" y="408" width="18" height="18" rx="3" fill="#2f6fed"/>',
        '<text x="508" y="422" class="legend">Expert execution</text>',
        f'<text x="32" y="454" class="sub">Loading is {load_mean / execute_mean:.2f}× execution and {100.0 * load_mean / sequential_mean:.1f}% of the no-prefetch expert path.</text>',
        '</svg>',
    ]
)
SVG_OUT.write_text("\n".join(svg), encoding="utf-8")
print(f"wrote={SVG_OUT}")
