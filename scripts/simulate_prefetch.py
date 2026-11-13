#!/usr/bin/env python3
"""Trace-driven Mixtral expert-prefetch timing experiment.

The first half of the submitted AllenAI routing trace profiles per-layer expert
frequencies and adjacent-layer correlation tables.  The held-out half compares
on-demand loading with static, previous-token history (HT), cross-layer
correlation (CCT), hybrid HT+CCT, and oracle policies.

The timing model is calibrated to the submitted LLMSimulator measurements.  A
two-entry next-layer buffer receives a fixed transfer budget while the current
layer's experts execute.  At the next layer, correctly predicted bytes reduce
the exposed on-demand transfer; incorrect bytes are counted as wasted traffic.
The buffer is cleared at each layer boundary, which is a conservative explicit
residency policy and prevents predictions from receiving free long-term reuse.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path


POLICIES = ("no_prefetch", "static", "ht", "cct", "hybrid", "oracle")


def top_indices(values: list[float] | list[int], count: int) -> list[int]:
    return sorted(range(len(values)), key=lambda i: (-values[i], i))[:count]


def iter_sequences(path: Path, token_limit: int):
    """Yield complete trace rows without joining unrelated C4 sequences."""
    used = 0
    with path.open("rb") as handle:
        for raw in handle:
            try:
                row = json.loads(raw)
            except json.JSONDecodeError:
                continue
            routes = row["exp_ids"]
            take = min(len(routes), token_limit - used)
            if take:
                yield used, routes[:take]
                used += take
            if used >= token_limit:
                return
    raise RuntimeError(
        f"Trace contains {used:,} complete tokens; {token_limit:,} are required"
    )


def trace_prefix_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_profile(
    trace: Path,
    profile_tokens: int,
    layers: int,
    experts: int,
):
    # cct[transition][source expert][target expert]
    cct = [
        [[0 for _ in range(experts)] for _ in range(experts)]
        for _ in range(layers - 1)
    ]
    layer_frequency = [[0 for _ in range(experts)] for _ in range(layers)]
    profiled = 0
    for _, routes in iter_sequences(trace, profile_tokens):
        for token in routes:
            for layer in range(layers):
                current = (token[0][layer], token[1][layer])
                for expert in current:
                    layer_frequency[layer][expert] += 1
                if layer + 1 < layers:
                    following = (token[0][layer + 1], token[1][layer + 1])
                    for source in current:
                        for target in following:
                            cct[layer][source][target] += 1
            profiled += 1
    if profiled != profile_tokens:
        raise RuntimeError(f"Profiled {profiled:,}, expected {profile_tokens:,}")
    return cct, layer_frequency


def cct_scores(
    cct: list[list[list[int]]],
    layer: int,
    token,
    layer_frequency: list[list[int]],
    experts: int,
) -> list[float]:
    if layer == 0:
        total = sum(layer_frequency[0])
        return [count / total for count in layer_frequency[0]]
    sources = (token[0][layer - 1], token[1][layer - 1])
    scores = [0.0 for _ in range(experts)]
    for source in sources:
        row = cct[layer - 1][source]
        denominator = sum(row)
        if denominator:
            for target in range(experts):
                scores[target] += row[target] / denominator
    return scores


@dataclass
class Accumulator:
    tokens: int = 0
    layers: int = 0
    actual_experts: int = 0
    predicted_experts: int = 0
    correct_predictions: int = 0
    useful_prefetch_ms: float = 0.0
    wasted_prefetch_ms: float = 0.0
    latency_sum_ms: float = 0.0
    latency_square_sum_ms: float = 0.0

    def add_token(self, latency_ms: float) -> None:
        self.tokens += 1
        self.latency_sum_ms += latency_ms
        self.latency_square_sum_ms += latency_ms * latency_ms


def evaluate(config: dict, root: Path, cct, layer_frequency):
    trace = root / config["trace"]
    profile_tokens = int(config["profile_tokens"])
    evaluation_tokens = int(config["evaluation_tokens"])
    total_tokens = profile_tokens + evaluation_tokens
    layers = int(config["layers"])
    experts = int(config["experts_per_layer"])
    top_k = int(config["top_k"])
    budget = int(config["candidate_budget"])
    hybrid_bonus = float(config["hybrid_history_bonus"])

    total_load_ms = float(config["measured_offchip_expert_load_ms"])
    total_execution_ms = float(config["measured_offchip_expert_execution_ms"])
    decoder_ms = float(config["measured_offchip_decoder_ms"])
    non_expert_ms = decoder_ms - total_load_ms - total_execution_ms
    if non_expert_ms < 0:
        raise ValueError("Measured decoder latency is smaller than its components")

    load_per_expert_ms = total_load_ms / (layers * top_k)
    execution_per_layer_ms = total_execution_ms / layers
    overlap_budget_ms = (
        execution_per_layer_ms * float(config["overlap_multiplier"])
    )

    accumulators = {policy: Accumulator() for policy in POLICIES}
    evaluated = 0

    for offset, routes in iter_sequences(trace, total_tokens):
        start = max(0, profile_tokens - offset)
        if start >= len(routes):
            continue
        for position in range(start, len(routes)):
            token = routes[position]
            previous = routes[position - 1] if position > 0 else None
            token_exposed = {policy: 0.0 for policy in POLICIES}

            for layer in range(layers):
                actual = {token[0][layer], token[1][layer]}
                if len(actual) != top_k:
                    raise RuntimeError("Trace row does not contain distinct top-k experts")

                static = set(top_indices(layer_frequency[layer], budget))
                history = (
                    {previous[0][layer], previous[1][layer]}
                    if previous is not None
                    else static
                )
                scores = cct_scores(cct, layer, token, layer_frequency, experts)
                cct_prediction = set(top_indices(scores, budget))
                hybrid_scores = [
                    scores[expert]
                    + (hybrid_bonus if expert in history else 0.0)
                    for expert in range(experts)
                ]
                predictions = {
                    "no_prefetch": set(),
                    "static": static,
                    "ht": history,
                    "cct": cct_prediction,
                    "hybrid": set(top_indices(hybrid_scores, budget)),
                    "oracle": set(actual),
                }

                full_layer_load_ms = top_k * load_per_expert_ms
                for policy, predicted in predictions.items():
                    stats = accumulators[policy]
                    stats.layers += 1
                    stats.actual_experts += len(actual)
                    stats.predicted_experts += len(predicted)
                    hits = len(actual & predicted)
                    misses = len(predicted - actual)
                    stats.correct_predictions += hits

                    progress_per_candidate_ms = (
                        min(load_per_expert_ms, overlap_budget_ms / len(predicted))
                        if predicted
                        else 0.0
                    )
                    useful_ms = hits * progress_per_candidate_ms
                    wasted_ms = misses * progress_per_candidate_ms
                    stats.useful_prefetch_ms += useful_ms
                    stats.wasted_prefetch_ms += wasted_ms
                    token_exposed[policy] += full_layer_load_ms - useful_ms

            for policy in POLICIES:
                token_latency_ms = (
                    non_expert_ms + total_execution_ms + token_exposed[policy]
                )
                accumulators[policy].add_token(token_latency_ms)
            evaluated += 1

    if evaluated != evaluation_tokens:
        raise RuntimeError(f"Evaluated {evaluated:,}, expected {evaluation_tokens:,}")

    baseline_mean = accumulators["no_prefetch"].latency_sum_ms / evaluated
    if not math.isclose(baseline_mean, decoder_ms, rel_tol=0, abs_tol=1e-9):
        raise RuntimeError(
            f"Calibration failed: baseline {baseline_mean:.9f} != {decoder_ms:.9f}"
        )

    rows = []
    necessary_transfer_ms = evaluation_tokens * total_load_ms
    for policy in POLICIES:
        stats = accumulators[policy]
        mean = stats.latency_sum_ms / stats.tokens
        variance = max(
            0.0, stats.latency_square_sum_ms / stats.tokens - mean * mean
        )
        recall = (
            stats.correct_predictions / stats.actual_experts
            if stats.predicted_experts
            else 0.0
        )
        precision = (
            stats.correct_predictions / stats.predicted_experts
            if stats.predicted_experts
            else 0.0
        )
        rows.append(
            {
                "policy": policy,
                "tokens": stats.tokens,
                "candidate_budget": 0 if policy == "no_prefetch" else budget,
                "prediction_recall": recall,
                "prediction_precision": precision,
                "mean_decoder_latency_ms": mean,
                "stdev_decoder_latency_ms": math.sqrt(variance),
                "speedup_vs_no_prefetch": baseline_mean / mean,
                "latency_reduction_pct": 100.0 * (baseline_mean - mean) / baseline_mean,
                "useful_prefetch_ms_per_token": stats.useful_prefetch_ms / stats.tokens,
                "wasted_prefetch_ms_per_token": stats.wasted_prefetch_ms / stats.tokens,
                "wasted_transfer_pct_of_demand": (
                    100.0 * stats.wasted_prefetch_ms / necessary_transfer_ms
                ),
            }
        )

    metadata = {
        "trace": str(trace.relative_to(root)),
        "trace_sha256": trace_prefix_sha256(trace),
        "profile_tokens": profile_tokens,
        "evaluation_tokens": evaluation_tokens,
        "layers": layers,
        "experts_per_layer": experts,
        "top_k": top_k,
        "candidate_budget": budget,
        "load_per_expert_per_layer_ms": load_per_expert_ms,
        "execution_per_layer_ms": execution_per_layer_ms,
        "prefetch_overlap_budget_per_layer_ms": overlap_budget_ms,
        "non_expert_decoder_ms": non_expert_ms,
        "buffer_policy": (
            "two next-layer candidates; partial bytes retained until that layer "
            "and discarded immediately afterward"
        ),
        "calibration": {
            "offchip_expert_load_ms": total_load_ms,
            "offchip_expert_execution_ms": total_execution_ms,
            "offchip_decoder_no_prefetch_ms": decoder_ms,
        },
    }
    return rows, metadata


def write_svg(rows: list[dict], destination: Path) -> None:
    width, height = 1000, 500
    left, top, plot_width, plot_height = 190, 90, 720, 300
    values = [row["mean_decoder_latency_ms"] for row in rows]
    maximum = max(values) * 1.08
    bar_height = 36
    gap = 12
    colors = {
        "no_prefetch": "#e45756",
        "static": "#9aa4b2",
        "ht": "#13a89e",
        "cct": "#2f6fed",
        "hybrid": "#7b61d1",
        "oracle": "#17233c",
    }
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Segoe UI,Arial,sans-serif;fill:#17233c}.title{font-size:25px;font-weight:700}.sub{font-size:14px;fill:#596579}.label{font-size:15px;font-weight:600}.value{font-size:14px;font-weight:700}.tick{font-size:12px;fill:#6b7280}</style>',
        '<text x="32" y="40" class="title">Measured-calibrated expert prefetch latency</text>',
        '<text x="32" y="66" class="sub">Held-out 50,000-token Mixtral trace · fixed two-expert budget · identical overlap window</text>',
    ]
    for tick in range(0, 61, 10):
        x = left + plot_width * tick / maximum
        svg.append(
            f'<line x1="{x:.1f}" y1="{top-10}" x2="{x:.1f}" y2="{top+plot_height}" stroke="#e2e7ef"/>'
        )
        svg.append(
            f'<text x="{x:.1f}" y="{top+plot_height+24}" text-anchor="middle" class="tick">{tick}</text>'
        )
    for index, row in enumerate(rows):
        y = top + index * (bar_height + gap)
        value = row["mean_decoder_latency_ms"]
        bar_width = plot_width * value / maximum
        policy = row["policy"]
        svg.extend(
            [
                f'<text x="{left-14}" y="{y+24}" text-anchor="end" class="label">{policy.replace("_", " ")}</text>',
                f'<rect x="{left}" y="{y}" width="{bar_width:.1f}" height="{bar_height}" rx="5" fill="{colors[policy]}"/>',
                f'<text x="{left+bar_width+9:.1f}" y="{y+24}" class="value">{value:.3f} ms · {row["speedup_vs_no_prefetch"]:.3f}×</text>',
            ]
        )
    svg.append(
        f'<text x="{left+plot_width/2}" y="{height-45}" text-anchor="middle" class="sub">Mean modeled decoder latency (ms)</text>'
    )
    svg.append('</svg>')
    destination.write_text("\n".join(svg), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/final_prefetch.json"),
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    config_path = args.config if args.config.is_absolute() else root / args.config
    config = json.loads(config_path.read_text(encoding="utf-8"))
    trace = root / config["trace"]
    output = root / config["output_directory"]
    output.mkdir(parents=True, exist_ok=True)

    cct, layer_frequency = build_profile(
        trace,
        int(config["profile_tokens"]),
        int(config["layers"]),
        int(config["experts_per_layer"]),
    )
    rows, metadata = evaluate(config, root, cct, layer_frequency)

    csv_path = output / "policy_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    json_path = output / "prefetch_results.json"
    json_path.write_text(
        json.dumps({"metadata": metadata, "policies": rows}, indent=2) + "\n",
        encoding="utf-8",
    )
    svg_path = output / "prefetch_latency.svg"
    write_svg(rows, svg_path)

    for row in rows:
        print(
            f'{row["policy"]:12s} '
            f'latency={row["mean_decoder_latency_ms"]:.6f} ms '
            f'speedup={row["speedup_vs_no_prefetch"]:.6f}x '
            f'recall={100*row["prediction_recall"]:.3f}% '
            f'waste={row["wasted_transfer_pct_of_demand"]:.3f}%'
        )
    print(f"wrote={csv_path}")
    print(f"wrote={json_path}")
    print(f"wrote={svg_path}")


if __name__ == "__main__":
    main()
