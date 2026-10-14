#!/usr/bin/env python3
"""Analyze spatial and temporal expert correlation in AllenAI's Mixtral trace.

The input is a byte-range sample of c4_results.jsonl from
https://huggingface.co/datasets/allenai/analysis_mixtral.  Each complete row
contains 2,048 causal-token routing decisions with shape [token, top-2, 32].
Only the first 100,000 complete token records are used.  The first half builds
cross-layer correlation tables; all predictor results use the held-out half.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRACE = ROOT / "data" / "day3" / "c4_results.first24MiB.jsonl.part"
OUT = ROOT / "results" / "day3"
TOTAL_TOKENS = 100_000
PROFILE_TOKENS = 50_000
N_LAYERS = 32
N_EXPERTS = 8
TOP_K = 2
RANDOM_OVERLAP = TOP_K * TOP_K / N_EXPERTS
RANDOM_RECALL = TOP_K / N_EXPERTS


def iter_sequences(path: Path, token_limit: int):
    """Yield complete or final truncated sequences, ignoring a partial last line."""
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
    raise RuntimeError(f"Trace has only {used:,} complete tokens; need {token_limit:,}")


def top_indices(values, count=TOP_K):
    return sorted(range(len(values)), key=lambda i: (-values[i], i))[:count]


def build_profile():
    # transition -> source expert -> target expert
    cct = [[[0 for _ in range(N_EXPERTS)] for _ in range(N_EXPERTS)]
           for _ in range(N_LAYERS - 1)]
    target_freq = [[0 for _ in range(N_EXPERTS)] for _ in range(N_LAYERS - 1)]
    profiled = 0
    for offset, routes in iter_sequences(TRACE, PROFILE_TOKENS):
        del offset
        for token in routes:
            for layer in range(N_LAYERS - 1):
                sources = (token[0][layer], token[1][layer])
                targets = (token[0][layer + 1], token[1][layer + 1])
                for dst in targets:
                    target_freq[layer][dst] += 1
                for src in sources:
                    for dst in targets:
                        cct[layer][src][dst] += 1
            profiled += 1
    assert profiled == PROFILE_TOKENS
    return cct, target_freq


def association_v(matrix):
    total = sum(map(sum, matrix))
    rows = [sum(row) for row in matrix]
    cols = [sum(matrix[r][c] for r in range(N_EXPERTS)) for c in range(N_EXPERTS)]
    chi2 = 0.0
    for r in range(N_EXPERTS):
        for c in range(N_EXPERTS):
            expected = rows[r] * cols[c] / total
            if expected:
                chi2 += (matrix[r][c] - expected) ** 2 / expected
    return math.sqrt(chi2 / (total * (N_EXPERTS - 1)))


def evaluate(cct, target_freq):
    cct_entries = [[top_indices(cct[layer][src]) for src in range(N_EXPERTS)]
                   for layer in range(N_LAYERS - 1)]
    static_entries = [top_indices(target_freq[layer]) for layer in range(N_LAYERS - 1)]

    spatial_hits = [0] * (N_LAYERS - 1)
    spatial_union_hits = [0] * (N_LAYERS - 1)
    spatial_union_size = [0] * (N_LAYERS - 1)
    static_hits = [0] * (N_LAYERS - 1)
    spatial_trials = [0] * (N_LAYERS - 1)
    temporal_overlap = [0] * N_LAYERS
    temporal_pairs = [0] * N_LAYERS
    evaluated = 0

    for offset, routes in iter_sequences(TRACE, TOTAL_TOKENS):
        start = max(0, PROFILE_TOKENS - offset)
        if start >= len(routes):
            continue
        for pos in range(start, len(routes)):
            token = routes[pos]
            for layer in range(N_LAYERS - 1):
                sources = (token[0][layer], token[1][layer])
                actual = {token[0][layer + 1], token[1][layer + 1]}

                # Fixed two-expert budget: sum P(target expert | each current expert).
                scores = []
                for dst in range(N_EXPERTS):
                    score = 0.0
                    for src in sources:
                        denom = sum(cct[layer][src])
                        score += cct[layer][src][dst] / denom if denom else 0.0
                    scores.append(score)
                predicted = set(top_indices(scores))
                spatial_hits[layer] += len(actual & predicted)

                # Literal CCT lookup: union of two Top-2 entries (2--4 candidates).
                union = set(cct_entries[layer][sources[0]]) | set(cct_entries[layer][sources[1]])
                spatial_union_hits[layer] += len(actual & union)
                spatial_union_size[layer] += len(union)

                static_hits[layer] += len(actual & set(static_entries[layer]))
                spatial_trials[layer] += len(actual)

            # Do not create a temporal pair across independent 2,048-token rows.
            if pos > 0 and offset + pos > PROFILE_TOKENS:
                previous = routes[pos - 1]
                for layer in range(N_LAYERS):
                    current_set = {token[0][layer], token[1][layer]}
                    previous_set = {previous[0][layer], previous[1][layer]}
                    temporal_overlap[layer] += len(current_set & previous_set)
                    temporal_pairs[layer] += 1
            evaluated += 1

    rows = []
    for layer in range(N_LAYERS):
        row = {
            "layer": layer,
            "temporal_mean_overlap": temporal_overlap[layer] / temporal_pairs[layer],
            "temporal_recall": temporal_overlap[layer] / (temporal_pairs[layer] * TOP_K),
            "temporal_random_overlap": RANDOM_OVERLAP,
        }
        if layer < N_LAYERS - 1:
            row.update({
                "spatial_transition": f"{layer}->{layer + 1}",
                "spatial_top2_recall": spatial_hits[layer] / spatial_trials[layer],
                "spatial_static_top2_recall": static_hits[layer] / spatial_trials[layer],
                "spatial_random_top2_recall": RANDOM_RECALL,
                "cct_union_recall": spatial_union_hits[layer] / spatial_trials[layer],
                "cct_union_mean_candidates": spatial_union_size[layer] / evaluated,
                "cramers_v_profile": association_v(cct[layer]),
            })
        rows.append(row)
    return rows, evaluated


def weighted_mean(rows, key, layers):
    return sum(rows[i][key] for i in layers) / len(layers)


def write_svg(rows, summary):
    width, height = 1280, 720
    bg, navy, blue, orange, teal, gray, grid = (
        "#f7f8fb", "#17233c", "#2f6fed", "#ef8354", "#13a89e", "#697386", "#d9deea"
    )

    def poly(values, x0, y0, w, h, ymin, ymax):
        pts = []
        for i, value in enumerate(values):
            x = x0 + w * i / (len(values) - 1)
            y = y0 + h * (ymax - value) / (ymax - ymin)
            pts.append(f"{x:.1f},{y:.1f}")
        return " ".join(pts)

    def panel(x, y, w, h, title, ylabel, ymin, ymax, series, baseline=None):
        out = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="14" fill="white"/>',
               f'<text x="{x+28}" y="{y+40}" class="panel">{title}</text>']
        px, py, pw, ph = x + 76, y + 68, w - 108, h - 120
        for j in range(5):
            val = ymin + (ymax - ymin) * j / 4
            yy = py + ph * (ymax - val) / (ymax - ymin)
            out.append(f'<line x1="{px}" y1="{yy:.1f}" x2="{px+pw}" y2="{yy:.1f}" stroke="{grid}"/>')
            out.append(f'<text x="{px-12}" y="{yy+5:.1f}" text-anchor="end" class="tick">{val:.0f}</text>')
        if baseline:
            label, val, color = baseline
            yy = py + ph * (ymax - val) / (ymax - ymin)
            out.append(f'<line x1="{px}" y1="{yy:.1f}" x2="{px+pw}" y2="{yy:.1f}" stroke="{color}" stroke-width="2" stroke-dasharray="7 6"/>')
            out.append(f'<text x="{px+pw-4}" y="{yy-8:.1f}" text-anchor="end" fill="{color}" class="small">{label}</text>')
        for label, values, color in series:
            out.append(f'<polyline points="{poly(values, px, py, pw, ph, ymin, ymax)}" fill="none" stroke="{color}" stroke-width="4" stroke-linejoin="round"/>')
        for i in (0, 7, 15, 23, len(series[0][1]) - 1):
            xx = px + pw * i / (len(series[0][1]) - 1)
            out.append(f'<text x="{xx:.1f}" y="{py+ph+25}" text-anchor="middle" class="tick">{i}</text>')
        out.append(f'<text x="{px+pw/2}" y="{y+h-18}" text-anchor="middle" class="axis">Layer / transition start</text>')
        out.append(f'<text x="{x+18}" y="{py+ph/2}" text-anchor="middle" class="axis" transform="rotate(-90 {x+18} {py+ph/2})">{ylabel}</text>')
        lx = px + 8
        for label, _, color in series:
            out.append(f'<line x1="{lx}" y1="{y+h-46}" x2="{lx+25}" y2="{y+h-46}" stroke="{color}" stroke-width="4"/>')
            out.append(f'<text x="{lx+32}" y="{y+h-41}" class="small">{label}</text>')
            lx += 170
        return "".join(out)

    temporal = [100 * r["temporal_recall"] for r in rows]
    spatial = [100 * rows[i]["spatial_top2_recall"] for i in range(N_LAYERS - 1)]
    static = [100 * rows[i]["spatial_static_top2_recall"] for i in range(N_LAYERS - 1)]
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
           '<style>text{font-family:Segoe UI,Arial,sans-serif}.title{font-size:28px;font-weight:700;fill:#17233c}.sub{font-size:15px;fill:#697386}.panel{font-size:18px;font-weight:700;fill:#17233c}.tick{font-size:12px;fill:#697386}.axis{font-size:13px;font-weight:600;fill:#697386}.small{font-size:12px;font-weight:600}</style>',
           f'<rect width="100%" height="100%" fill="{bg}"/>',
           '<text x="48" y="48" class="title">Real Mixtral routes show exploitable expert correlation</text>',
           '<text x="48" y="76" class="sub">100,000 C4 token routes · Mixtral-8×7B · top-2 of 8 experts · predictor evaluated on held-out 50,000 tokens</text>',
           panel(38, 104, 592, 430, "A. Consecutive-token reuse (HT)", "Previous-token recall (%)", 0, 50,
                 [("Observed", temporal, teal)], ("Random expectation: 25%", 25, gray)),
           panel(650, 104, 592, 430, "B. Adjacent-layer prediction (CCT)", "Held-out top-2 recall (%)", 0, 80,
                 [("CCT", spatial, blue), ("Static top-2", static, orange)], ("Random expectation: 25%", 25, gray)),
           f'<rect x="38" y="558" width="1204" height="122" rx="14" fill="{navy}"/>',
           '<text x="70" y="592" font-family="Segoe UI,Arial" font-size="14" fill="#bdc8df">MEAN RESULTS</text>',
           f'<text x="70" y="636" font-family="Segoe UI,Arial" font-size="29" font-weight="700" fill="white">{summary["temporal_recall"]*100:.1f}%</text>',
           '<text x="70" y="660" font-family="Segoe UI,Arial" font-size="13" fill="#bdc8df">HT recall at 2-expert budget</text>',
           f'<text x="350" y="636" font-family="Segoe UI,Arial" font-size="29" font-weight="700" fill="white">{summary["temporal_overlap"]:.3f}</text>',
           '<text x="350" y="660" font-family="Segoe UI,Arial" font-size="13" fill="#bdc8df">shared experts vs 0.500 random</text>',
           f'<text x="680" y="636" font-family="Segoe UI,Arial" font-size="29" font-weight="700" fill="white">{summary["spatial_recall"]*100:.1f}%</text>',
           '<text x="680" y="660" font-family="Segoe UI,Arial" font-size="13" fill="#bdc8df">held-out CCT recall, top-2 budget</text>',
           f'<text x="1010" y="636" font-family="Segoe UI,Arial" font-size="29" font-weight="700" fill="white">{summary["cct_union_recall"]*100:.1f}%</text>',
           f'<text x="1010" y="660" font-family="Segoe UI,Arial" font-size="13" fill="#bdc8df">CCT union recall ({summary["cct_union_candidates"]:.2f} candidates)</text>',
           '</svg>']
    (OUT / "day3_correlation.svg").write_text("".join(svg), encoding="utf-8")


def main():
    if not TRACE.exists():
        raise SystemExit(f"Missing trace: {TRACE}")
    OUT.mkdir(parents=True, exist_ok=True)
    cct, target_freq = build_profile()
    rows, evaluated = evaluate(cct, target_freq)

    with (OUT / "layer_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = list(rows[0].keys())
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "tokens_total": TOTAL_TOKENS,
        "tokens_profile": PROFILE_TOKENS,
        "tokens_evaluation": evaluated,
        "temporal_overlap": weighted_mean(rows, "temporal_mean_overlap", range(N_LAYERS)),
        "temporal_recall": weighted_mean(rows, "temporal_recall", range(N_LAYERS)),
        "random_overlap": RANDOM_OVERLAP,
        "spatial_recall": weighted_mean(rows, "spatial_top2_recall", range(N_LAYERS - 1)),
        "static_top2_recall": weighted_mean(rows, "spatial_static_top2_recall", range(N_LAYERS - 1)),
        "random_top2_recall": RANDOM_RECALL,
        "cct_union_recall": weighted_mean(rows, "cct_union_recall", range(N_LAYERS - 1)),
        "cct_union_candidates": weighted_mean(rows, "cct_union_mean_candidates", range(N_LAYERS - 1)),
        "mean_cramers_v": weighted_mean(rows, "cramers_v_profile", range(N_LAYERS - 1)),
        "trace_partial_sha256": hashlib.sha256(TRACE.read_bytes()).hexdigest(),
        "source_lfs_sha256": "55c76a34adf54510ad0f64a161236da93fdc742ca8eecbbe776f21f8d7e0d1cd",
        "source_byte_range": "0-25165823",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_svg(rows, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
