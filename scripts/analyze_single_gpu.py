#!/usr/bin/env python3
"""Summarize the final single-GPU, batch-1 LLMSimulator calibration runs."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results" / "final_day2" / "single_gpu_batch1" / "raw"
OUT = ROOT / "results" / "final_day2" / "single_gpu_batch1"


def rows(pattern: str):
    matches = list(RAW.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {pattern} file, found {len(matches)}")
    with matches[0].open(newline="", encoding="utf-8") as handle:
        return matches[0], list(csv.DictReader(handle))


def mean_ms(data, field):
    return statistics.fmean(float(row[field]) for row in data) / 1e6


def stdev_ms(data, field):
    return statistics.stdev(float(row[field]) for row in data) / 1e6


def main() -> None:
    baseline_path, baseline = rows("*baseline*GPU_compute.csv")
    compute_path, compute = rows("*CPU_memory*GPU_compute.csv")
    load_path, load = rows("*offloaded_weight_copy.csv")
    if not (len(baseline) == len(compute) == len(load) == 7):
        raise RuntimeError("Expected seven matched samples in every run")

    compute_energy_columns = (
        "fc_dram", "fc_comp", "attn_dram", "attn_comp", "ffn_dram",
        "ffn_comp", "shared_moe_hbm", "shared_moe_comp", "moe_hbm", "moe_comp",
    )
    compute_dynamic_j = statistics.fmean(
        sum(float(row[column]) for column in compute_energy_columns)
        for row in compute
    ) / 1e9
    offload_dynamic_j = statistics.fmean(float(row["moe_offload"]) for row in load) / 1e9

    metrics = [
        ("resident_expert_execution", mean_ms(baseline, "expert_ffn"), stdev_ms(baseline, "expert_ffn")),
        ("offchip_expert_execution", mean_ms(compute, "expert_ffn"), stdev_ms(compute, "expert_ffn")),
        ("offchip_expert_loading", mean_ms(load, "expert_ffn"), stdev_ms(load, "expert_ffn")),
        ("offchip_expert_no_prefetch", mean_ms(compute, "expert_ffn") + mean_ms(load, "expert_ffn"), math_stdev_sum(compute, load, "expert_ffn")),
        ("resident_decoder", mean_ms(baseline, "MoE_decoders"), stdev_ms(baseline, "MoE_decoders")),
        ("offchip_decoder_no_prefetch", mean_ms(compute, "MoE_decoders") + mean_ms(load, "expert_ffn"), math_stdev_decoder(compute, load)),
    ]

    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "single_gpu_calibration.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("metric", "samples", "mean_ms", "stdev_ms"))
        for metric, mean, stdev in metrics:
            writer.writerow((metric, 7, mean, stdev))

    summary = {
        "configuration": {
            "nodes": 1,
            "devices": 1,
            "batch_size": 1,
            "expert_tensor_degree": 1,
            "non_expert_tensor_degree": 1,
            "precision": "FP16",
            "offchip_link_GBps": 450,
            "samples": 7,
        },
        "files": {
            "baseline": baseline_path.name,
            "offchip_compute": compute_path.name,
            "offchip_load": load_path.name,
        },
        "metrics": {metric: {"mean_ms": mean, "stdev_ms": stdev} for metric, mean, stdev in metrics},
        "energy": {
            "compute_dynamic_j": compute_dynamic_j,
            "offload_dynamic_j": offload_dynamic_j,
            "baseline_total_dynamic_j": compute_dynamic_j + offload_dynamic_j,
            "source_unit": "CSV energy fields are nJ",
        },
    }
    (OUT / "single_gpu_calibration.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


def math_stdev_sum(compute, load, field):
    values = [
        (float(compute[i][field]) + float(load[i][field])) / 1e6
        for i in range(len(compute))
    ]
    return statistics.stdev(values)


def math_stdev_decoder(compute, load):
    values = [
        (float(compute[i]["MoE_decoders"]) + float(load[i]["expert_ffn"])) / 1e6
        for i in range(len(compute))
    ]
    return statistics.stdev(values)


if __name__ == "__main__":
    main()
