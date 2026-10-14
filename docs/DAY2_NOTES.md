# Day 2: Off-chip expert-weight path

## Outcome

The patched SSD-offloading branch of LLMSimulator builds and runs. The experiment removes Logic-PIM execution, disables expert tensor sharding, and models every selected gated expert as being loaded from off-chip CPU DRAM before GPU execution.

The primary result is a clear memory wall: mean expert-weight loading latency is **48.774 ms**, compared with **7.570 ms** for expert execution. With no prefetch overlap, the expert path is **56.344 ms** per emitted decode step. Loading is **6.44x** execution and accounts for **86.6%** of that path.

## Revisions and build

- SSD-offloading wrapper: `99f3903d67cda89471881a531dd57420abc6e8be`
- LLMSimulator: `3a0ea80c343ac52b67e0945fca7bbdb8a7a4adef`
- Ramulator2: `b5262869b24790e5ec62d9d9974fca9bbb4f54e0`
- Compiler: GCC/G++ 13.3.0
- Required compatibility fix: restore `<cstdint>` in Ramulator2 `src/base/utils.h`

## Matched configuration

- Model: Mixtral
- Mode: decode
- Input length: 1024 tokens
- Requested output length / iterations: 8 / 8
- Emitted measurements: 7 decode steps
- Batch size: 4
- Precision: FP16 (`precision_byte: 2`)
- System: one node, four H100 devices
- Processor: `GPU` only; no Logic or PIM path is present
- Expert tensor degree: 1, so an expert matrix is not split across GPUs
- Non-expert tensor degree: 4
- Off-chip store: CPU memory (DDR5 energy model)
- Effective CPU-memory-to-GPU link: NVLink 4, 450 GB/s
- Workload trace: synthetic

The output filename contains `TP4` because the patched evaluator names files using the **non-expert** tensor degree. The expert tensor degree remains 1 in all saved configurations.

## Runs

1. `baseline`: GPU-resident weights, GPU compute collection.
2. `offchip_compute`: CPU-memory expert offload enabled, GPU compute collection. The simulator reduces available HBM bandwidth by the concurrent 450 GB/s offload bandwidth.
3. `offchip_load`: CPU-memory expert offload enabled, weight-copy collection. The `expert_ffn` field gives selected expert-weight transfer latency.

All three runs completed successfully. Their exact configurations, logs, status files, and raw CSVs are under `results/day2/`.

## Results

| Measurement | Mean (ms) | Std. dev. (ms) |
|---|---:|---:|
| Baseline expert execution | 6.554 | 0.117 |
| Off-chip-case expert execution | 7.570 | 0.135 |
| Off-chip expert loading | 48.774 | 0.871 |
| Off-chip expert path, no prefetch | 56.344 | 1.006 |
| Baseline MoE decoder | 7.141 | 0.117 |
| Off-chip MoE decoder execution | 8.196 | 0.135 |
| Off-chip MoE decoder, no prefetch | 56.970 | 1.006 |

The off-chip compute run is 15.5% slower than baseline expert execution because the model reserves 450 GB/s of the HBM bandwidth for incoming weights. The transfer itself remains dominant.

For the current mid-evaluation design, weights load before execution, so the no-prefetch expert latency is:

`T_expert = T_load + T_execute = 48.774 ms + 7.570 ms = 56.344 ms`

## Interpretation boundary

The copy-only simulator mode substitutes transfer timing for all linear operations. The analysis therefore uses only its `expert_ffn` field and combines that with the normal GPU-compute run. Counting the copy-only attention or projection fields would incorrectly imply that non-expert weights were offloaded.

The synthetic trace is sufficient for the latency breakdown but cannot establish realistic spatial or temporal expert correlation. Correlation plots require router traces from real prompts; that remains a separate mid-evaluation task.

## Reproduction

On the VM, the driver is:

`/home/verso/ca-assignment/Project/SSD-offloading/LLMSimulator/run_day2_offload.sh`

Local analysis:

```powershell
python Project\scripts\analyze_day2.py
```

This writes `results/day2/summary.csv` and `results/day2/day2_breakdown.svg`.
