# Final-project Day 2 results

## Final simulation configuration

The final hardware calibration uses the architecture requested in the project
specification:

- one modeled H100 GPU;
- batch size 1 for per-token latency;
- Mixtral top-2 routing across eight experts and 32 MoE layers;
- FP16 expert weights;
- expert and non-expert tensor degrees both set to 1;
- Logic-PIM disabled;
- all routed experts stored in CPU DRAM;
- a 450 GB/s modeled off-chip link.

Seven matched simulator samples were collected for the resident baseline,
off-chip GPU execution, and expert-weight copy configurations.

| Measurement | Mean latency |
|---|---:|
| GPU-resident expert execution | 6.731 ms |
| Off-chip-case expert execution | 7.775 ms |
| Off-chip expert loading | 50.115 ms |
| Off-chip expert path, no prefetch | 57.890 ms |
| GPU-resident decoder | 7.576 ms |
| Off-chip decoder, no prefetch | 58.866 ms |

At 450 GB/s, the measured transfer time corresponds to approximately 336 MiB
per expert. This agrees with the three large FP16 matrices in one Mixtral expert
and confirms that the batch-1 calibration is suitable for a per-token model.

## Selected prefetch design

The selected design is a training-free CCT predictor with one 336 MiB
next-layer buffer slot. At layer `L`, the two routed experts index a profiled
cross-layer correlation table. The highest-scoring expert for layer `L+1` is
prefetched while layer `L` executes. The entry remains until layer `L+1` and is
then consumed or discarded. This explicit eviction prevents free long-term
reuse.

The first 50,000 trace tokens construct the CCT. All reported results use the
next 50,000 held-out tokens. The prefetch window is limited to the measured
0.243 ms execution time of one expert layer.

| Metric | On demand | Selected CCT design |
|---|---:|---:|
| Prediction recall | — | 22.98% |
| Prediction precision | — | 45.96% |
| Exposed expert-loading stall | 50.115 ms | **46.541 ms** |
| Total modeled decoder latency | 58.866 ms | **55.292 ms** |
| Speedup | 1.000× | **1.065×** |
| Stall reduction | — | **7.13%** |
| Additional transfer traffic | 0% | **8.38%** |
| HBM prefetch buffer | 0 MiB | **336 MiB** |

The CCT top-2 predictor has 41.08% recall, but the one-slot policy is faster.
With a fixed link budget, two candidates split the overlap window and reduce the
useful progress made on each expert. This is the same motivation behind an
adaptive prefetch budget: prediction recall alone does not determine system
performance.

## Candidate-budget sensitivity

The realistic one-layer overlap window gives:

| Buffer slots | Buffer | Recall | Precision | Decoder latency | Stall reduction | Extra traffic |
|---:|---:|---:|---:|---:|---:|---:|
| **1** | **336 MiB** | 22.98% | 45.96% | **55.292 ms** | **7.13%** | **8.38%** |
| 2 | 672 MiB | 41.08% | 41.08% | 55.672 ms | 6.37% | 9.14% |
| 3 | 1,008 MiB | 55.34% | 36.89% | 55.997 ms | 5.72% | 9.79% |
| 4 | 1,344 MiB | 67.62% | 33.81% | 56.237 ms | 5.25% | 10.27% |

Longer artificial overlap windows improve performance, but they are reported as
sensitivity results rather than the primary claim. At four times the measured
window, the two-slot CCT reaches 46.090 ms; the simulator does not establish
that this much independent computation is actually available.

## Energy and traffic

The SSD-offloading CSVs report 0.953 J of compute dynamic energy and 2.417 J of
off-chip-transfer dynamic energy for the no-prefetch step. Correct prefetches
move the same required bytes earlier and therefore do not reduce dynamic
transfer energy. Incorrect predictions increase the selected policy's dynamic
energy to 3.573 J, a 6.01% increase over 3.370 J on demand.

Reduced stall time can still lower total energy if the GPU consumes power while
waiting. Because the simulator does not provide one fixed idle-power value, the
result is reported as a sensitivity:

| Assumed power during exposed stalls | Energy change with CCT |
|---:|---:|
| 0 W | 6.01% increase |
| 100 W | 1.85% saving |
| 200 W | 3.82% saving |
| 350 W | 5.01% saving |
| 500 W | 5.57% saving |
| 700 W | 5.98% saving |

The break-even stalled-GPU power is 56.7 W. The 350 W row is a sensitivity
point, not a measured physical-GPU power claim.

## Reproduction

Run the three LLMSimulator configurations inside the VM:

```bash
chmod +x scripts/run_final_single_gpu.sh
SIM_ROOT=/home/verso/ca-assignment/Project/SSD-offloading/LLMSimulator \
SUBMISSION_ROOT="$PWD" \
bash scripts/run_final_single_gpu.sh
```

Regenerate the calibration and trace-driven results from the repository root:

```bash
python3 scripts/analyze_single_gpu.py
python3 scripts/simulate_prefetch.py \
  --config configs/final_prefetch_single_gpu_batch1.json
python3 scripts/analyze_prefetch_sensitivity.py \
  --config configs/final_prefetch_single_gpu_batch1.json
```

## Scope boundary

LLMSimulator/SSD-offloading supplies the single-GPU latency and dynamic-energy
calibration. The routing-aware predictor, next-layer buffer, early-transfer
progress, eviction, exposed stalls, and incorrect traffic are implemented in
the trace-driven simulator because the offloading branch removes expert IDs
before producing aggregate copy timing. No physical GPU result is claimed.
