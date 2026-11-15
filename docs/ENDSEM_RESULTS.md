# End-semester results

This document is the concise technical reference for the final implementation.
The presentation order and commands are in `DEMO_GUIDE.md`; machine-readable
values are under `results/final_endsem/`.

## Primary configuration

- one modeled H100, FP16, batch 1 and context length 1,024;
- Mixtral top-2 routing across 32 layers and eight experts;
- 50,000 profile routes and 50,000 disjoint evaluation routes;
- 16 expert slots (5,377 MiB) reserved for the persistent HBM cache;
- two CCT candidates issued one layer ahead;
- transfer overlap limited to one measured expert-layer execution period;
- 450 GB/s modeled CPU-DRAM-to-HBM link.

## Component ablation

| Scenario | Prediction recall | Prefetch coverage | HBM hit | Exposed stall | Token latency | Speedup | Traffic change |
|---|---:|---:|---:|---:|---:|---:|---:|
| On demand + LRU | — | — | 0.00% | 50.115 ms | 58.866 ms | 1.000× | 0.00% |
| Prediction only | 41.53% | — | 0.00% | 50.115 ms | 58.866 ms | 1.000× | 0.00% |
| CCT prefetch + LRU | 41.53% | 41.53% | 0.00% | 46.986 ms | 55.737 ms | 1.056× | +8.79% |
| CCT prefetch + CAR | 41.53% | 39.16% | 4.68% | **45.047 ms** | **53.798 ms** | **1.094×** | +4.91% |

Prediction alone deliberately has identical latency to the baseline. CCT+LRU
isolates the benefit of early transfer. The complete CCT+CAR design additionally
retains experts based on immediate CCT targets, previous-token history, profile
frequency, layer proximity, and LRU tie-breaking.

CAR reports slightly lower prefetch coverage because 2.37% of eligible correct
predictions are already resident and require no transfer. Its HBM hits reduce
demand traffic enough to offset part of the inaccurate prefetch traffic.

## Prefetch effectiveness

At the primary one-layer overlap, every issued correct prefetch remains partial
at demand: the timely rate is 0% and the late rate is 100%. These late
prefetches are still useful because transferred bytes reduce the remaining
on-demand stall. The complete design has:

- 39.16% transfer-based coverage;
- 63.83% unused-prefetch traffic rate;
- 45.047 ms exposed expert-load latency per token;
- 4.68% HBM hit rate;
- 4.91% total additional memory traffic.

At 4× overlap, 4.18% of issued correct prefetches become timely and speedup
rises to 1.289×, but additional traffic rises to 32.69%. This is a sensitivity
result, not the primary claim.

## HBM, aggressiveness and overlap sensitivity

| Dimension | Setting | Complete-design speedup | HBM hit | Traffic change |
|---|---:|---:|---:|---:|
| HBM capacity | 8 experts | 1.059× | 1.08% | +8.47% |
| HBM capacity | **16 experts** | **1.094×** | 4.63% | +4.89% |
| HBM capacity | 32 experts | 1.170× | 11.34% | −2.12% |
| Candidate budget | 1 | 1.093× | 5.40% | +4.19% |
| Candidate budget | **2** | **1.094×** | 4.63% | +4.89% |
| Candidate budget | 4 | 1.076× | 3.33% | +6.74% |
| Overlap | 0.5× | 1.067× | 4.63% | +0.13% |
| Overlap | **1×** | **1.094×** | 4.63% | +4.89% |
| Overlap | 2× | 1.154× | 4.63% | +14.41% |
| Overlap | 4× | 1.289× | 5.99% | +32.69% |

Four candidates divide the fixed transfer window and create more unused
traffic, so aggressiveness degrades performance relative to the selected
two-candidate setting. More HBM capacity improves retention and can eliminate
net extra traffic.

## Batch and context sensitivity

Nine matched LLMSimulator calibrations cover batch sizes 1, 4 and 8 and input
contexts 512, 1,024 and 2,048. Each point contains seven samples for the
resident, off-chip-compute and weight-copy cases.

| Batch | Complete-design speedup range | Normalized latency range |
|---:|---:|---:|
| 1 | 1.0943–1.0944× | 53.765–53.838 ms/sequence |
| 4 | 1.0925–1.0927× | 36.303–36.376 ms/sequence |
| 8 | 1.1232–1.1236× | 23.028–23.101 ms/sequence |

Increasing context from 512 to 2,048 adds only a small latency change in this
decode simulation. Batching improves normalized per-sequence latency. At batch
4 and 8, the union of real per-sequence predictions covers all demanded expert
IDs, but transfers remain partial under the primary one-layer window.

## Energy

The primary complete design increases measured dynamic energy by 3.52% because
of extra transfers. When 350 W is explicitly assumed during exposed stalls,
the shorter stall time gives a 7.92% total-energy saving. The value is an energy
sensitivity point, not a measured physical-GPU idle-power claim.

## Scope

LLMSimulator supplies hardware latency and dynamic-energy calibration. The
training-free predictor, real routing events, early-transfer progress,
persistent HBM state, LRU/CAR eviction and effectiveness metrics are implemented
in the trace-driven simulator because the SSD-offloading timing path aggregates
expert IDs before producing its CSV timing output.
