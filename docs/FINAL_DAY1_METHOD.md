# Final-project Day 1 prefetch experiment

This experiment converts the mid-evaluation routing and latency measurements
into a fixed-budget prefetch timing comparison. It is deliberately separate
from the validated C++ LLMSimulator tree because that offloading branch removes
expert identities before producing aggregate copy timing.

## Calibration

- Off-chip selected-expert loading: 48.7737327 ms per decoder step.
- Off-chip-case expert execution: 7.5702882 ms per decoder step.
- Off-chip no-prefetch decoder: 56.970 ms per decoder step.
- 32 Mixtral MoE layers and two selected experts per layer.
- Per-expert, per-layer transfer: 48.7737327 / (32 × 2) ms.

The no-prefetch policy is checked in code to reconstruct 56.970 ms exactly.

## Policies

- `no_prefetch`: all selected weights are loaded on demand.
- `static`: the two most frequent experts for each layer in the profile split.
- `ht`: the two experts used at the same layer for the previous token.
- `cct`: two experts selected from the profiled adjacent-layer correlation
  table, using the current token's preceding layer.
- `hybrid`: CCT scores augmented by previous-token history.
- `oracle`: the two actual experts, subject to the same transfer and overlap
  budget as every predictor.

The first 50,000 trace tokens build the tables. The next 50,000 are held out.
No predictor is trained on evaluation routes.

## Buffer and overlap model

Each policy has a two-candidate buffer for the next layer. Transfers may use
only the measured execution time of one expert layer as their overlap window.
That budget is split equally across the two candidates. Correctly predicted
bytes reduce exposed loading at the next layer. Incorrect bytes are discarded
at the layer boundary and counted as wasted traffic. There is no free
long-term cache reuse.

Run from the repository root:

```bash
python3 scripts/simulate_prefetch.py --config configs/final_prefetch.json
```

Outputs are written to `results/final_day1/`.

## Day 1 held-out results

| Policy | Recall | Mean decoder latency | Speedup | Wasted transfer vs. demand |
|---|---:|---:|---:|---:|
| No prefetch | — | 56.970 ms | 1.000× | 0.00% |
| Static top-2 | 26.88% | 54.935 ms | 1.037× | 11.35% |
| HT | 36.22% | 54.228 ms | 1.051× | 9.90% |
| CCT | **41.08%** | **53.860 ms** | **1.058×** | **9.15%** |
| HT+CCT hybrid | 36.52% | 54.205 ms | 1.051× | 9.85% |
| Oracle | 100.00% | 49.400 ms | 1.153× | 0.00% |

CCT is the strongest evaluated real predictor under the fixed two-candidate
budget. It reduces modeled decoder latency by 5.46% relative to on-demand
loading. The oracle is constrained by the same overlap window and therefore
sets a 13.29% latency-reduction ceiling for this particular timing assumption.
The remaining gap comes from both prediction errors and the fact that one
layer's 0.237 ms execution window is much shorter than the 0.762 ms required
to transfer one complete expert.

These are measured-calibrated trace-simulation results, not a new physical-GPU
measurement and not a cycle-level C++ scheduler result. Day 2 should evaluate
lead-time and candidate-budget sensitivity before selecting the final claim.
