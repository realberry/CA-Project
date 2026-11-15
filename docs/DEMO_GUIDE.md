# Project demonstration guide

This is the concise demonstration path for the complete project. All commands
run from the repository root. The submitted files and the files used in the
demo are the same.

## 1. Start with the artifact check

```bash
python3 scripts/demo.py check
python3 scripts/demo.py summary
```

`check` verifies every canonical table needed by the demo. `summary` prints the
main mid-semester and end-semester results without rerunning a long experiment.

## 2. Explain the project in five results

1. **Memory wall:** `results/day2/day2_breakdown.png` shows that off-chip expert
   loading dominates expert execution. This is the motivating mid-semester
   result.
2. **Predictability:** `results/day3/day3_correlation.png` shows temporal and
   adjacent-layer correlation on real Mixtral routes. The first 50,000 tokens
   profile the training-free tables and the next 50,000 are held out.
3. **Component contribution:**
   `results/final_endsem/hbm_policies/hbm_policy_comparison.png` compares
   on-demand LRU, prediction alone, CCT prefetch with LRU, and the complete CCT
   prefetch plus correlation-aware retention design.
4. **When prefetching helps:**
   `results/final_endsem/sensitivity/endsem_sensitivity.png` varies HBM capacity,
   candidate budget, and overlap. The realistic one-layer window produces
   partial/late prefetches; longer windows produce timely completions but can
   increase unused traffic.
5. **Batch and context:**
   `results/final_endsem/batch_context_prefetch/batch_context_prefetch.png`
   combines nine matching LLMSimulator calibrations with real routing traffic.

The energy trade-off is shown separately in
`results/final_day2/single_gpu_batch1_prefetch/energy_sensitivity.png`.

## 3. Map deliverables to artifacts

| Deliverable | Implementation | Main evidence |
|---|---|---|
| Training-free prediction | CCT, previous-token history, static and hybrid predictors | `scripts/simulate_prefetch.py`, `results/day3/summary.json` |
| Predictor-driven prefetch | Next-layer issue during current expert execution | `scripts/evaluate_hbm_policies.py` |
| Effectiveness metrics | Coverage, timely/late/unused, HBM hits, traffic, stalls and latency | `results/final_endsem/hbm_policies/hbm_policy_summary.csv` |
| HBM management | Standard LRU versus correlation-aware retention (CAR) | HBM policy comparison plot and CSV |
| Component ablation | Baseline, prediction only, prefetch+LRU, complete CAR | HBM policy comparison plot and CSV |
| Sensitivity | Capacity, candidate budget, overlap, batch and context | `results/final_endsem/sensitivity/`, `batch_context_prefetch/` |
| Energy | Measured dynamic energy plus stalled-GPU-power sensitivity | Energy CSV and plot |
| Real routing traffic | AllenAI Mixtral trace drives predictor and HBM events | Trace hash in result JSON files |

## 4. Reproduce individual stages

Download the external trace once:

```bash
bash scripts/fetch_mixtral_trace.sh
```

Then run only the stage being demonstrated:

```bash
python3 scripts/demo.py run mid
python3 scripts/demo.py run primary
python3 scripts/demo.py run matrix
python3 scripts/demo.py run sensitivity
```

`primary` and `sensitivity` are CPU-bound trace evaluations and take longer than
the summary command. The VM hardware matrix is reproduced separately with:

```bash
bash scripts/run_endsem_matrix.sh
```

The VM matrix uses one modeled H100, no physical GPU, unsharded experts, and no
Logic-PIM. Its canonical experiment dimensions are in
`configs/endsem_experiments.json`.

## 5. Key interpretation boundaries

- LLMSimulator supplies measured latency and dynamic-energy calibration.
- The predictor, prefetch timing, persistent HBM cache, LRU and CAR eviction,
  and real routing traffic are evaluated by the trace-driven simulator.
- Teacher-forced C4 Mixtral routes are real learned-gate decisions but are not
  free-running generated-token traces.
- At the primary overlap window, transfers begin early but generally remain
  incomplete at demand; they reduce stalls even though they are classified as
  late rather than timely.
- Energy savings depend on stalled-GPU power. Dynamic transfer energy alone can
  increase when inaccurate prefetches add traffic.
