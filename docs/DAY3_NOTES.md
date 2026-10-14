# Day 3: Real-trace expert correlation

## Outcome

Real Mixtral-8x7B router decisions show both consecutive-token reuse and
adjacent-layer predictability. The analysis uses 100,000 token routes with a
strict 50,000-token profile / 50,000-token evaluation split. Predictor accuracy
is measured only on the held-out half.

At the same two-expert budget as Mixtral's router:

- The previous token's experts recover **36.2%** of the current token's experts,
  versus **25%** expected from uniform random routing.
- A cross-layer correlation table (CCT) recovers **41.5%** of the next layer's
  experts, versus **25%** at random and **26.9%** for always choosing each
  target layer's two most frequent experts.
- Taking the literal union of the two CCT entries raises recall to **58.0%**, at
  the cost of prefetching **3.37 experts** on average instead of two.
- Consecutive tokens share **0.724 experts per layer** on average, compared with
  the random-routing expectation of `K^2/N = 2^2/8 = 0.5`.

These results establish the mid-evaluation premise: Mixtral expert requests are
structured enough to predict before use, and the gain is not explained merely
by choosing globally popular experts.

## Trace provenance

- Dataset: [AllenAI `analysis_mixtral`](https://huggingface.co/datasets/allenai/analysis_mixtral)
- Generation code: [AllenAI OLMoE `run_moe_analysis.py`](https://github.com/allenai/OLMoE/blob/main/scripts/run_moe_analysis.py)
- Model: `mistralai/Mixtral-8x7B-v0.1`
- Source corpus: sampled C4 English validation text
- Source file: `c4_results.jsonl`, 405 rows, 173,934,249 bytes
- Source LFS SHA-256: `55c76a34adf54510ad0f64a161236da93fdc742ca8eecbbe776f21f8d7e0d1cd`
- Downloaded byte range: `0-25165823` (24 MiB)
- Local partial-file SHA-256: `4cb79ab27c967d008878981859f5883243c5890556fbaf4c85afc60f2797c6fe`
- Complete rows in the local range: 58, containing 118,784 token routes
- Analyzed prefix: exactly 100,000 routes

Each row contains `input_ids`, `predicted_token_ids`, and `exp_ids`. The routing
array has shape `[2048, 2, 32]`: 2 selected experts in each of 32 layers for
each of 2,048 tokens. Expert IDs range from 0 to 7.

## Method

### Temporal correlation / History Table

For each held-out token and layer, the predictor uses the two experts selected
for the preceding token in the same layer. Recall is the intersection with the
current token's two experts, divided by two. Pairs never cross independent
2,048-token trace rows.

Under uniform independent routing, two top-2 sets drawn from eight experts have
expected intersection `K^2/N = 0.5`, or 25% recall. The measured mean is 0.724,
or 36.2% recall: **1.45x the random expectation**.

### Spatial correlation / Cross-layer Correlation Table

The first 50,000 routes build an 8x8 expert co-activation matrix for every
adjacent pair of the 32 MoE layers. For a held-out token, the two experts in
layer `L` index the learned conditional counts for layer `L+1`.

The fixed-budget predictor sums the two conditional score vectors and chooses
the highest-scoring two target experts. Its 41.5% mean recall is **1.66x the
random expectation** and 14.7 percentage points above the frequency-only
baseline. The mean profile-set Cramer's V across the 31 transitions is 0.144.

The CCT-union measurement follows the basic table organization in ST-MoE more
literally: each source expert contributes its own top two target experts. The
union contains 2--4 candidates and averages 3.37.

## Connection to Day 2

Day 2 measured 48.774 ms of expert loading and 7.570 ms of expert execution,
for a 56.344 ms no-prefetch expert path. The correlation analysis indicates
that some requests may be identifiable early, but it does not by itself
determine how much loading latency can be hidden. This mid-project submission
therefore makes no prefetch speedup claim. The next implementation stage must
model prefetch issue time, transfer overlap, buffer capacity, evictions, and
wasted traffic before reporting end-to-end performance.

## Interpretation boundary

The routes are genuine outputs of Mixtral's learned gates on natural C4 text.
AllenAI collected all positions in a causal forward pass using ground-truth
tokens rather than sampling tokens autoregressively. Thus this is a real
causal-token routing trace and a useful decode proxy, but it should be described
as **teacher-forced C4 routing**, not as a generated-output decode trace.

The 24 MiB range is a deterministic prefix of the official dataset, rather than
a randomly sampled subset of its 405 rows. The held-out split protects predictor
evaluation from table-fitting leakage, but it does not measure variation across
application domains.

## Artifacts and reproduction

- `results/day3/day3_correlation.svg`: presentation-ready plot
- `results/day3/day3_correlation.png`: rendered plot
- `results/day3/layer_metrics.csv`: all layer-level measurements
- `results/day3/summary.json`: aggregate results and trace hashes
- `data/day3/c4_results.first24MiB.jsonl.part`: downloaded trace range

Run locally with no third-party Python packages:

```powershell
python Project\scripts\analyze_day3.py
```
