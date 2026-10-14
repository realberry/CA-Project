# Slide-ready results and captions

## Architecture slide

**Suggested headline:** Moving experts off chip changes MoE inference from a
compute problem into a data-movement problem.

The original Duplex run demonstrates GPU and Logic-PIM execution. The modified
experiment disables Logic-PIM for the expert path, keeps each expert unsharded,
stores expert weights in CPU DRAM, and executes selected experts on the GPU
after their weights reach HBM.

Technical precision: the simulated node has four H100 devices. Expert tensor
degree is 1, while non-expert tensor degree is 4. Therefore, `TP4` in raw
filenames describes non-expert tensor parallelism.

## Day 2 plot

**Suggested headline:** Expert loading takes 6.44x longer than expert execution.

**Caption:** Mean latency across seven matched Mixtral decode samples. Loading
selected experts from modeled CPU DRAM takes 48.774 ms, compared with 7.570 ms
for their GPU execution. Without prefetch overlap, the expert path is 56.344 ms
and loading contributes 86.6%.

**Explanation:** Sparse routing reduces how many experts execute but does not
reduce the size of each selected expert's matrices. Those weights must cross the
off-chip link before computation, leaving the GPU waiting for data.

## Day 3 plot

**Suggested headline:** Real Mixtral routing contains usable temporal and
cross-layer structure.

**Caption:** Analysis of 100,000 Mixtral token routes with separate 50,000-token
profile and held-out evaluation halves. Reusing the previous token's experts
achieves 36.2% recall. A fixed-budget adjacent-layer CCT achieves 41.5% recall,
compared with 25% random and 26.9% for a frequency-only predictor.

**Interpretation:** Routing decisions are not independent. The current layer
and the previous token both provide information that can support training-free
expert prefetching.

**Trace boundary:** AllenAI collected the routes on natural C4 text through
causal teacher-forced forward passes. Call them real Mixtral C4 routes or a
decode proxy, rather than sampled generated-output routes.

## Final mid-evaluation conclusion

The simulator reproduction works, off-chip expert loading is the dominant
latency, and real expert routes are predictable above random and
frequency-only baselines. The final-project stage will integrate this predictor
with a prefetcher and measure end-to-end stalls, traffic, and energy.

The defensible conclusion is that the expert-fetch problem exists and dominates
the modeled off-chip path. The correlation measurements motivate the next
stage, but the mid evaluation does not claim a completed prefetcher, measured
prefetch speedup, traffic reduction, or energy saving.
