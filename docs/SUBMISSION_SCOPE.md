# Mid-evaluation submission scope

This package is the frozen artifact set for the mid-project evaluation of
Project 2, *The MoE Memory Wall: Predicting and Prefetching Experts for Faster
LLM Inference*.

The package demonstrates:

1. a working LLMSimulator/Duplex build and simulation;
2. a GPU-only, unsharded-expert path with expert weights in modeled off-chip
   CPU DRAM;
3. separate measurements of expert loading and expert execution;
4. preliminary spatial and temporal correlation analysis on real Mixtral
   routing decisions.

The principal finding is that expert fetching is a real bottleneck in the
modeled system: 48.774 ms loading versus 7.570 ms execution. The package does
not contain a completed simulator-integrated prefetcher and makes no measured
prefetch speedup or energy-saving claim.

The files under `presentation_assets/` are the exact plots, compact tables, and
configurations intended for the demonstration. They are copied from the same
source results included elsewhere in the package.

Because submitted files may not be modified between the deadline and the demo,
the final ZIP should be treated as read-only after creation. Any presentation
used during the demo should use the included figures and values without
regenerating or editing them after submission.

