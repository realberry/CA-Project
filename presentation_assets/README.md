# Presentation assets

This folder contains the curated figures and compact result files for the
mid-project presentation. The complete methodology and interpretation are in
`Project/PROJECT_README.md`.

The `final/` subdirectory contains the final-project figures generated from
the single-GPU, batch-1 calibration. Each figure is supplied as both SVG and
PNG:

- `final/primary_latency`: baseline, selected CCT policies, and oracle latency;
- `final/prefetch_sensitivity`: buffer-budget and overlap-window sweep;
- `final/energy_sensitivity`: total-energy change versus stalled-GPU power.

The final-project numbers, caveats, and reproduction commands are documented
in `docs/FINAL_DAY2_RESULTS.md`.

## Plots

- `plots/01_day2_memory_wall.png`: slide-ready raster figure for the loading
  versus execution result.
- `plots/01_day2_memory_wall.svg`: vector version for lossless scaling.
- `plots/02_day3_expert_correlation.png`: slide-ready temporal and spatial
  correlation figure.
- `plots/02_day3_expert_correlation.svg`: vector version.
- `plots/03_architecture_flow.png`: overview of the original and modified expert
  paths.
- `plots/03_architecture_flow.svg`: vector version.

## Results

- `results/key_results.csv`: the main numbers likely to appear in slides.
- `results/day2_full_summary.csv`: complete Day 2 aggregate statistics.
- `results/day3_summary.json`: complete Day 3 aggregate statistics and hashes.
- `results/day3_layer_metrics.csv`: per-layer correlation measurements.
- `results/PRESENTATION_RESULTS.md`: concise captions, talking points, and
  interpretation boundaries.

## Configurations

The `configs/` directory contains the exact Duplex, Ramulator, resident,
off-chip compute, and off-chip loading YAML files.

