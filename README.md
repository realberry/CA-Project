# The MoE Memory Wall — Mid-Project Submission

> **GitHub repository:** <https://github.com/realberry/CA-Project>

## Executive summary

This submission documents the work completed for the mid-project evaluation.
The primary conclusion at this stage is deliberately limited: **when Mixtral
expert weights are placed in modeled off-chip CPU DRAM, fetching the selected
experts becomes a clear performance bottleneck**.

The measured expert-weight loading time is 48.774 ms, while the corresponding
GPU expert execution time is 7.570 ms. Loading is therefore 6.44 times longer
than execution and contributes 86.6% of the 56.344 ms no-prefetch expert path.
This verifies the memory-wall problem that motivates the remainder of the
project.

The submission also includes the correlation characterization requested for
the mid evaluation. Real Mixtral routing decisions show above-random temporal
and adjacent-layer structure. These measurements are preliminary evidence that
prediction may be possible; they are not evidence of a completed prefetcher.
No end-to-end prefetch speedup, stall reduction, traffic overhead, or energy
saving is claimed in this submission.

## 1. Project objective

Mixture-of-Experts (MoE) models activate only a small number of experts for
each token. This reduces computation, but the complete expert pool is too large
to keep in fast GPU memory. If expert weights are stored in slower off-chip
memory, every routing decision can trigger a large weight transfer before the
selected expert can execute.

The mid-project work answers two questions:

1. Does loading selected experts dominate their GPU computation?
2. Are future expert requests predictable from earlier layers or tokens?

The work uses LLMSimulator, its Ramulator 2.0 integration, and the
SSD-offloading extension. All hardware results are simulation results; no
physical GPU is required.

## 2. Mid-evaluation requirements and completion status

| Required item | Work completed | Main evidence |
|---|---|---|
| Reproduce Duplex and demonstrate a working simulation | Built LLMSimulator and ran Mixtral with both GPU and Logic-PIM operations. Also completed a reduced Ramulator-backed run. | `DAY1_NOTES.md`, `results/day1/` |
| Disable expert sharding and Logic-PIM; store experts off chip | Used the SSD-offloading branch with `processor_type: GPU`, expert tensor degree 1, and expert offload to CPU DRAM. | `DAY2_NOTES.md`, `configs/day2_*.yaml` |
| Measure and explain loading versus execution | Measured separate GPU-compute and expert-weight-copy runs and combined matched samples into a no-prefetch critical path. | `results/day2/summary.csv`, Day 2 plot |
| Analyze temporal and spatial expert correlation | Analyzed 100,000 real Mixtral routes with separate 50,000-token profile and evaluation sets. | `DAY3_NOTES.md`, Day 3 plot |

All four items listed for the mid evaluation have supporting artifacts.
Predictor-integrated simulator prefetching, end-to-end stall reduction, traffic
overhead, and energy savings belong to the final-project stage.

## 3. Software and experiment provenance

### Simulator revisions

- Day 1 LLMSimulator commit: `419252761fbdb95b789778a02256d458a5537ec7`
- Day 2 SSD-offloading wrapper: `99f3903d67cda89471881a531dd57420abc6e8be`
- Day 2 LLMSimulator: `3a0ea80c343ac52b67e0945fca7bbdb8a7a4adef`
- Ramulator 2.0: `b5262869b24790e5ec62d9d9974fca9bbb4f54e0`
- Compiler: GCC/G++ 13.3.0

Ramulator required one compiler-compatibility patch. Its `src/base/utils.h`
used `uint64_t` without including `<cstdint>`. The patch in
`patches/ramulator2-gcc13-cstdint.patch` restores that header and does not alter
simulator behavior.

### Model and numerical format

- Model: Mixtral-8x7B family configuration
- Routing: top 2 of 8 experts per MoE layer
- Precision: FP16 (`precision_byte: 2`)
- Primary phase: autoregressive decode

## 4. Day 1 — Duplex reproduction

### Purpose

Day 1 established that the provided simulator and both compute paths work before
changing the architecture. Duplex dynamically places operations on conventional
GPU resources or Logic-PIM resources according to their characteristics.

### Build and baseline

The repository baseline first ran unchanged and exited successfully. This was a
build smoke test and used the repository's default GPU-only DeepSeek V3
configuration.

The actual Duplex smoke configuration then used:

- Mixtral, FP16
- one node with four H100-class devices
- `processor_type: GPU+LOGIC`
- expert tensor degree 1
- synthetic routing with zero skew
- decode mode, input length 2,048
- analytical memory timing

It completed with exit status 0. The operation trace contained 96 GPU-tagged
operations and 320 Logic-tagged operations, demonstrating that both Duplex
execution paths were active.

A second reduced test enabled detailed Ramulator memory timing:

- input length 128
- batch size 4
- two iterations
- `use_ramulator: on`

It also exited successfully and exercised the same two processor paths. The
reduced size was chosen because detailed DRAM simulation is much slower than the
analytical timing model.

### Day 1 conclusion

The LLMSimulator build, Duplex GPU/Logic dispatch, and Ramulator-backed memory
path are operational. This provides the trusted baseline for the architectural
modification.

## 5. Day 2 — Off-chip expert memory and the memory wall

### Modified architecture

The experiment changes the expert path as follows:

1. The Mixtral router selects two experts for a token.
2. The selected expert weights reside in off-chip CPU DRAM.
3. Their weights transfer to GPU HBM over the modeled NVLink 4 path.
4. The expert executes on the GPU after its weights arrive.
5. The system repeats this process for subsequent MoE layers and tokens.

Logic-PIM is absent because `processor_type` is `GPU`. Expert sharding is
disabled using `expert_tensor_degree: 1`, so an expert matrix is not split
across GPUs. The simulated node still contains four H100 devices and retains
`none_expert_tensor_degree: 4` for non-expert operations. Consequently, raw
filenames contain `TP4`; this refers to non-expert tensor parallelism, not
expert sharding.

The off-chip link is modeled by the SSD-offloading branch's NVLink 4 setting,
corresponding to 450 GB/s. The off-chip store uses its CPU-memory/DDR5 energy
model.

### Matched configurations

All three Day 2 runs use:

- Mixtral, FP16, decode mode
- input length 1,024
- requested output length / iterations: 8 / 8
- batch size 4
- one node, four H100 devices
- expert tensor degree 1
- non-expert tensor degree 4
- synthetic workload trace

The simulator emitted seven matched decode-step samples for each run.

### Three-run measurement method

1. **GPU-resident baseline:** offload disabled and normal GPU compute collected.
2. **Off-chip compute:** CPU-memory offload enabled while normal GPU compute was
   collected. The model reserves bandwidth for incoming weights.
3. **Off-chip load:** CPU-memory offload enabled in weight-copy collection mode.
   Only the `expert_ffn` field is used as expert-weight transfer latency.

The copy-only mode replaces linear-operation computation with transfer timing,
so its attention and projection fields must not be added to the result. The
no-prefetch expert path is formed from matched measurements:

`T_expert = T_load + T_execute`

### Day 2 results

| Measurement | Mean | Standard deviation |
|---|---:|---:|
| GPU-resident expert execution | 6.554 ms | 0.117 ms |
| Off-chip-case expert execution | 7.570 ms | 0.135 ms |
| Off-chip expert loading | **48.774 ms** | 0.871 ms |
| Off-chip expert path, no prefetch | **56.344 ms** | 1.006 ms |
| Off-chip MoE decoder, no prefetch | 56.970 ms | 1.006 ms |

Loading is **6.44 times** expert execution and accounts for **86.6%** of the
no-prefetch expert path. The off-chip-case execution time is 15.5% above the
resident baseline because 450 GB/s of HBM bandwidth is reserved for concurrent
weight arrival.

### Why loading dominates

Each selected Mixtral expert is a large feed-forward network containing multiple
weight matrices. Sparse activation reduces the number of experts computed, but
it does not make the selected matrices smaller. In the modified architecture,
those matrices must cross the slower off-chip link before useful GPU arithmetic
can begin. Decode also has limited per-token parallelism, making it difficult to
hide this transfer automatically. The GPU therefore waits for data even though
the expert calculation itself is relatively short.

This is the measured MoE memory wall: performance is constrained by expert
weight movement rather than arithmetic throughput.

## 6. Day 3 — Temporal and spatial expert correlation

### Trace source and boundary

Day 3 uses AllenAI's official `analysis_mixtral` dataset, generated from
`mistralai/Mixtral-8x7B-v0.1` on sampled C4 English validation text. Each row
contains 2,048 token positions and an expert-routing array of shape
`[2048, 2, 32]`: top-2 experts across 32 Mixtral layers.

The routes are genuine outputs of Mixtral's learned gates. They were collected
for all positions in causal, teacher-forced forward passes rather than by
sampling generated output tokens. They should therefore be described as real
teacher-forced C4 routing or a decode proxy, not as a generated-output trace.

The analysis uses exactly 100,000 routes:

- first 50,000: construct correlation tables
- second 50,000: held-out predictor evaluation

This separation prevents the predictor from being evaluated on the same routes
used to build its table.

### Temporal correlation

The History Table (HT) predictor uses the previous token's two selected experts
at the same layer. For top-2 routing among eight uniformly random experts, the
expected intersection is:

`E[overlap] = K^2/N = 2^2/8 = 0.5 expert`

Measured consecutive tokens share **0.724 expert** per layer on average. This is
**36.2% recall** at a two-expert budget, compared with **25% random recall**.

### Spatial correlation

For each adjacent layer pair, the profile half constructs an 8x8 expert
co-activation matrix. During evaluation, the two experts selected at layer `L`
index their learned conditional target scores for layer `L+1`. The scores are
summed and the best two next-layer experts are predicted.

| Predictor | Held-out recall | Candidate budget |
|---|---:|---:|
| Random top-2 | 25.0% | 2 |
| Always choose layer's most frequent experts | 26.9% | 2 |
| Cross-layer correlation table | **41.5%** | 2 |
| Union of both CCT entries | **58.0%** | 3.37 average |

The fixed-budget CCT is 1.66 times random recall and exceeds the
frequency-only baseline by 14.7 percentage points. This shows that the result
comes from knowledge of the current layer's routing, rather than only global
expert popularity.

### Interpretation at the mid-project stage

The correlation results establish only that expert selections contain
structure above simple baselines. A complete prefetch implementation must still
model when predictions are issued, whether transfers overlap useful work, HBM
capacity, evictions, prediction errors, and wasted memory traffic. Consequently,
this submission reports no prefetch latency or speedup.

## 7. Complete setup and reproduction instructions

### 7.1 Tested environment

The simulator runs inside a Linux VM. The result-analysis scripts may run on
Linux or Windows because they use only the Python standard library.

| Item | Tested value |
|---|---|
| Guest operating system | Linux Mint / Ubuntu 24.04 base |
| VM resources | 8 vCPUs, approximately 15 GiB RAM |
| Compiler | GCC/G++ 13.3.0 |
| Build system | CMake and Make |
| Python | Python 3 |
| Physical GPU | Not required |

Install the required Linux packages:

```bash
sudo apt update
sudo apt install -y git build-essential cmake clang libyaml-cpp-dev python3
```

The upstream repositories list CMake, G++, and Clang as prerequisites. The
submitted runs used GCC/G++ 13.3.0 successfully.

### 7.2 Clone this submission repository

Clone the same repository linked in the final report:

```bash
mkdir -p ~/ca-assignment
cd ~/ca-assignment
git clone https://github.com/realberry/CA-Project.git submission
export SUBMISSION_ROOT="$HOME/ca-assignment/submission"
```

The commands below assume that the submitted `README.md`, `configs/`,
`scripts/`, `patches/`, `results/`, and `data/` directories are directly under
`$SUBMISSION_ROOT`.

### 7.3 Build the Day 1 LLMSimulator/Duplex baseline

Clone the upstream simulator and select the revision used for Day 1:

```bash
mkdir -p ~/ca-assignment/Project
cd ~/ca-assignment/Project
git clone https://github.com/scale-snu/LLMSimulator.git
cd LLMSimulator
git checkout 419252761fbdb95b789778a02256d458a5537ec7
git submodule update --init --recursive
```

Apply the upstream Logic-PIM Ramulator patch, then the submitted GCC 13
compatibility patch:

```bash
cd ~/ca-assignment/Project/LLMSimulator
git -C src/dram/ramulator2 apply ../../../patch/ramulator2_pim.patch
git -C src/dram/ramulator2 apply "$SUBMISSION_ROOT/patches/ramulator2-gcc13-cstdint.patch"
```

Build the simulator:

```bash
cd ~/ca-assignment/Project/LLMSimulator
mkdir -p build
cd build
cmake ..
make -j"$(nproc)"
```

The expected executable is:

```text
~/ca-assignment/Project/LLMSimulator/build/run
```

Verify the revisions before running:

```bash
git -C ~/ca-assignment/Project/LLMSimulator rev-parse HEAD
git -C ~/ca-assignment/Project/LLMSimulator/src/dram/ramulator2 rev-parse HEAD
```

They should report:

```text
419252761fbdb95b789778a02256d458a5537ec7
b5262869b24790e5ec62d9d9974fca9bbb4f54e0
```

### 7.4 Install the Day 1 configurations and run scripts

The Day 1 drivers use the tested path `~/ca-assignment/Project`. Copy the exact
submitted files into that layout:

```bash
cp "$SUBMISSION_ROOT/configs/duplex_smoke.yaml" \
  ~/ca-assignment/Project/duplex_smoke.yaml
cp "$SUBMISSION_ROOT/configs/duplex_ramulator_smoke.yaml" \
  ~/ca-assignment/Project/duplex_ramulator_smoke.yaml

cp "$SUBMISSION_ROOT/scripts/run_day1_baseline.sh" \
  ~/ca-assignment/Project/LLMSimulator/run_day1_baseline.sh
cp "$SUBMISSION_ROOT/scripts/run_day1_duplex.sh" \
  ~/ca-assignment/Project/LLMSimulator/run_day1_duplex.sh
cp "$SUBMISSION_ROOT/scripts/run_day1_ramulator.sh" \
  ~/ca-assignment/Project/LLMSimulator/run_day1_ramulator.sh

chmod +x ~/ca-assignment/Project/LLMSimulator/run_day1_*.sh
```

Run the untouched repository baseline, the analytical Duplex smoke test, and
the reduced Ramulator-backed test:

```bash
bash ~/ca-assignment/Project/LLMSimulator/run_day1_baseline.sh
bash ~/ca-assignment/Project/LLMSimulator/run_day1_duplex.sh
bash ~/ca-assignment/Project/LLMSimulator/run_day1_ramulator.sh
```

Check that every run returned status 0:

```bash
cat ~/ca-assignment/Project/day1/baseline.status
cat ~/ca-assignment/Project/day1/duplex-smoke.status
cat ~/ca-assignment/Project/day1/duplex-ramulator-smoke.status
```

The analytical and Ramulator-backed logs should contain both GPU-tagged and
Logic-tagged operations. Copies of the verified outputs are under
`results/day1/` in the submission.

### 7.5 Build the Day 2 SSD-offloading simulator

Clone the wrapper, select its recorded revision, and initialize its submodules:

```bash
cd ~/ca-assignment/Project
git clone https://github.com/scale-snu/SSD-offloading.git
cd SSD-offloading
git checkout 99f3903d67cda89471881a531dd57420abc6e8be
git submodule update --init --recursive
./patches.sh
```

Confirm the nested LLMSimulator revision:

```bash
git -C ~/ca-assignment/Project/SSD-offloading/LLMSimulator rev-parse HEAD
```

The expected revision is:

```text
3a0ea80c343ac52b67e0945fca7bbdb8a7a4adef
```

If the GCC 13 `<cstdint>` compatibility change is not already present after
`patches.sh`, apply it once:

```bash
cd ~/ca-assignment/Project/SSD-offloading/LLMSimulator
git -C src/dram/ramulator2 apply \
  "$SUBMISSION_ROOT/patches/ramulator2-gcc13-cstdint.patch"
```

Build the nested simulator:

```bash
cd ~/ca-assignment/Project/SSD-offloading/LLMSimulator
mkdir -p build
cd build
cmake ..
make -j"$(nproc)"
```

### 7.6 Install and run the Day 2 configurations

The submitted driver uses the exact tested path
`/home/verso/ca-assignment/Project/SSD-offloading/LLMSimulator`. If the Linux
username is not `verso`, edit only the `root=` line at the top of
`run_day2_offload.sh` to match the actual absolute path.

```bash
export SSD_SIM="$HOME/ca-assignment/Project/SSD-offloading/LLMSimulator"
mkdir -p "$SSD_SIM/day2_configs" "$SSD_SIM/results/day2/raw"

cp "$SUBMISSION_ROOT/configs/day2_baseline.yaml" \
  "$SSD_SIM/day2_configs/day2_baseline.yaml"
cp "$SUBMISSION_ROOT/configs/day2_offchip_compute.yaml" \
  "$SSD_SIM/day2_configs/day2_offchip_compute.yaml"
cp "$SUBMISSION_ROOT/configs/day2_offchip_load.yaml" \
  "$SSD_SIM/day2_configs/day2_offchip_load.yaml"
cp "$SUBMISSION_ROOT/scripts/run_day2_offload.sh" \
  "$SSD_SIM/run_day2_offload.sh"
chmod +x "$SSD_SIM/run_day2_offload.sh"

bash "$SSD_SIM/run_day2_offload.sh"
```

Verify completion and inspect the recorded stages:

```bash
test -f "$SSD_SIM/results/day2/SUCCESS" && echo "ALL THREE RUNS COMPLETED"
cat "$SSD_SIM/results/day2/"*.status
find "$SSD_SIM/results/day2/raw" -maxdepth 1 -type f -printf '%f\n' | sort
```

Expected raw outputs are one GPU-resident compute CSV, one CPU-memory/GPU-compute
CSV, and one CPU-memory expert-weight-copy CSV. The submitted copies are under
`results/day2/raw/`.

### 7.7 Regenerate the Day 2 tables and plot

Copy fresh raw CSVs into the submission tree if the simulation was rerun:

```bash
mkdir -p "$SUBMISSION_ROOT/results/day2/raw"
cp "$SSD_SIM/results/day2/raw/"*.csv "$SUBMISSION_ROOT/results/day2/raw/"
```

Run the standard-library analysis:

```bash
cd "$SUBMISSION_ROOT"
python3 scripts/analyze_day2.py
```

Expected outputs:

```text
results/day2/summary.csv
results/day2/day2_breakdown.svg
```

The expected aggregate values are approximately 48.774 ms expert loading,
7.570 ms off-chip-case expert execution, and 56.344 ms for the no-prefetch
expert path.

### 7.8 Regenerate the Day 3 correlation results

The submission already contains the deterministic 24 MiB range of AllenAI's
Mixtral routing trace at:

```text
data/day3/c4_results.first24MiB.jsonl.part
```

Its SHA-256 digest is:

```text
4cb79ab27c967d008878981859f5883243c5890556fbaf4c85afc60f2797c6fe
```

Run:

```bash
cd "$SUBMISSION_ROOT"
python3 scripts/analyze_day3.py
```

Expected outputs:

```text
results/day3/summary.json
results/day3/layer_metrics.csv
results/day3/day3_correlation.svg
```

The expected aggregate results are 36.2% previous-token recall and 41.5%
fixed-budget CCT recall, both evaluated on the held-out 50,000-token half.

### 7.9 Reproduce only the submitted processed results

The slow simulator runs are unnecessary if the goal is only to verify the
submitted analysis. After cloning the group repository, run:

```bash
cd "$SUBMISSION_ROOT"
python3 scripts/analyze_day2.py
python3 scripts/analyze_day3.py
```

Both scripts read the submitted raw data and overwrite only their derived
CSV/JSON/SVG outputs.

On Windows PowerShell, the equivalent commands are:

```powershell
cd <path-to-cloned-repository>
python scripts\analyze_day2.py
python scripts\analyze_day3.py
```

### 7.10 Troubleshooting

- **`uint64_t` build errors in Ramulator:** apply
  `patches/ramulator2-gcc13-cstdint.patch` once. Do not apply it repeatedly.
- **`YAML::BadFile: config.yaml`:** run the executable from the LLMSimulator
  repository root, or use the submitted driver, which changes to the required
  directory before execution.
- **`SIGFPE` with the four-device configuration:** use batch size 4 or larger.
  Batch size 1 produces zero sequences per data-parallel device in this build.
- **Day 2 filename contains `TP4`:** this denotes non-expert tensor degree 4.
  The expert tensor degree is 1, so expert weights are not sharded.
- **No physical GPU is detected:** expected. These are simulator executions and
  do not require CUDA or GPU passthrough.
- **Different absolute VM path:** update the `root=` variable in
  `scripts/run_day2_offload.sh` before running it.

## 8. Repository layout

```text
Project/
├── PROJECT_README.md               # This complete mid-project description
├── DAY1_NOTES.md                   # Duplex build and reproduction evidence
├── DAY2_NOTES.md                   # Off-chip architecture and latency analysis
├── DAY3_NOTES.md                   # Real-route correlation methodology
├── configs/                        # Exact YAML configurations
├── data/day3/                      # Saved official routing-trace range
├── patches/                        # Ramulator GCC compatibility patch
├── scripts/                        # VM drivers and local analysis scripts
├── results/day1/                   # Build, logs, status, and simulator output
├── results/day2/                   # Raw latency CSVs, summary, and plot
├── results/day3/                   # Correlation metrics and plot
└── mid_submission/
    └── presentation_assets/        # Curated files for making presentation slides
```

## 9. Mid-evaluation conclusions

1. The provided Duplex simulator works with GPU, Logic-PIM, and detailed
   Ramulator memory timing.
2. Moving unsharded experts to off-chip DRAM creates a clear memory bottleneck:
   48.774 ms loading versus 7.570 ms execution.
3. Preliminary routing analysis shows above-random temporal and adjacent-layer
   correlation. This motivates testing a training-free prefetcher during the
   final-project stage.
4. No prefetcher has yet been integrated into the simulator, so no end-to-end
   performance or energy improvement is claimed.

## 10. Remaining final-project work

- Integrate the predictor into the simulation path.
- Issue expert transfers early and model overlap with useful computation.
- Manage HBM expert residency and evictions.
- Measure total/token latency and exposed loading stalls.
- Quantify incorrect-prefetch traffic and buffer pressure.
- Measure energy consumption and savings.
- Compare the completed prefetch design against on-demand loading.

## 11. References and local papers

1. S. Yun et al., *Duplex: A Device for Large Language Models with Mixture of
   Experts, Grouped Query Attention, and Continuous Batching*, MICRO 2024,
   <https://doi.org/10.1109/MICRO61859.2024.00105>.
2. Y. Zhao et al., *A Spatio-Temporal Expert Prefetching Framework for
   Efficient MoE-based LLM Inference*, arXiv:2606.15453, 2026,
   <https://arxiv.org/abs/2606.15453>.
3. A. Kanani et al., *APEX: Adaptive Expert Prefetching for Memory-Efficient
   Edge MoE Inference*, arXiv:2608.11688, 2026,
   <https://arxiv.org/abs/2608.11688>.
4. LLMSimulator: <https://github.com/scale-snu/LLMSimulator>
5. SSD-offloading: <https://github.com/scale-snu/SSD-offloading>
6. AllenAI Mixtral routing trace:
   <https://huggingface.co/datasets/allenai/analysis_mixtral>
