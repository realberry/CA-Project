# Day 1: LLMSimulator and Duplex reproduction

## Source and build

- LLMSimulator commit: `419252761fbdb95b789778a02256d458a5537ec7`
- Ramulator 2.0 submodule: `b5262869b24790e5ec62d9d9974fca9bbb4f54e0`
- VM: Linux Mint/Ubuntu 24.04 base, 8 vCPUs, 15 GiB RAM
- Compiler: GCC 13.3.0
- Build type: Release
- Executable: `~/ca-assignment/Project/LLMSimulator/build/run`

The official Ramulator patch removed the direct `<cstdint>` include from
`src/base/utils.h`. GCC 13 does not provide `uint64_t` transitively here, so the
recorded `patches/ramulator2-gcc13-cstdint.patch` restores the include. No
simulator behavior was changed.

## Untouched repository baseline

The first successful run preserved the repository's default `config.yaml`:
DeepSeek V3, B100, one node with eight devices, GPU-only execution, synthesized
requests, and analytical memory timing. It completed with exit status 0. The
raw log, configuration snapshot, compiler version, commit IDs, and timestamps
are under `results/day1/`.

This is a build and execution smoke test. It is not the Duplex result because
the default configuration specifies `processor_type: GPU`.

## Duplex smoke configuration

`configs/duplex_smoke.yaml` follows the paper's principal Mixtral system:

- Mixtral with FP16 weights
- one node and four H100-class devices
- `processor_type: GPU+LOGIC`
- synthetic routing with zero skew
- decode mode, input length 2048, maximum batch size 32
- analytical memory timing (`use_ramulator: off`)

The run completed with exit status 0. Its operation trace contains both GPU and
Logic-PIM execution: 96 GPU-tagged operations and 320 Logic-tagged operations.
Across the four recorded token-to-token rows, mean simulated latency was
3,199,574 ns and mean expert-FFN time was 2,161,004 ns. These are smoke-run
values and should not be presented as final experimental results.

## Detailed HBM3/Ramulator check

The corrected reduced `GPU+LOGIC` run used `use_ramulator: on`, input length
128, maximum batch size four, and two simulator iterations. It completed with
exit status 0 and exercised both processor paths (96 GPU-tagged operations and
320 Logic-tagged operations). The single recorded token-to-token row reports
3,410,169 ns latency and 2,321,189 ns in expert FFNs. The top-level operation
trace reports 3,565.4 us because it describes the model graph timeline rather
than the scheduler's token-to-token record.

This verifies that the patched Ramulator/HBM3 integration works. Full-size
measurements and the single-GPU off-chip-memory architecture belong to the next
stage.

The first reduced attempt used a batch size of one across four data-parallel
devices. Integer division in `Scheduler::getMaxMetadata` produced `SIGFPE`
because this gives zero sequences per device. The corrected minimum batch size
is four.
