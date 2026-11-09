# vLLM 0.33.1.dev166 — rtx40 cu134 (0.33.1.dev166+rtx40.cu134.20261109.1)

Single-arch vllm wheel for RTX 4080/4090 (Ada Lovelace, SM 89) (TORCH_CUDA_ARCH_LIST=8.9), CUDA 13.4.

- Release label: 20261109.1 (UTC date of the tuned commit)
- Built from commit 9d0fb7493a1f7e63b938903608be8a48249b8d7d; tested at 9d0fb7493a1f7e63b938903608be8a48249b8d7d
- Upstream base: upstream/main merge-base 4135590ffe0119dab806d78a7eee6dff3be2618e (the same base as 0.33.1.dev149+gb10.cu134.20261102.1)
- Prebuilt dependencies: requirements/rtx40.txt (torch, flashinfer) and https://github.com/zbrad/flash-attention-vllm/releases/download/v2.7.2.post1%2Brtx40.cu134.20261109.0/vllm_flash_attn-2.7.2.post1%2Brtx40.cu134.20261109.0-cp314-cp314-linux_x86_64.whl
- Wheel sha256: 1344411ec97e86fcc8f263ba1701746c11e667f931ff9ab8d81c80e571ca9bed

The second rtx40 release of the rtx40 cu134 chain: torch `2.15.0+rtx40.cu134.20261109.0`, vllm_flash_attn
`2.7.2.post1+rtx40.cu134.20261109.0` and flashinfer-python `0.7.3+rtx40.cu134.20261109.0` (all zbrad releases, each with
its own notes). Built and tested on one RTX 4090 (24 GiB) under WSL2 (Ubuntu 24.04), Windows driver 621.14. Not run on
native Linux or on an RTX 4080.

## Changes since 0.33.1.dev162+rtx40.cu134.20261109.0

Same source, dependencies and kernels; three changes, none in the compiled extensions.

- **The wheel sets `NCCL_MAX_NCHANNELS=2` for single-rank jobs.** A new baked flag,
  `TunedFlags.NCCL_SINGLE_RANK_FEW_CHANNELS` (set by `tuned/package.sh` for rtx40 only), makes the GPU worker set it
  before the first NCCL communicator when `world_size_across_dp == 1` and the variable is unset, and log
  "Single-rank job: set NCCL_MAX_NCHANNELS=2 (tuned-build default; set NCCL_MAX_NCHANNELS to override)". Checked on
  this packaged wheel with the 4B stand-in: with the variable unset the engine saw 22.18 GiB free at startup (19.27 GiB
  without the setting, measured on the earlier rtx40 builds); with `NCCL_MAX_NCHANNELS=8` the user's value was kept (no log line, 21.9 GiB
  free). Multi-rank jobs, including data parallel, keep NCCL's default (that is the condition in the code; no multi-rank
  job was run).
- **`requirements/gb10.txt` pins the pip CUDA components to 13.4.92** as `rtx40.txt` does, and
  `tuned/tests/test_gb10_requirements.py` now checks `rtx40.txt` against upstream's `cuda.txt` too (39 tests pass).
  Neither changes this wheel.
- **`tuned/docs/nccl_single_rank_memory.md`** records the NCCL measurement with the script that reproduces it.

## What is different from the gb10 build

- **Dependencies.** `requirements/rtx40.txt` (new) pins the rtx40 torch and flashinfer-python; `setup.py` now reads
  `requirements/<variant>.txt` on the prebuilt-dependency path (gb10 still reads `gb10.txt`). Like gb10, no
  torchvision, torchaudio or torchcodec (the PyPI builds are made against a different torch). **flashinfer is 0.7.3**,
  where `requirements/cuda.txt` and `gb10.txt` pin 0.7.2.post1; this release's acceptance run is the first time this
  vLLM has run with it.
- **No flashinfer jit-cache.** rtx40's flashinfer ships only the python wheel, so its kernels JIT-compile on first use
  (needs a CUDA toolkit and `ninja`; see the flashinfer release), and `FLASHINFER_DISABLE_JIT` is not set (the gb10
  wheel sets it).
- **The pip CUDA components are pinned to 13.4.92** (`nvidia-cuda-runtime`, `-nvcc`, `-nvrtc`, `nvidia-nvvm`,
  `nvidia-cuda-crt`). `humming-kernels[cu13]` requires the runtime and nvcc with no version, and a fresh resolve took
  13.5.37; the process then mapped two different `libcudart.so.13` (pip 13.5.37 and the 13.4.92 toolkit's), which the
  smoke test refused. Checked: with the pin, the two copies are byte-identical.
- **Marlin and Hadacore are built for SM 89.** vLLM's CMake listed `8.0+PTX` (and 12.x) for the Marlin, MoE Marlin and
  Hadacore kernels but not 8.9, so an 8.9 target fell back to 8.0, and the first rtx40 build held 34 sm_80 cubins next
  to its sm_89 code. `CMakeLists.txt` now lists 8.9 in those six arch lists (fork commits `4fd451dfe6`, `61012bd689`);
  an 8.0 or 8.6 target still selects 8.0, and 12.x targets are unchanged (checked with `cmake -P`). Every extension in
  this wheel is single-arch sm_89 with no PTX. Checked.

## Limitations

"Checked" means seen on that RTX 4090 with this wheel's dependencies on 2026-11-09.

- **vLLM's NCCL process groups cost about 3 GiB of GPU memory on one GPU, unless NCCL_MAX_NCHANNELS is small.**
  vLLM creates the world group and one group per parallel dimension (TP, PP, DP, PCP, EP, ETP), each an NCCL
  communicator even with one rank. With this torch's NCCL (2.32.3, statically linked) and the default channel count,
  each took 0.38-0.51 GiB of buffers; with `NCCL_MAX_NCHANNELS=2`, the world group plus 6 groups took 0.24 GiB. This
  wheel sets 2 for single-rank jobs (above); a job with more ranks, or one that sets `NCCL_MAX_NCHANNELS` itself, pays
  the full cost. Measurement and script: `tuned/docs/nccl_single_rank_memory.md`. Not compared with a stock NCCL build.
  A single-rank communicator moves no data between GPUs, so 2 channels should cost nothing on one GPU (Claude's
  reasoning, not measured); multi-GPU rtx40 is untested either way.
- **24 GiB is tight for the NVFP4 30B stand-in.** It loads with 18.22 GiB of weights; it ran only with
  `NCCL_MAX_NCHANNELS=2`, `gpu_memory_utilization=0.9` (1.29 GiB of KV cache) and `max_num_seqs=64` (it has 129 Mamba
  cache blocks at that share, one per decoding sequence, so vLLM's default of 256 refuses to start). Checked. Under
  WSL2 the Windows desktop shares the GPU (0.5 GiB in use at idle here), so a higher share can fail at startup.
- **NVFP4 runs weight-only on SM 89.** vLLM selects `MarlinNvFp4LinearKernel` and the `MARLIN` NVFP4 MoE backend and
  warns that the GPU has no native FP4 compute and performance may degrade. Checked that it loads and generates;
  throughput was not measured.
- **Pinned memory is off under WSL2 unless `VLLM_WSL2_ENABLE_PIN_MEMORY=1` is set.** vLLM then logs "Pinned memory is
  not available on this platform" and uses device memory for its UVA buffers (`vllm/platforms/cuda.py`,
  `is_pin_memory_available`). Checked with the 4B stand-in: with `VLLM_WSL2_ENABLE_PIN_MEMORY=1`, pinned memory and UVA
  report available, the warning is gone, and the greedy output is the same as without it. Speed was not measured.
- **The torch has no CPU LAPACK, BLAS, MPI or cuDNN** (see its release notes). Not checked here: whether any vLLM path
  calls CPU linear algebra.
- **Single GPU only.** No multi-GPU test was run.

## Tests

Scope (from the attached test log): gates; import check; tuned/gb10_acceptance_test.sh (stand-ins, all variants); python -m pytest tuned/tests
Skips by design: upstream's pytest suite (tests/) is not run (G3, recorded decision, docs/ARCHITECTURE.md); no multi-GPU tests
Result: === Full test suite summary: acceptance + tuned tests passed ===

- Gates: `_C_stable_libtorch`, `_moe_C_stable_libtorch` and `vllm_flash_attn/_vllm_fa2_C` are single-arch sm_89 with no
  PTX (package.sh's check; also `cuobjdump --list-elf`/`--list-ptx` by hand); FA2 present, FA3 not built.
- Acceptance (`tuned/gb10_acceptance_test.sh`, rtx40 settings from fork commit `99e349e9c3`: the three above):
  - PASS `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16`: FLASH_ATTN backend (FlashAttention 2, the rtx40 vllm_flash_attn),
    triton Mamba SSU, 7.47 GiB of weights, 13.0 GiB of KV cache.
  - PASS `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4`: FLASHINFER attention backend (fp8 KV cache, `arch=sm89`),
    flashinfer Mamba SSU backend, Marlin NVFP4 GEMM and MoE, 18.22 GiB of weights, 1.29 GiB of KV cache.
  - SKIP the same five stand-ins as the gb10 releases, for the same recorded reasons (`deepgemm-sm121`,
    `bad-checkpoint`, `deps-gap-torchvision`, `mla-dims-too-small`, `trtllm-moe-sm100-only`); they were not
    re-examined on SM 89.
- `tuned/tests`: 39 passed, including the check that the pinned flashinfer provides the names vLLM v0.33.0 imports
  (here flashinfer 0.7.3), and the drift checks of `gb10.txt` and `rtx40.txt` against upstream's `cuda.txt`.
- Smoke test of the packaged wheel: passed, including `test_rms_norm` (216), `test_act_and_mul` (126) and
  `test_rotary_embedding_opcheck` (192) from the packaged extensions.

## Install

Install the rtx40 torch, flashinfer-python and vllm_flash_attn wheels first; `requirements/rtx40.txt` and
`tuned/repo.env` have their URLs.

```
pip install "vllm @ https://github.com/zbrad/vllm/releases/download/v0.33.1.dev166%2Brtx40.cu134.20261109.1/vllm-0.33.1.dev166+rtx40.cu134.20261109.1-cp314-cp314-linux_x86_64.whl"
```
