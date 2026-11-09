# vLLM 0.33.1.dev162 — rtx40 cu134 (0.33.1.dev162+rtx40.cu134.20261109.0)

Single-arch vllm wheel for RTX 4080/4090 (Ada Lovelace, SM 89) (TORCH_CUDA_ARCH_LIST=8.9), CUDA 13.4.

- Release label: 20261109.0 (UTC date of the tuned commit)
- Built from commit decee49e49f482de025983ce99fe8dd4ecf5bbd5; tested at decee49e49f482de025983ce99fe8dd4ecf5bbd5
- Upstream base: upstream/main merge-base 4135590ffe0119dab806d78a7eee6dff3be2618e (the same base as 0.33.1.dev149+gb10.cu134.20261102.1)
- Prebuilt dependencies: requirements/rtx40.txt (torch, flashinfer) and https://github.com/zbrad/flash-attention-vllm/releases/download/v2.7.2.post1%2Brtx40.cu134.20261109.0/vllm_flash_attn-2.7.2.post1%2Brtx40.cu134.20261109.0-cp314-cp314-linux_x86_64.whl
- Wheel sha256: fec6a63fa5eedfc37e0ce1dc63afce14e1fc7c96da96b6ca7e5898c5c7ed92f5

The first rtx40 release, completing the rtx40 cu134 chain: torch `2.15.0+rtx40.cu134.20261109.0`, vllm_flash_attn
`2.7.2.post1+rtx40.cu134.20261109.0` and flashinfer-python `0.7.3+rtx40.cu134.20261109.0` (all zbrad releases, each with
its own notes). Built and tested on one RTX 4090 (24 GiB) under WSL2 (Ubuntu 24.04), Windows driver 621.14. Not run on
native Linux or on an RTX 4080.

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

- **vLLM's NCCL process groups cost about 3 GiB of GPU memory on one GPU.** vLLM creates the world group and one group
  per parallel dimension (TP, PP, DP, PCP, EP, ETP), each an NCCL communicator even with one rank. With this torch's
  NCCL (2.32.3, statically linked) and the default channel count, each took 0.38-0.51 GiB of buffers: vLLM's engine saw
  19.27 GiB free at startup where a bare torch process saw 22.45 GiB. `NCCL_MAX_NCHANNELS=2` cut the world group plus
  6 groups from 2.78 GiB to 0.24 GiB, and the engine then saw 22.18 GiB free. Checked, with a standalone script and in
  the acceptance run. Not compared with a stock NCCL build. A single-rank communicator moves no data between GPUs, so
  2 channels should cost nothing on one GPU (Claude's reasoning, not measured); do not carry the setting to multi-GPU
  runs untested.
- **24 GiB is tight for the NVFP4 30B stand-in.** It loads with 18.22 GiB of weights; it ran only with
  `NCCL_MAX_NCHANNELS=2`, `gpu_memory_utilization=0.9` (1.29 GiB of KV cache) and `max_num_seqs=64` (it has 129 Mamba
  cache blocks at that share, one per decoding sequence, so vLLM's default of 256 refuses to start). Checked. Under
  WSL2 the Windows desktop shares the GPU (0.5 GiB in use at idle here), so a higher share can fail at startup.
- **NVFP4 runs weight-only on SM 89.** vLLM selects `MarlinNvFp4LinearKernel` and the `MARLIN` NVFP4 MoE backend and
  warns that the GPU has no native FP4 compute and performance may degrade. Checked that it loads and generates;
  throughput was not measured.
- **Pinned memory is off under WSL2.** vLLM logs "Pinned memory is not available on this platform" and uses device
  memory for its UVA buffers: on WSL2 it disables pinned memory unless `VLLM_WSL2_ENABLE_PIN_MEMORY=1` is set
  (`vllm/platforms/cuda.py`, `is_pin_memory_available`). Read from the source; enabling it was not tested.
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
- `tuned/tests`: 35 passed, including the check that the pinned flashinfer provides the names vLLM v0.33.0 imports
  (here flashinfer 0.7.3). Not covered: `test_gb10_requirements.py` keeps `gb10.txt` in step with upstream's
  `cuda.txt`; no test does that for `rtx40.txt` yet.
- Smoke test of the packaged wheel: passed, including `test_rms_norm` (216), `test_act_and_mul` (126) and
  `test_rotary_embedding_opcheck` (192) from the packaged extensions.

## Install

Install the rtx40 torch, flashinfer-python and vllm_flash_attn wheels first; `requirements/rtx40.txt` and
`tuned/repo.env` have their URLs.

```
pip install "vllm @ https://github.com/zbrad/vllm/releases/download/v0.33.1.dev162%2Brtx40.cu134.20261109.0/vllm-0.33.1.dev162+rtx40.cu134.20261109.0-cp314-cp314-linux_x86_64.whl"
```
