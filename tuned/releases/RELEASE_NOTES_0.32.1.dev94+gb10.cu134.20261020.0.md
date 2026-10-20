# vLLM 0.32.1.dev94 — GB10 (DGX Spark, SM 12.1), CUDA 13.4 — v0.32.1.dev94+gb10.cu134.20261020.0

## Install

This release is **cu134 only**. The torch, flash-attention and flashinfer wheels it needs exist for CUDA 13.4 only, so there is no cu133
build. The `cu134` in the release name is the toolkit: `CUDA_TAG` below selects it, and `tuned/env.sh` in the source refuses any other.

```
CUDA_TAG=cu134
TAG='v0.32.1.dev94%2Bgb10.cu134.20261020.0'          # the + in the tag is %2B in a URL
python3.14 -m venv ~/vllm-${CUDA_TAG} && . ~/vllm-${CUDA_TAG}/bin/activate

# vllm. Its metadata pins the GB10 torch, flashinfer-python and flashinfer-jit-cache wheels, so pip fetches those too.
pip install "https://github.com/zbrad/vllm/releases/download/${TAG}/vllm-0.32.1.dev94%2Bgb10.cu134.20261020.0-cp314-cp314-linux_aarch64.whl"

# The flash-attention wheel is not a dependency (its own metadata pins torch==2.4.0), so install it without dependencies.
pip install --no-deps "https://github.com/zbrad/flash-attention-vllm/releases/download/v2.7.2.post1%2Bgb10.cu134.tuning.37/vllm_flash_attn-2.7.2.post1%2Bgb10.cu134.tuning.37-cp314-cp314-linux_aarch64.whl"

# export FLASHINFER_DISABLE_JIT=1      # optional: fail on any flashinfer op outside the precompiled set instead of compiling it (see Known limits).
export VLLM_FLASH_ATTN_NO_PTX=1        # the flash-attention build is native sm_121a with no PTX (see tuned/run_gb10.sh)
vllm serve <model>
```

- **Path:** the venv is `~/vllm-cu134/` (any directory works); the CUDA libraries come from `/usr/local/cuda-13.4/lib64`, which must be on the host (the
  shared objects use the toolkit named by `CUDA_TAG`).
- **Needs:** aarch64, a GB10 GPU (SM 12.1), driver R580 or newer, Python 3.14.
- **Not tested as written:** the commands above were not run against the published URLs (the release is not published yet), and a dry-run resolution of this wheel in a fresh
  venv was not repeated for this version. What was checked: this wheel was installed `--no-deps` into the build venv, which holds the torch and flashinfer wheels from `requirements/gb10.txt`,
  and run there (see Tests). The two flashinfer URLs in `requirements/gb10.txt` return HTTP 200.

## Components

| Component | What ships | Device code | Tests | Result |
|---|---|---|---|---|
| vllm 0.32.1.dev94 (upstream `v0.32.0` merged into `tuned-builds`) | `vllm-0.32.1.dev94+gb10.cu134.20261020.0-cp314-cp314-linux_aarch64.whl`, 181 MiB | `_C_stable_libtorch.abi3.so` and `_moe_C_stable_libtorch.abi3.so`: single-arch `sm_121a`, no PTX (checked by `package.sh`); other extensions not re-counted for this release | `tuned/full_test.sh` (see Tests) | passed |
| vllm_flash_attn (separate wheel, `zbrad/flash-attention-vllm` `tuning.37`) | `_vllm_fa2_C.abi3.so` | native `sm_121a`, no PTX (flash-attention repo's own check, not repeated here) | exercised by the NemotronH stand-ins (flash-attn attention) | passed |
| torch (separate wheel, `zbrad/pytorch` `2.15.0+gb10.cu134.20261012`) | pinned dependency | n/a here | exercised by every test | passed |
| flashinfer (`zbrad/flashinfer` `0.7.0+gb10.cu134.20261019.0`, python + jit-cache) | pinned dependency; 174 precompiled objects including the `xqa` and `trtllm_utils` modules | see the flashinfer release notes | both NemotronH stand-ins with `FLASHINFER_DISABLE_JIT=1` and an empty workspace | passed |
| Package as installed | stamped wheel (`sha256 e8eb4a9f…38e1`) | build-info stamp on `_C` and `_moe_C` | smoke: 216 + 126 + 192 kernel tests against the installed package | passed |

## Tests

From the attached test log (`tuned/full_test.sh`, run 2026-10-20 on one GB10 at commit `0917319d83`, `ACCEPT_GPU_MEM_UTIL=0.25`, `FLASHINFER_DISABLE_JIT=1`, empty `FLASHINFER_WORKSPACE_BASE`):

- **Acceptance stand-ins:** `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16` and `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4` each loaded and generated text with flashinfer JIT disabled. No shared object was
  found in the empty flashinfer workspace afterwards, so nothing was compiled at run time. The same 30B NVFP4 check **failed** with the previous flashinfer cache (`tuning.35`, 2026-10-19, `MissingJITCacheError` in `get_xqa_module`).
- **Tooling tests:** 24 passed (`tuned/tests`, including a check that `requirements/gb10.txt` has not drifted from upstream's `cuda.txt`).
- **Import and library checks:** the compiled extensions load, are the wheel's own files, and resolve their CUDA libraries under `/usr/local/cuda-13.4`.
- **Smoke test (attached log):** the wheel was installed from the packaged file; its stamps read back as this release; `tests/kernels/core/test_layernorm.py::test_rms_norm` (216 passed),
  `test_activation.py::test_act_and_mul` (126 passed) and `test_rotary_embedding.py::test_rotary_embedding_opcheck` (192 passed) ran against the installed package.

**What was not tested (recorded decision G3, docs/ARCHITECTURE.md):** upstream's pytest suite (`tests/`) was not run; there are no multi-GPU tests. Four stand-ins were skipped by design:
`yujiepan/deepseek-v4-tiny-random` (DeepGEMM does not run on SM 12.1), `silence09/DeepSeek-V4-Pro-Tiny` (bad checkpoint), `yujiepan/qwen2-vl-tiny-random` (torchvision is not
installed), `yujiepan/deepseek-v3-tiny-random` (MLA dimensions too small). **DeepSeek-V4, Qwen2-VL and the int8 zero-point scaled-mm path are untested on this build.**

## What changed in this release

- **Upstream:** merged vLLM `v0.32.0` (1,327 upstream commits). Three conflicts resolved: the GB10 prebuilt flash-attention block in `cmake/external_projects/vllm_flash_attn.cmake` was kept with upstream's
  new `GIT_TAG`; `deepgemm.cmake` keeps the cached-checkout tag check with upstream's new existence test; `kernel_warmup.py` keeps both imports.
- **Everything is pre-compiled for sm_121a.** Two fork edits make that so. Upstream's CUDA 13.4 supported-architecture list (`CMakeLists.txt`) omitted 12.1, so a GB10 target was lowered to 12.0; it is restored.
  And kernels that upstream builds as lower-architecture SASS plus PTX (Marlin variants, Hadacore, the c2x scaled-mm used by the int8 zero-point path) are built natively for the target
  (`VLLM_TUNED_NATIVE_ONLY`, `cmake/utils.cmake`). `tuned/package.sh` fails if any stamped extension holds another architecture or any PTX, and now also fails when `cuobjdump` cannot read the file instead of
  reading that as no PTX. The previous GB10 build (2026-09-18) had sm_120, sm_80, sm_89 and sm_90 images and PTX, so some kernels ran through the driver's runtime compiler.
- **flashinfer 0.7.0+gb10.cu134.20261019.0.** `requirements/gb10.txt` now pins the new flashinfer release instead of `tuning.35`. Its jit-cache holds the `xqa` and `trtllm_utils` modules the NVFP4 NemotronH stand-in needs.
- **Dependencies:** `requirements/gb10.txt` followed upstream's pins (`humming-kernels` 0.1.12 to 0.1.17, `nvidia-cutlass-dsl` 4.6.2 to 4.8.0, `quack-kernels`, `tokenspeed-mla`, `fastsafetensors`). The old pins made the worker fail to start.
- **Build and release tooling:** flat per-toolkit tree and venv (`tuned/build/cu134/`), `full_test`, `package`, `smoke_test`, and release/publish split in the fleet's layout.

## Known limits

- **Built against torch 2.15, not upstream's pinned 2.13.** `pyproject.toml` pins `torch == 2.13.0`; the build skips that check (`tuned/wheel.sh`) because the venv holds the tuned torch. The two stand-ins above ran; other code paths have not.
- **flashinfer: only the listed shapes are precompiled.** Anything outside them compiles at first use, or fails with `MissingJITCacheError` under `FLASHINFER_DISABLE_JIT=1`. Measured by flashinfer's
  shape survey (`tuned/docs/shape_survey.md` in zbrad/flashinfer, 2026-10-19), with JIT allowed: with vLLM's default flash-attn backend nine small models compiled no flashinfer module, but with
  `attention_backend="FLASHINFER"` seven small models each compiled modules the cache lacks (XQA at head_dim 128 with group ratios other than 8 and 16, head_dim 64 and 256 prefill and XQA). Speculative-decode
  and sliding-window XQA are not precompiled either. `tuned/run_gb10.sh` sets `FLASHINFER_DISABLE_JIT=1` by default; `FLASHINFER_DISABLE_JIT= tuned/run_gb10.sh …` (empty) allows the runtime compile.
  vLLM's own warning "FlashInfer's precompiled kernels are missing or do not match" fires whenever `flashinfer-cubin` is not installed, whatever the jit-cache holds, and the zbrad fork deliberately uses the jit-cache instead, so the warning is expected for this setup.
- **Other runtime compilation this release does not control:** DeepGEMM (compiles its kernels at runtime; `_C_extension.abi3.so` has no device code) and Triton kernels (for example the Mamba SSD kernels warm up at start).
- **`libcudart.so.13` is mapped twice** in a running process (the pip `nvidia-cuda-runtime` wheel and the system toolkit, both 13.4.92 and byte-identical). The smoke check allows this only because the files are identical.
- **Memory:** the GB10 has one memory pool shared by GPU and system. vLLM sizes its KV cache from `--gpu-memory-utilization`; a high value on a busy node can exhaust it (a test run with 0.85 hung the GPU driver and the node's health watchdog rebooted the machine).

## Provenance

- Release label `20261020.0` (UTC date of the tuned commit). Built, acceptance-tested and packaged at `0917319d838d32472eecab2c29b00398904408ca` (the pin bump of `requirements/gb10.txt` to the new flashinfer release).
- Wheel sha256 `e8eb4a9f23e3489e5a15a3809db33d928b94fcd29e54608e5a9ac84df23938e1`.
- Upstream base: `v0.32.0` (`5df63a7203`), merge-base with upstream/main `fe341a53dda00472c281a2d1f4bd34a74e7eae6e`.
- Prebuilt dependencies: `requirements/gb10.txt` (torch, flashinfer) and `vllm_flash_attn` `2.7.2.post1+gb10.cu134.tuning.37` from `zbrad/flash-attention-vllm`.
