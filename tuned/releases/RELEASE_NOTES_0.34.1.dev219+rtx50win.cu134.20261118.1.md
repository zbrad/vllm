# vLLM 0.34.1.dev219 — rtx50win cu134 (0.34.1.dev219+rtx50win.cu134.20261118.1)

Single-arch vllm wheel for RTX 50 (Blackwell, SM 120) on **native Windows** (no WSL), built with MSVC 14.44 and
CUDA 13.4 (TORCH_CUDA_ARCH_LIST=12.0a). The first Windows release in this series. Python 3.14, win_amd64 only.

- Channel: dev (a build of upstream main, not of an upstream release tag; published as a pre-release)
- Release label: 20261118.1 (UTC build date; `.1` because an earlier `.0` build of the same commit was not
  path-trimmed and is not published)
- Built from commit 18fd5a51bf7a757b26f739072c05b4e3a28f0104 (branch `tuned-win`); tested at the same commit
- Upstream base: upstream/main merge-base 34b780fe66bdf3df173358774eb4f29f684b8368
- Prebuilt dependencies: torch 2.15.0+rtx50.win.cu134.20261114.2 (zbrad/pytorch pre-release, linked below) and the
  zbrad/pyzmq `27.3.0+ipc.20261110.0` wheel (vLLM's requirements point at it by URL). FlashAttention 2 is built
  into this wheel (`vllm/vllm_flash_attn`), from zbrad/flash-attention-vllm `tuned-win`
  cc3cb027e4046e79a7aa12c3a51b3d80728d5bda; it is not a separate download.
- Wheel sha256: 3380c5b087767719d6155f4792f2e7ba1818657290b8607e9d379ce2a111fb9f

## What's in this build

- **Upstream v0.34.0** merged (via `tuned-builds`). Upstream removed QuTLASS and fp_quant in this range (#65230), so
  there is no `_qutlass_C` any more.
- **Windows fixes on top of the merge:** `extensible_tensor.py` took `malloc` from `ctypes.CDLL(None)`, which is
  `dlopen(NULL)` and raises on Windows, so the engine core could not start (18fd5a51bf); and two call sites from the
  merge needed `quant_type_max<T>::val()` to compile with MSVC (31d9f2bdbf).
- **No local build paths in the binaries.** Compiled with `/experimental:deterministic` and `/pathmap` (source
  locations in `__FILE__` strings become `vllm\...`, `flash-attention\...`, `build\...`) and linked with
  `/DEBUG:NONE`, so the extension modules carry no debug directory or `.pdb` path.
- Extension modules: `_C_stable_libtorch`, `_moe_C_stable_libtorch`, `_flashkda_C`, `cumem_allocator`, `fs_io_C`,
  `spinloop`, `vllm_flash_attn._vllm_fa2_C`. Not built: DeepGEMM (it compiles kernels at runtime, which needs nvcc and
  MSVC on the user's machine), FlashMLA and deepselect (SM 90/100 only).

## Tested configuration and limits

- Attention backend `FLASH_ATTN` with `VLLM_USE_FLASHINFER_SAMPLER=0`. The FlashInfer backend and sampler were not
  tested with this build.
- Single GPU. There is no NCCL on Windows (torch's gloo backend is built; multi-GPU was not tested).
- Engine and frontend talk over ZMQ `ipc://` (AF_UNIX); this needs the zbrad/pyzmq `+ipc` wheel, which the install
  below pulls in.
- Tested on Windows 11 (10.0.26100), NVIDIA driver 621.14, an RTX 5090 Laptop GPU.

## Tests

Scope (from the attached smoke log):
- Release-path check: no local work path, user profile path or build host name in any file of the wheel.
- Clean environment: a new venv with only this wheel and the published torch release (`pip check` clean), with no
  CUDA toolkit or MSVC on PATH: offline generate with Qwen/Qwen3-0.6B, eager and `torch.compile` + CUDA graphs
  (identical greedy output), and `vllm serve` answering `/v1/completions` and `/v1/chat/completions`.
- Build environment: import of every extension module, version check (`vllm.__version__` equals the wheel version),
  generate eager and compiled, and the FlashKDA upstream test subset (test_fwd.py, test_vsplit.py: 24 passed).

Skips by design: upstream's pytest suite (tests/) is not run; no multi-GPU tests.

## Install

Python 3.14 on Windows x64. torch must come from its release in the same command; otherwise `torch==2.15.0` resolves
to the PyPI build:

```
pip install "torch @ https://github.com/zbrad/pytorch/releases/download/v2.15.0%2Brtx50.win.cu134.20261114.2/torch-2.15.0%2Brtx50.win.cu134.20261114.2-cp314-cp314-win_amd64.whl" "vllm @ https://github.com/zbrad/vllm/releases/download/v0.34.1.dev219%2Brtx50win.cu134.20261118.1/vllm-0.34.1.dev219+rtx50win.cu134.20261118.1-cp314-cp314-win_amd64.whl"
```

Run with:

```
set VLLM_USE_FLASHINFER_SAMPLER=0
vllm serve Qwen/Qwen3-0.6B --attention-backend FLASH_ATTN
```
