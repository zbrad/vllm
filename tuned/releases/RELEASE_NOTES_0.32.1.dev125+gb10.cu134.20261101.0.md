# vLLM 0.32.1.dev125 — gb10 cu134 (0.32.1.dev125+gb10.cu134.20261101.0)

Single-arch vllm wheel for NVIDIA GB10 (DGX Spark, Blackwell, SM 12.1 accelerated) (TORCH_CUDA_ARCH_LIST=12.1a), CUDA 13.4.

- Release label: 20261101.0 (UTC date of the tuned commit)
- Built from commit a8ad8813712e16e5d22a340cb0d7ea62930dd718; tested at a8ad8813712e16e5d22a340cb0d7ea62930dd718
- Upstream base: upstream/main merge-base fe341a53dda00472c281a2d1f4bd34a74e7eae6e
- Prebuilt dependencies: requirements/gb10.txt (torch, flashinfer) and https://github.com/zbrad/flash-attention-vllm/releases/download/v2.7.2.post1%2Bgb10.cu134.20261028.0/vllm_flash_attn-2.7.2.post1%2Bgb10.cu134.20261028.0-cp314-cp314-linux_aarch64.whl
- Wheel sha256: f2380773814174c3116e14af0e9304bf74d3e195d49bc17c29b52732e8744280

## Changes since 0.32.1.dev122+gb10.cu134.20261030.0

- No upstream merge: the upstream base is unchanged (fe341a53dd). The version moves from dev122 to dev125 because it counts
  commits, and the three since 20261030.0's build commit are this fork's own: the notes and the two pin changes below.
  `git diff` of the pin commits changes only `requirements/gb10.txt`.
- Prebuilt dependencies move to torch 2.15.0+gb10.cu134.20261030.1 and flashinfer 0.7.0+gb10.cu134.20261101.0 (python and
  jit-cache), in `requirements/gb10.txt`. The torch change is its source (upstream v2.15.0) and nothing in its build
  configuration; flashinfer was rebuilt against that torch with no source change. vllm_flash_attn stays at 20261028.0.
- Not checked: vllm_flash_attn (20261028.0) was built against an earlier torch and was not rebuilt. The import gate in the
  test log passed; whether the acceptance models exercised its kernels was not checked.
- The torch in this release has no CPU LAPACK or BLAS (see its release notes, Limitations): CPU `torch.linalg` calls fail.
  A grep of this repo's runtime code on 2026-10-24 (Claude's grep, not a run) found no such call; the acceptance test below
  ran the stand-in models listed as PASS.
- Acceptance test (`tuned/gb10_acceptance_test.sh`): PASS for `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16` and
  `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4`; SKIP for four stand-ins with the reasons the script reports
  (`deepgemm-sm121`, `bad-checkpoint`, `deps-gap-torchvision`, `mla-dims-too-small`). The skipped stand-ins are not
  coverage of this release.

## Tests

Scope (from the attached test log): gates; import check; tuned/gb10_acceptance_test.sh (stand-ins, all variants); python -m pytest tuned/tests
Skips by design: upstream's pytest suite (tests/) is not run (G3, recorded decision, docs/ARCHITECTURE.md); no multi-GPU tests
Result: === Full test suite summary: acceptance + tuned tests passed ===

## Install

```
pip install "vllm @ https://github.com/zbrad/vllm/releases/download/v0.32.1.dev125%2Bgb10.cu134.20261101.0/vllm-0.32.1.dev125+gb10.cu134.20261101.0-cp314-cp314-linux_aarch64.whl"
```
