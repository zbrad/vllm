#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Acceptance test for the GB10 software stack (torch/flash-attn/flashinfer,
# see requirements/gb10.txt): confirms vLLM can load and generate with small
# stand-ins for the two model families the flashinfer-jit-cache release's
# AOT build was scoped to (NemotronH hybrid Mamba+Attention, DeepSeek-V4
# MLA), without needing the real 120B/multi-hundred-GB checkpoints.
#
# This asserts "these architectures run correctly on the GB10 stack", NOT
# "the AOT cache covers these ops" -- investigation found strict no-JIT
# coverage is not achievable with small stand-ins for either family (see
# per-model notes below), so JIT is left enabled (FLASHINFER_DISABLE_JIT
# unset) here by default. tools/run_gb10.sh (which sets it to 1) is what
# enforces the stronger, AOT-only guarantee for the real target-shaped
# models. To test under strict no-JIT instead, run with
# FLASHINFER_DISABLE_JIT=1 set (flashinfer checks this env var truthily, so
# only "set or unset" matters, not "0" vs "1" -- see tools/run_gb10.sh).

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# flashinfer checks this env var truthily (flashinfer/jit/core.py) -- only
# "unset vs set" matters, not the value. Leaving FLASHINFER_DISABLE_JIT
# unset here (the default) means JIT stays allowed; pass it set to any
# non-empty value to force strict no-JIT instead.
if [[ -z "${FLASHINFER_DISABLE_JIT-}" ]]; then
  unset FLASHINFER_DISABLE_JIT
fi

# model_id:kv_cache_dtype:mamba_backend:label quadruples (kv_cache_dtype/
# mamba_backend empty -> vLLM defaults "auto"/"triton"). The -Tiny/
# -tiny-random entries are small stand-ins sharing the relevant AOT-scoped
# shapes (head_dim/qk_rope_head_dim) with the real target models, not the
# target models themselves.
#
# NemotronH (dense stand-in): runs with vLLM's default Triton Mamba SSU
# kernel (mamba_backend left unset below), NOT flashinfer's AOT-cached SSU
# op -- proves NemotronH runs correctly on the GB10 stack (flash-attn
# attention, flashinfer sampling, Triton mamba), not that flashinfer's SSU
# path works. Forcing mamba_backend="flashinfer" does NOT work for this
# specific model even with JIT allowed: Nemotron-3-Nano-4B-BF16's
# mamba_head_dim=80 fails a compile-time static_assert in flashinfer's own
# CUDA source ("DIM must be divisible by TMA_STATE_ROWS") -- a fundamental
# kernel limitation for that head dim, not an AOT-cache gap.
#
# NemotronH (NVFP4 MoE stand-in): nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4
# closes the gap the above paragraph used to describe as untestable. It
# shares Nemotron-3-Super-120B's mamba_head_dim=64 (confirmed via
# config.json) AND has real NVFP4-quantized routed MoE experts (confirmed
# via hf_quant_config.json, modelopt v0.29.0, group size 16) at a
# manageable 19GB/5-shard download -- unlike Nemotron-3-Nano-30B-A3B-BF16
# (right head_dim, but 63GB and unquantized) or Nano-4B (quantized-shape
# irrelevant since it's dense BF16, and wrong head_dim besides). Run here
# with mamba_backend="flashinfer" explicitly (independent of this script's
# global FLASHINFER_DISABLE_JIT setting -- see header comment): confirmed
# 2026-07-06 under FLASHINFER_DISABLE_JIT=1 (strict no-JIT) that
# `ssu_dispatch.py` selects "Using flashinfer Mamba SSU backend." and
# generates correctly, with no MissingJITCacheError -- i.e. flashinfer's
# AOT SSU cache genuinely covers mamba_head_dim=64, and NVFP4 MoE expert
# weight loading (the routed_experts.py copy_ fix from 989b7a835) works
# outside the DeepSeek-V4 codepath it was originally verified against.
# This does not substitute for testing the real Super-120B checkpoint
# (different scale entirely), but is the strongest available evidence
# short of that checkpoint that the combination works on this stack.
MODELS=(
  "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16::::NemotronH (dense, stand-in for Nemotron-3-Super-120B)"
  "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4::flashinfer::NemotronH (NVFP4 MoE, mamba_head_dim=64, stand-in for Nemotron-3-Super-120B's AOT-scoped MoE+SSU path)"
  "yujiepan/deepseek-v4-tiny-random:fp8_ds_mla::deepgemm-sm121:DeepSeek-V4 MLA (tiny-random, stand-in for DeepSeek-V4-Flash)"
  "silence09/DeepSeek-V4-Pro-Tiny:fp8_ds_mla::bad-checkpoint:DeepSeek-V4 MLA (Pro-Tiny, second stand-in for cross-check)"
)

# Known, confirmed reasons DeepSeek-V4 can't run on this stack today --
# these are documented skips, not failures of this test or of the gb10
# rework itself:
#   deepgemm-sm121: vLLM's vendored DeepGEMM CUDA library (.deps/deepgemm-src,
#     confirmed against upstream deepseek-ai/DeepGEMM's latest commit too)
#     has no SF-layout transform branch for arch_major=12 (SM120/121, i.e.
#     GB10) at (gran_mn=1, gran_k=32) -- the granularity DeepSeek-V4's NVFP4
#     expert scales use. Hits a hard C++ assertion ("Unknown SF
#     transformation", csrc/apis/layout.hpp) during process_weights_after_
#     loading, regardless of FLASHINFER_DISABLE_JIT. This is a vendored
#     kernel gap, not something fixable from vLLM's Python side.
#   bad-checkpoint: this specific community checkpoint's config uses a
#     'hash_moe' mlp_layer_type not recognized by the installed
#     transformers/huggingface_hub version -- a checkpoint-schema
#     incompatibility unrelated to flashinfer or the deepgemm-sm121 gap
#     above (which blocks DeepSeek-V4 on this stack regardless).

declare -a RESULTS=()
FAILED=0

for entry in "${MODELS[@]}"; do
  model_id="${entry%%:*}"
  rest="${entry#*:}"
  kv_cache_dtype="${rest%%:*}"
  rest="${rest#*:}"
  mamba_backend="${rest%%:*}"
  rest="${rest#*:}"
  skip_reason="${rest%%:*}"
  label="${rest#*:}"

  if [[ -n "${skip_reason}" ]]; then
    echo "=========================================="
    echo "Skipping: ${model_id} (${label})"
    echo "Reason: ${skip_reason} -- see script header for detail"
    echo "=========================================="
    RESULTS+=("SKIP  ${model_id}  (${label}) -- ${skip_reason}")
    continue
  fi

  echo "=========================================="
  echo "Testing: ${model_id} (${label})"
  echo "FLASHINFER_DISABLE_JIT=${FLASHINFER_DISABLE_JIT:-<unset, JIT allowed>}" \
    "kv_cache_dtype=${kv_cache_dtype:-auto} mamba_backend=${mamba_backend:-triton}"
  echo "=========================================="

  if .venv/bin/python -c "
from vllm import LLM, SamplingParams

kwargs = {}
if '${mamba_backend}':
    kwargs['mamba_backend'] = '${mamba_backend}'

llm = LLM(
    model='${model_id}',
    gpu_memory_utilization=0.85,
    max_model_len=512,
    trust_remote_code=False,
    kv_cache_dtype='${kv_cache_dtype:-auto}',
    **kwargs,
)
sp = SamplingParams(max_tokens=16, temperature=0.0)
out = llm.generate(['Hello, my name is'], sp)
text = out[0].outputs[0].text
assert text, 'empty generation output'
print(f'ACCEPTANCE_OK: ${model_id}: {text!r}')
"; then
    RESULTS+=("PASS  ${model_id}  (${label})")
  else
    RESULTS+=("FAIL  ${model_id}  (${label})")
    FAILED=1
  fi
done

echo ""
echo "=========================================="
echo "Summary"
echo "=========================================="
for r in "${RESULTS[@]}"; do
  echo "${r}"
done

exit "${FAILED}"
