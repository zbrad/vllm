# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""RoutedExpertsExt.copy_quantized_weight: packed sub-byte dtype copies."""

import torch

from vllm.ext.model_executor.layers.fused_moe.routed_experts_ext import (
    RoutedExpertsExt,
)


def test_copy_quantized_weight_reinterprets_uint8_into_packed_fp4():
    """torch has no cross-dtype copy_ for float4_e2m1fn_x2, so the bytes must
    be copied through a uint8 view rather than cast."""
    src = torch.arange(16, dtype=torch.uint8)
    dst = torch.zeros(16, dtype=torch.uint8).view(torch.float4_e2m1fn_x2)

    RoutedExpertsExt.copy_quantized_weight(dst, src)

    assert torch.equal(dst.view(torch.uint8), src)


def test_copy_quantized_weight_reinterprets_packed_fp4_into_uint8():
    src = torch.arange(16, dtype=torch.uint8).view(torch.float4_e2m1fn_x2)
    dst = torch.zeros(16, dtype=torch.uint8)

    RoutedExpertsExt.copy_quantized_weight(dst, src)

    assert torch.equal(dst, src.view(torch.uint8))


def test_copy_quantized_weight_casts_ordinary_dtypes():
    """Different floating-point dtypes must still get a real numeric cast."""
    src = torch.tensor([1.5, -2.0, 0.25], dtype=torch.float32)
    dst = torch.zeros(3, dtype=torch.bfloat16)

    RoutedExpertsExt.copy_quantized_weight(dst, src)

    assert torch.equal(dst, src.to(torch.bfloat16))


def test_copy_quantized_weight_same_dtype_is_plain_copy():
    src = torch.arange(8, dtype=torch.float32)
    dst = torch.zeros(8, dtype=torch.float32)

    RoutedExpertsExt.copy_quantized_weight(dst, src)

    assert torch.equal(dst, src)
