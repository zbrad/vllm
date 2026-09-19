# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""tuned-builds extension for fused_moe/routed_experts.py: weight copies for
native packed sub-byte quantization dtypes. Kept out of routed_experts.py so
upstream merges of that file stay conflict-free; it only imports
`RoutedExpertsExt` and calls `RoutedExpertsExt.copy_quantized_weight`.
"""

import torch


class RoutedExpertsExt:
    """Weight-copy helper for vllm/model_executor/layers/fused_moe/routed_experts.py."""

    # Native packed sub-byte dtypes with no general cross-dtype copy_/cast
    # support in torch (they only support same-dtype identity copy_, or lack
    # even that -- e.g. fill_). Genuinely different floating-point dtypes of
    # the same width (e.g. float8_e4m3fn <- float8_e5m2) DO support a real
    # numeric cast via copy_, so this set must stay narrow: it's an allowlist
    # of dtypes known to need the uint8-reinterpret workaround, not a generic
    # "same element size" heuristic that would also swallow those valid casts.
    PACKED_SUBBYTE_DTYPES: frozenset[torch.dtype] = frozenset(
        {torch.float4_e2m1fn_x2, torch.int4, torch.uint4}
    )

    @classmethod
    def copy_quantized_weight(cls, dst: torch.Tensor, src: torch.Tensor) -> None:
        """Copy ``src`` into ``dst``, handling native packed sub-byte dtypes.

        Works around torch's lack of a cross-dtype ``copy_`` for native packed
        sub-byte quantization dtypes (e.g. ``Float4_e2m1fn_x2``, used by
        DeepSeek-V4's fp4 experts). When exactly one side is such a dtype and
        the other is same-width ``uint8`` (the usual checkpoint/parameter dtype
        mismatch for these formats), reinterpret both as ``uint8`` before
        copying -- the same pattern already used in
        ``fused_moe/oracle/mxfp4.py``. Any other dtype mismatch falls through
        to a plain ``copy_``, which performs a real cast. The chunked copy in
        ``RoutedExperts._load_w2`` goes through this too, so the workaround
        applies to every chunk.

        Args:
            dst: Destination tensor, written in place.
            src: Source tensor.
        """
        if (
            dst.dtype != src.dtype
            and (
                dst.dtype in cls.PACKED_SUBBYTE_DTYPES
                or src.dtype in cls.PACKED_SUBBYTE_DTYPES
            )
            and dst.element_size() == src.element_size()
        ):
            dst.view(torch.uint8).copy_(src.view(torch.uint8))
        else:
            dst.copy_(src)
