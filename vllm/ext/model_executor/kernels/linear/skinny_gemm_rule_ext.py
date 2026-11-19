# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""tuned-builds extension for cute_dsl/skinny_gemm_plans.py and layers/utils.py:
a default skinny GEMM config for weight shapes that have no measured plan.

Upstream runs the skinny GEMM only for the ``(N, K)`` shapes listed in its
per-device plan table, which on GB10 (sm121) holds GLM-5.3-Flash shapes only.
Every other bf16 linear runs cuBLAS, whose M=1 kernel reads weights at well
under the memory bandwidth on GB10. Kept out of the upstream files so merges
stay conflict-free; they only import ``SkinnyGemmRuleExt`` and call
``SkinnyGemmRuleExt.plan`` and ``SkinnyGemmRuleExt.device_supported``.

The choice is made inside the ``cuda_skinny_unquantized_gemm`` custom op, at
run time. ``maybe_use_skinny_gemm`` marks every eligible layer on a device that
has a plan table, so neither the measured plans nor this rule are baked into a
compiled graph: a cached graph cannot silently keep ``F.linear`` for a layer
the rule now covers, and the rule can change without a stale cache hiding it.
"""

import functools

from vllm import envs
from vllm.model_executor.kernels.linear.cute_dsl.skinny_gemm import (
    SkinnyGemmConfig,
)


class SkinnyGemmRuleExt:
    """Default skinny GEMM plan for shapes without a measured one."""

    # Token counts the rule covers. Measured on GB10 for M = 1, 2 and 4 only;
    # other counts use the measured plan, if any, or F.linear.
    TOKEN_COUNTS: tuple[int, ...] = (1, 2, 4)

    # Block sizes to try, largest first. Each needs K divisible by
    # block_size * VECTOR_WIDTH and at least two such tiles (a static-K kernel
    # requirement, cute_dsl/_skinny_gemm.py).
    BLOCK_SIZES: tuple[int, ...] = (128, 64, 32)
    VECTOR_WIDTH: int = 2

    @classmethod
    @functools.cache
    def plan(cls, n: int, k: int) -> dict[int, SkinnyGemmConfig] | None:
        """Return the default token-count plan for an ``(N, K)`` weight.

        The caller has already checked the device and that the CuTe DSL kernel
        is available. One output per block (so any N works), vector width 2,
        no K unrolling and a static K; the block size is the largest of
        ``BLOCK_SIZES`` that tiles K at least twice.

        Args:
            n: Output features (weight rows).
            k: Input features (weight columns).

        Returns:
            Config per token count, or None if the rule is disabled
            (``VLLM_SKINNY_GEMM_DEFAULT_RULE=0``) or no block size tiles K.
        """
        if not envs.VLLM_SKINNY_GEMM_DEFAULT_RULE:
            return None
        block_size = next(
            (
                b
                for b in cls.BLOCK_SIZES
                if k % (b * cls.VECTOR_WIDTH) == 0 and k >= 2 * b * cls.VECTOR_WIDTH
            ),
            None,
        )
        if block_size is None:
            return None
        return {
            m: SkinnyGemmConfig(
                m,
                block_size,
                1,
                k_unroll=1,
                vector_width=cls.VECTOR_WIDTH,
                static_k=k,
            )
            for m in cls.TOKEN_COUNTS
        }

    @classmethod
    def device_supported(cls) -> bool:
        """Whether this device has a skinny GEMM plan table and the kernel.

        Marks a layer for the skinny GEMM op without looking at its shape; the
        op picks a config, or F.linear, per call.
        """
        from vllm.model_executor.kernels.linear.cute_dsl.skinny_gemm_plans import (
            SKINNY_GEMM_PLANS_BY_CAPABILITY,
            current_platform,
            shape_dynamic_skinny_gemm,
        )

        return (
            current_platform.get_device_capability()
            in SKINNY_GEMM_PLANS_BY_CAPABILITY
            and shape_dynamic_skinny_gemm.is_available()
        )
