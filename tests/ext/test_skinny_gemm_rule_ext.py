# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The default skinny GEMM rule for unplanned shapes (SkinnyGemmRuleExt)."""

from collections.abc import Iterator

import pytest
import torch
from torch import nn

from vllm.ext.model_executor.kernels.linear.skinny_gemm_rule_ext import (
    SkinnyGemmRuleExt,
)
from vllm.model_executor.kernels.linear.cute_dsl import (
    skinny_gemm_plans as skinny_plans,
)
from vllm.model_executor.layers.utils import maybe_use_skinny_gemm
from vllm.platforms import current_platform

# Qwen3-4B qkv, Qwen2.5-0.5B (K=896), a small K, a MoE router, a Llama down.
RULE_SHAPES = [(6144, 2560), (1152, 896), (2048, 192), (128, 2048), (4096, 14336)]


@pytest.fixture(autouse=True)
def fresh_rule_cache() -> Iterator[None]:
    """The rule caches per shape and reads its env var on a miss."""
    SkinnyGemmRuleExt.plan.cache_clear()
    yield
    SkinnyGemmRuleExt.plan.cache_clear()


def _fake_device(monkeypatch: pytest.MonkeyPatch, capability: tuple[int, int]) -> None:
    monkeypatch.setattr(
        skinny_plans.current_platform, "get_device_capability", lambda: capability
    )
    monkeypatch.setattr(
        skinny_plans.shape_dynamic_skinny_gemm, "is_available", lambda: True
    )


class TestRulePlan:
    """SkinnyGemmRuleExt.plan picks a valid config or none."""

    @pytest.mark.parametrize(
        "k,block_size",
        [
            (2560, 128),
            (4096, 128),
            (896, 64),
            (192, 32),
            (128, 32),
            (64, None),
            (100, None),
        ],
    )
    def test_block_size_by_k(self, k: int, block_size: int | None) -> None:
        plan = SkinnyGemmRuleExt.plan(1000, k)
        if block_size is None:
            assert plan is None
            return
        assert plan is not None
        assert set(plan) == set(SkinnyGemmRuleExt.TOKEN_COUNTS)
        assert {c.block_size for c in plan.values()} == {block_size}

    @pytest.mark.parametrize("n,k", RULE_SHAPES)
    def test_configs_meet_kernel_constraints(self, n: int, k: int) -> None:
        # Same checks as test_skinny_gemm_plans_are_valid for measured plans.
        plan = SkinnyGemmRuleExt.plan(n, k)
        assert plan is not None
        for num_tokens, config in plan.items():
            assert config.num_rows == num_tokens
            assert config.block_size % 32 == 0
            assert n % config.outputs_per_block == 0
            k_tile = config.block_size * config.vector_width
            assert k % k_tile == 0
            assert config.static_k == k
            assert config.static_k >= 2 * k_tile

    def test_env_off_disables(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("VLLM_SKINNY_GEMM_DEFAULT_RULE", "0")
        assert SkinnyGemmRuleExt.plan(6144, 2560) is None


class TestPlanLookup:
    """get_skinny_gemm_plan: measured plans first, then the rule, sm121 only."""

    def test_measured_plan_wins(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _fake_device(monkeypatch, (12, 1))
        measured = skinny_plans.SM121_SKINNY_GEMM_PLANS[(4096, 8192)]
        # The measured plan has no M=1 entry; the rule must not fill it in.
        assert 1 not in measured
        assert skinny_plans.get_skinny_gemm_plan(4096, 8192) is measured

    def test_unplanned_shape_gets_rule(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _fake_device(monkeypatch, (12, 1))
        assert (6144, 2560) not in skinny_plans.SM121_SKINNY_GEMM_PLANS
        assert skinny_plans.get_skinny_gemm_plan(
            6144, 2560
        ) == SkinnyGemmRuleExt.plan(6144, 2560)

    def test_other_device_gets_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _fake_device(monkeypatch, (8, 9))
        assert skinny_plans.get_skinny_gemm_plan(6144, 2560) is None


class TestLayerMarking:
    """maybe_use_skinny_gemm marks by device, not by shape."""

    @pytest.mark.parametrize("capability,expected", [((12, 1), True), ((8, 9), False)])
    def test_marks_shape_without_plan_or_rule(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capability: tuple[int, int],
        expected: bool,
    ) -> None:
        _fake_device(monkeypatch, capability)
        # K=100 has neither a measured plan nor a rule config: the op falls
        # back to F.linear per call, but the layer is still marked so the
        # compiled graph does not depend on the plans.
        layer = nn.Linear(100, 4096, bias=False, dtype=torch.bfloat16, device="cpu")
        assert SkinnyGemmRuleExt.plan(4096, 100) is None
        maybe_use_skinny_gemm(layer)
        assert getattr(layer, "use_skinny_gemm", False) is expected


@pytest.mark.skipif(
    not current_platform.is_cuda()
    or current_platform.get_device_capability() != (12, 1)
    or not skinny_plans.shape_dynamic_skinny_gemm.is_available(),
    reason="needs GB10 (sm121) and the CuTe DSL skinny GEMM",
)
@pytest.mark.parametrize("n,k", RULE_SHAPES)
@pytest.mark.parametrize("num_tokens", SkinnyGemmRuleExt.TOKEN_COUNTS)
def test_rule_matches_f_linear_on_gb10(n: int, k: int, num_tokens: int) -> None:
    torch.manual_seed(n + k + num_tokens)
    x = torch.randn(num_tokens, k, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(n, k, dtype=torch.bfloat16, device="cuda") * 0.02
    output = torch.ops.vllm.cuda_skinny_unquantized_gemm(x, weight)
    reference = torch.nn.functional.linear(x, weight)
    cosine = torch.nn.functional.cosine_similarity(
        output.float().flatten(), reference.float().flatten(), dim=0
    ).item()
    assert cosine > 0.999
