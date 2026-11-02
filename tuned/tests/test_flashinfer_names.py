"""The flashinfer pinned in requirements/gb10.txt must provide what this vllm imports from it.

vLLM v0.33.0 imports modules and names that the fork's flashinfer 0.7.0 lacked; flashinfer 0.7.2.post1 has them. An
upstream merge can move vllm ahead of the pinned flashinfer again, and most of these imports are lazy or guarded, so a
gap shows up only when the model that needs it runs. This test fails on any of them that does not import.

Run from the repo root, in a venv with the pinned flashinfer installed:
    python -m pytest tuned/tests/test_flashinfer_names.py
"""

from importlib import import_module

import pytest

# Modules and names found missing from flashinfer 0.7.0 when vllm v0.33.0 was merged (2026-11-02). Users: the Muse Spark
# model code (vllm/models/muse_spark/nvidia/) and vllm/distributed/device_communicators/all2all.py.
MODULES = [
    "flashinfer.comm.mnnvl_cutedsl_ag",
    "flashinfer.comm.mnnvl_cutedsl_moe_finalize",
    "flashinfer.comm.tp_moe_finalize_rmsnorm",
    "flashinfer.norm.kernels.gated_residual_next_rmsnorm",
]
NAMES = [
    ("flashinfer.moe_ep", "NVLinkOneSidedAlltoAll"),
    ("flashinfer.norm", "gated_residual_rmsnorm"),
    ("flashinfer.norm", "rmsnorm_sigmoid_gate"),
    ("flashinfer.norm", "supports_gated_residual_rmsnorm"),
]


@pytest.mark.parametrize("module", MODULES)
def test_flashinfer_module_imports(module: str) -> None:
    import_module(module)


@pytest.mark.parametrize(("module", "name"), NAMES)
def test_flashinfer_name_is_present(module: str, name: str) -> None:
    assert hasattr(import_module(module), name), f"{module} has no {name}"
