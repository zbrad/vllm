"""requirements/gb10.txt and requirements/rtx40.txt must not drift from upstream's requirements/cuda.txt.

Each is cuda.txt with the variant's substitutions applied (torch, flashinfer, and the packages that have no build for
the tuned torch). An upstream merge moves the cuda.txt pins; a pin left behind in gb10.txt broke the worker at start-up
once (humming-kernels 0.1.12 against a vllm that needs 0.1.17). These tests fail on any pin that differs or is missing
without being named below.

Run from the repo root:
    python -m pytest tuned/tests/test_gb10_requirements.py
"""

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

# Substituted on purpose: each file points these at the zbrad wheels for its variant (see the file's header).
SUBSTITUTED = {"torch", "flashinfer-python"}
# Omitted on purpose: no build against the tuned torch exists for these, so a pin would not match it. Add one here only
# with the reason in the requirements file.
OMITTED = {"torchaudio", "torchvision", "torchcodec", "pynvvideocodec", "instanttensor"}
# The pip CUDA components, pinned to the cu134 toolkit in both files (humming-kernels[cu13] leaves them unversioned).
CUDA_COMPONENTS = {"nvidia-cuda-runtime", "nvidia-cuda-nvcc", "nvidia-cuda-nvrtc", "nvidia-nvvm", "nvidia-cuda-crt"}
# Only in the variant's file on purpose.
EXTRA = {
    "gb10": {"flashinfer-jit-cache", "flashinfer-jit-cache-sm121a"} | CUDA_COMPONENTS,
    "rtx40": set(CUDA_COMPONENTS),
}
VARIANTS = sorted(EXTRA)

PIN = re.compile(r"^([A-Za-z0-9_.\-]+)(\[[^\]]*\])?\s*(.*)$")


class Gb10Requirements:
    """Reader for the pinned requirements files."""

    @staticmethod
    def read_pins(path: Path) -> dict[str, str]:
        """Map normalised package name to its extras and specifier, ignoring comments and -r lines."""
        pins: dict[str, str] = {}
        for raw in path.read_text().splitlines():
            line = raw.split("#")[0].strip()
            if not line or line.startswith(("-r", "--")):
                continue
            match = PIN.match(line)
            if match:
                name = match.group(1).lower().replace("_", "-")
                pins[name] = (match.group(2) or "") + match.group(3)
        return pins


@pytest.mark.parametrize("variant", VARIANTS)
def test_variant_pins_match_cuda_pins_except_the_declared_ones(variant: str) -> None:
    cuda = Gb10Requirements.read_pins(REPO_ROOT / "requirements" / "cuda.txt")
    pins = Gb10Requirements.read_pins(REPO_ROOT / "requirements" / f"{variant}.txt")
    drifted = sorted(
        f"{name}: cuda.txt {cuda[name]!r} vs {variant}.txt {pins[name]!r}"
        for name in cuda
        if name in pins and name not in SUBSTITUTED and cuda[name] != pins[name]
    )
    assert not drifted, f"{variant}.txt pins differ from cuda.txt:\n  " + "\n  ".join(drifted)


@pytest.mark.parametrize("variant", VARIANTS)
def test_every_cuda_requirement_is_in_the_variant_or_declared_omitted(variant: str) -> None:
    cuda = Gb10Requirements.read_pins(REPO_ROOT / "requirements" / "cuda.txt")
    pins = Gb10Requirements.read_pins(REPO_ROOT / "requirements" / f"{variant}.txt")
    missing = sorted(name for name in cuda if name not in pins and name not in OMITTED)
    assert not missing, f"cuda.txt requires {missing}, which {variant}.txt neither pins nor declares omitted"


@pytest.mark.parametrize("variant", VARIANTS)
def test_variant_only_requirements_are_declared(variant: str) -> None:
    cuda = Gb10Requirements.read_pins(REPO_ROOT / "requirements" / "cuda.txt")
    pins = Gb10Requirements.read_pins(REPO_ROOT / "requirements" / f"{variant}.txt")
    extra = sorted(name for name in pins if name not in cuda and name not in EXTRA[variant])
    assert not extra, f"{variant}.txt has {extra}, which cuda.txt does not and this test does not declare"


def test_both_variants_pin_the_same_cuda_component_versions() -> None:
    gb10 = Gb10Requirements.read_pins(REPO_ROOT / "requirements" / "gb10.txt")
    rtx40 = Gb10Requirements.read_pins(REPO_ROOT / "requirements" / "rtx40.txt")
    differ = sorted(name for name in CUDA_COMPONENTS if gb10.get(name) != rtx40.get(name) or not gb10.get(name))
    assert not differ, f"the CUDA component pins differ between gb10.txt and rtx40.txt (or are missing): {differ}"
