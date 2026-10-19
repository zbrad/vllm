"""requirements/gb10.txt must not drift from upstream's requirements/cuda.txt.

gb10.txt is cuda.txt with the GB10 substitutions applied (torch, flashinfer, and the packages that have no GB10
build). An upstream merge moves the cuda.txt pins; a pin left behind in gb10.txt broke the worker at start-up once
(humming-kernels 0.1.12 against a vllm that needs 0.1.17). This test fails on any pin that differs or is missing
without being named below.

Run from the repo root:
    python -m pytest tuned/tests/test_gb10_requirements.py
"""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Substituted on purpose: gb10.txt points these at the zbrad GB10 wheels (see the header of gb10.txt).
SUBSTITUTED = {"torch", "flashinfer-python"}
# Omitted on purpose: no GB10 build exists for these, so a pin would not match the tuned torch. Add one here only with
# the reason in gb10.txt.
OMITTED = {"torchaudio", "torchvision", "torchcodec", "pynvvideocodec", "instanttensor"}
# Only in gb10.txt on purpose.
EXTRA = {"flashinfer-jit-cache"}

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


def test_gb10_pins_match_cuda_pins_except_the_declared_ones() -> None:
    cuda = Gb10Requirements.read_pins(REPO_ROOT / "requirements" / "cuda.txt")
    gb10 = Gb10Requirements.read_pins(REPO_ROOT / "requirements" / "gb10.txt")
    drifted = sorted(
        f"{name}: cuda.txt {cuda[name]!r} vs gb10.txt {gb10[name]!r}"
        for name in cuda
        if name in gb10 and name not in SUBSTITUTED and cuda[name] != gb10[name]
    )
    assert not drifted, "gb10.txt pins differ from cuda.txt:\n  " + "\n  ".join(drifted)


def test_every_cuda_requirement_is_in_gb10_or_declared_omitted() -> None:
    cuda = Gb10Requirements.read_pins(REPO_ROOT / "requirements" / "cuda.txt")
    gb10 = Gb10Requirements.read_pins(REPO_ROOT / "requirements" / "gb10.txt")
    missing = sorted(name for name in cuda if name not in gb10 and name not in OMITTED)
    assert (
        not missing
    ), f"cuda.txt requires {missing}, which gb10.txt neither pins nor declares omitted"


def test_gb10_only_requirements_are_declared() -> None:
    cuda = Gb10Requirements.read_pins(REPO_ROOT / "requirements" / "cuda.txt")
    gb10 = Gb10Requirements.read_pins(REPO_ROOT / "requirements" / "gb10.txt")
    extra = sorted(name for name in gb10 if name not in cuda and name not in EXTRA)
    assert (
        not extra
    ), f"gb10.txt has {extra}, which cuda.txt does not and this test does not declare"
