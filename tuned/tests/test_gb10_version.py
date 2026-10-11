"""Tests for tuned/gb10_version.py (the GB10 local-version suffix for vllm).

Run from the repo root:
    PYTHONPATH=tuned python -m pytest tuned/tests/test_gb10_version.py
"""

import subprocess
from pathlib import Path

import pytest

from gb10_version import ENV_LOCAL_VERSION, Gb10Version

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def env() -> dict[str, str]:
    """An isolated environment carrying a valid label for CUDA 13.4."""
    return {ENV_LOCAL_VERSION: "gb10.cu134.20261011"}


def test_local_suffix_returns_a_valid_label(env: dict[str, str]) -> None:
    assert Gb10Version.local_suffix("13.4", env) == "gb10.cu134.20261011"


def test_local_suffix_accepts_a_sequence_suffix() -> None:
    env = {ENV_LOCAL_VERSION: "gb10.cu134.20261011.2"}
    assert Gb10Version.local_suffix("13.4", env) == "gb10.cu134.20261011.2"


def test_local_suffix_reads_the_real_environment_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(ENV_LOCAL_VERSION, "gb10.cu133.20261011")
    assert Gb10Version.local_suffix("13.3") == "gb10.cu133.20261011"


def test_local_suffix_missing_variable_raises() -> None:
    with pytest.raises(ValueError, match="VLLM_GB10_LOCAL_VERSION is not"):
        Gb10Version.local_suffix("13.4", {})


def test_local_suffix_empty_variable_raises() -> None:
    with pytest.raises(ValueError, match="is not"):
        Gb10Version.local_suffix("13.4", {ENV_LOCAL_VERSION: ""})


@pytest.mark.parametrize(
    "bad",
    [
        "gb10.cu134",  # the old unlabelled shape
        "gb10.cu134.tuning.62",  # the old counter shape
        "gb10.20261011",  # no cuda segment
        "GB10.cu134.20261011",  # uppercase variant
        "gb10.cu134.20261011.0",  # .0 is not a canonical sequence
        "gb10.cu134.20261011.01",
        "gb10.cu134.02061011",  # zero-led date
        "gb10.cu134.2026101",  # seven digits
        "0.29.1+gb10.cu134.20261011",  # the whole version, not the suffix
    ],
)
def test_local_suffix_malformed_label_raises(bad: str) -> None:
    with pytest.raises(ValueError, match="is not <variant>"):
        Gb10Version.local_suffix("13.4", {ENV_LOCAL_VERSION: bad})


def test_local_suffix_impossible_calendar_date_raises() -> None:
    with pytest.raises(ValueError, match="not a real calendar date"):
        Gb10Version.local_suffix("13.4", {ENV_LOCAL_VERSION: "gb10.cu134.20260230"})


def test_local_suffix_cuda_mismatch_with_nvcc_raises(env: dict[str, str]) -> None:
    with pytest.raises(ValueError, match="says cu134 but nvcc reports CUDA 13.3"):
        Gb10Version.local_suffix("13.3", env)


def test_local_suffix_two_digit_minor_is_not_truncated() -> None:
    env = {ENV_LOCAL_VERSION: "gb10.cu1310.20261011"}
    assert Gb10Version.local_suffix("13.10", env) == "gb10.cu1310.20261011"


def test_local_suffix_truncated_cuda_tag_is_refused_for_two_digit_minor() -> None:
    """The old setup.py cut the tag to three characters; that must not be accepted."""
    env = {ENV_LOCAL_VERSION: "gb10.cu131.20261011"}
    with pytest.raises(ValueError, match="cu1310"):
        Gb10Version.local_suffix("13.10", env)


def test_local_suffix_agrees_with_the_shell_builder() -> None:
    """tuned-common's gpu_tuned_local_version_dated builds it; this accepts it."""
    script = (
        f"source '{REPO_ROOT}/tuned/common.sh'; "
        "gpu_tuned_local_version_dated gb10 134 20261011 3"
    )
    built = subprocess.run(
        ["bash", "-c", script], capture_output=True, text=True, check=True
    ).stdout.strip()
    assert built == "gb10.cu134.20261011.3"
    assert Gb10Version.local_suffix("13.4", {ENV_LOCAL_VERSION: built}) == built
