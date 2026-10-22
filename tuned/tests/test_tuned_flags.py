"""Tests that vllm/ext/tuned_flags.py and tuned/package.sh stay in step.

package.sh bakes the flags into a wheel with sed on the stub's exact lines. If either
side is edited alone the wheel would silently carry the source defaults, so these
tests compare them.

Run from the repo root:
    python -m pytest tuned/tests/test_tuned_flags.py
"""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
STUB = REPO_ROOT / "vllm" / "ext" / "tuned_flags.py"
PACKAGE_SH = REPO_ROOT / "tuned" / "package.sh"
FLAG_NAMES = ("FLASH_ATTN_FA2_NATIVE_ONLY", "FLASHINFER_DISABLE_JIT")


def test_stub_defaults_assert_nothing() -> None:
    text = STUB.read_text()
    for name in FLAG_NAMES:
        assert f"    {name}: bool = False\n" in text


def test_package_sh_edits_each_stub_line_and_reads_it_back() -> None:
    script = PACKAGE_SH.read_text()
    for name in FLAG_NAMES:
        sed = f"s/^    {name}: bool = False$/    {name}: bool = True/"
        assert sed in script, f"package.sh does not set {name}"
        assert f"'^    {name}: bool = True$'" in script, f"{name} is not read back"


def test_package_sh_checks_the_fa2_library_before_asserting_it() -> None:
    script = PACKAGE_SH.read_text()
    check = script.index('CHECK_LIBS+=("${FA2_LIB}")')
    loop = script.index('for so in "${CHECK_LIBS[@]}"')
    assert re.search(r"cuobjdump --list-ptx", script[loop:])
    assert check < loop < script.index("FLASH_ATTN_FA2_NATIVE_ONLY: bool = True")
