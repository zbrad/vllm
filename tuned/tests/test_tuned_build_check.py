"""Tests for vllm/ext/tuned_build_check.py (the load-time variant and CUDA-tag check).

Each test builds small shared objects with gcc and stamps them with objcopy, the
way tuned/common.sh stamps real ones.

Run from the repo root:
    python -m pytest tuned/tests/test_tuned_build_check.py
"""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from vllm.ext.tuned_build_check import TunedBuildCheck

pytestmark = pytest.mark.skipif(
    shutil.which("gcc") is None or shutil.which("objcopy") is None,
    reason="needs gcc and objcopy",
)


class stamped_library:
    """Builds and stamps throwaway shared objects in one directory."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.base = root / "base.so"
        source = root / "empty.c"
        source.write_text("int tuned_build_check_probe(void) { return 0; }\n")
        subprocess.run(
            ["gcc", "-shared", "-fPIC", "-o", str(self.base), str(source)], check=True
        )

    def make(
        self, name: str, variant: str | None, tag: str | None = "cu134"
    ) -> Path:
        """A copy of the base library, stamped unless `variant` is None."""
        path = self.root / name
        shutil.copyfile(self.base, path)
        if variant is not None:
            stamp = self.root / f"{name}.json"
            stamp.write_text(
                json.dumps(
                    {
                        "schema": "tuned-build-info/1",
                        "package": name.split(".")[0],
                        "release": f"1.0+{variant}.{tag}.20261117.0",
                        "variant": variant,
                        "cuda": {"tag": tag} if tag else {},
                    }
                )
            )
            subprocess.run(
                [
                    "objcopy",
                    "--add-section",
                    f".tuned_build_info={stamp}",
                    str(path),
                ],
                check=True,
            )
        return path


@pytest.fixture
def libs(tmp_path: Path) -> stamped_library:
    return stamped_library(tmp_path)


def test_read_stamp_returns_the_json(libs: stamped_library) -> None:
    stamp = TunedBuildCheck.read_stamp(libs.make("vllm.so", "gb10"))
    assert stamp is not None
    assert stamp["variant"] == "gb10"
    assert stamp["cuda"]["tag"] == "cu134"


def test_read_stamp_unstamped_or_not_elf_is_none(
    libs: stamped_library, tmp_path: Path
) -> None:
    text = tmp_path / "notes.txt"
    text.write_text("not an ELF file\n")
    assert TunedBuildCheck.read_stamp(libs.make("plain.so", None)) is None
    assert TunedBuildCheck.read_stamp(text) is None
    assert TunedBuildCheck.read_stamp(tmp_path / "missing.so") is None


def test_check_matching_stamps_passes(libs: stamped_library) -> None:
    ref = libs.make("vllm.so", "gb10")
    TunedBuildCheck.check(ref, [libs.make("torch.so", "gb10"), ref])


def test_check_unstamped_candidate_is_ignored(libs: stamped_library) -> None:
    TunedBuildCheck.check(libs.make("vllm.so", "gb10"), [libs.make("c10.so", None)])


def test_check_unstamped_reference_checks_nothing(libs: stamped_library) -> None:
    TunedBuildCheck.check(libs.make("vllm.so", None), [libs.make("torch.so", "rtx40")])


def test_check_variant_mismatch_raises(libs: stamped_library) -> None:
    other = libs.make("torch.so", "rtx40")
    with pytest.raises(ImportError) as err:
        TunedBuildCheck.check(libs.make("vllm.so", "gb10"), [other])
    message = str(err.value)
    assert str(other) in message
    assert "variant=rtx40" in message and "variant=gb10" in message


def test_check_cuda_tag_mismatch_raises(libs: stamped_library) -> None:
    other = libs.make("torch.so", "gb10", tag="cu130")
    with pytest.raises(ImportError) as err:
        TunedBuildCheck.check(libs.make("vllm.so", "gb10"), [other])
    message = str(err.value)
    assert "cuda.tag=cu130" in message and "cuda.tag=cu134" in message


def test_check_field_missing_from_a_stamp_is_not_compared(
    libs: stamped_library,
) -> None:
    ref = libs.make("vllm.so", "gb10")
    TunedBuildCheck.check(ref, [libs.make("torch.so", "gb10", tag=None)])


def test_mapped_files_keeps_regular_files_once(tmp_path: Path) -> None:
    lib = tmp_path / "libtorch_cuda.so"
    spaced = tmp_path / "my dir" / "lib.so"
    spaced.parent.mkdir()
    lib.write_bytes(b"")
    spaced.write_bytes(b"")
    maps = tmp_path / "maps"
    maps.write_text(
        f"aaaa-bbbb r-xp 00000000 08:01 1 {lib}\n"
        f"bbbb-cccc r--p 00010000 08:01 1 {lib}\n"
        "cccc-dddd rw-p 00000000 00:00 0 \n"
        "dddd-eeee rw-p 00000000 00:00 0 [heap]\n"
        f"eeee-ffff r-xp 00000000 08:01 2 {tmp_path}/old.so (deleted)\n"
        "eeef-ffff rw-s 00000000 00:05 4 /dev/null\n"
        f"ffff-0000 r-xp 00000000 08:01 3 {spaced}\n"
    )
    assert TunedBuildCheck._mapped_files(maps) == [lib, spaced]


def test_late_loaded_package_extension_is_checked(
    libs: stamped_library, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = tmp_path / "site" / "fake_tuned_pkg"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    other = libs.make("ext.so", "rtx40")
    shutil.move(other, package / "_ext_C.abi3.so")
    monkeypatch.syspath_prepend(str(tmp_path / "site"))
    found = TunedBuildCheck._package_extensions("fake_tuned_pkg")
    assert found == [package / "_ext_C.abi3.so"]
    with pytest.raises(ImportError, match="variant=rtx40"):
        TunedBuildCheck.check(libs.make("vllm.so", "gb10"), found)
