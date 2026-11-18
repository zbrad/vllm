# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Load-time variant and CUDA-tag check for tuned wheels (zbrad/vllm
tuned-builds; no upstream counterpart).

Tuned libraries carry a one-line JSON build-info stamp in the ELF section
`.tuned_build_info` (tuned/common.sh, `gpu_tuned_build_info_json`). The build
checks the stamps of torch and flash-attn (`tuned/build.sh`), but nothing
checked them in a running process, so a library from another variant or
toolkit (through `LD_LIBRARY_PATH`, a RUNPATH or a shared install directory)
failed later with a missing symbol or a wrong kernel.

`TunedBuildCheck.run()` is called from vllm/env_override.py after torch is
imported and before vllm's own extension modules load. It takes vllm's own
stamp as the reference and compares `variant` and `cuda.tag` (each when both
stamps carry it) against every
stamped library mapped into the process, plus the stamped extension modules
of vllm and vllm_flash_attn that load later. Unstamped files are ignored, and
with no reference stamp (a source checkout) nothing is checked.
"""

from __future__ import annotations

import importlib.util
import json
import struct
import sys
import time
from pathlib import Path

from vllm.logger import init_logger

logger = init_logger(__name__)


class TunedBuildCheck:
    """Fails fast when a loaded or loadable tuned library has another
    variant or CUDA tag than this vllm build."""

    SECTION_NAME: bytes = b".tuned_build_info"
    REFERENCE_MODULE: str = "vllm._C_stable_libtorch"
    LATE_PACKAGES: tuple[str, ...] = ("vllm", "vllm_flash_attn")
    PROC_MAPS: Path = Path("/proc/self/maps")

    @staticmethod
    def read_stamp(path: Path) -> dict | None:
        """The parsed `.tuned_build_info` JSON of an ELF file.

        Args:
            path: A shared object or executable.

        Returns:
            The stamp, or None when the file is unreadable, not a 64-bit
            little-endian ELF, has no such section, or the section is not a
            JSON object.

        """
        try:
            with path.open("rb") as f:
                ident = f.read(64)
                if len(ident) < 64 or ident[:4] != b"\x7fELF":
                    return None
                if ident[4] != 2 or ident[5] != 1:
                    return None
                (shoff,) = struct.unpack_from("<Q", ident, 0x28)
                shentsize, shnum, shstrndx = struct.unpack_from("<HHH", ident, 0x3A)
                if shoff == 0 or shnum == 0 or shentsize < 64 or shstrndx >= shnum:
                    return None
                f.seek(shoff)
                table = f.read(shentsize * shnum)
                if len(table) < shentsize * shnum:
                    return None
                headers = [
                    struct.unpack_from("<IIQQQQ", table, i * shentsize)
                    for i in range(shnum)
                ]
                names_off, names_size = headers[shstrndx][4], headers[shstrndx][5]
                f.seek(names_off)
                names = f.read(names_size)
                for name_idx, _, _, _, offset, size in headers:
                    end = names.find(b"\0", name_idx)
                    if names[name_idx:end] != TunedBuildCheck.SECTION_NAME:
                        continue
                    f.seek(offset)
                    text = f.read(size).rstrip(b"\0").decode("utf-8")
                    stamp = json.loads(text)
                    return stamp if isinstance(stamp, dict) else None
        except (OSError, ValueError, struct.error, UnicodeDecodeError):
            return None
        return None

    @staticmethod
    def _axes(stamp: dict) -> tuple[object, object]:
        """The (variant, cuda tag) pair of a stamp."""
        cuda = stamp.get("cuda")
        tag = cuda.get("tag") if isinstance(cuda, dict) else None
        return stamp.get("variant"), tag

    @staticmethod
    def _mapped_files(maps: Path) -> list[Path]:
        """Files mapped into this process, in first-mapped order."""
        try:
            lines = maps.read_text().splitlines()
        except OSError:
            return []
        seen: dict[str, None] = {}
        for line in lines:
            fields = line.split(maxsplit=5)
            if len(fields) == 6 and fields[5].startswith("/"):
                seen.setdefault(fields[5], None)
        return [Path(p) for p in seen if not p.endswith(" (deleted)")]

    @staticmethod
    def _package_extensions(package: str) -> list[Path]:
        """Extension modules directly in a top-level package's directory,
        found without importing the package."""
        try:
            spec = importlib.util.find_spec(package)
        except (ImportError, ValueError):
            return []
        if spec is None or not spec.submodule_search_locations:
            return []
        found: list[Path] = []
        for location in spec.submodule_search_locations:
            found.extend(sorted(Path(location).glob("*.so")))
        return found

    @staticmethod
    def check(reference: Path, candidates: list[Path]) -> None:
        """Compare each stamped candidate with the reference stamp.

        Args:
            reference: The library whose stamp names the expected axes.
            candidates: Libraries to compare; unstamped ones are skipped.

        Raises:
            ImportError: A candidate's variant or CUDA tag differs.

        """
        ref_stamp = TunedBuildCheck.read_stamp(reference)
        if ref_stamp is None:
            return
        expected = TunedBuildCheck._axes(ref_stamp)
        for path in candidates:
            stamp = TunedBuildCheck.read_stamp(path)
            if stamp is None:
                continue
            found = TunedBuildCheck._axes(stamp)
            # A field missing from either stamp (an older stamp) is not compared.
            if any(
                f is not None and e is not None and f != e
                for f, e in zip(found, expected)
            ):
                raise ImportError(
                    f"tuned build mismatch: {path} is variant={found[0]} "
                    f"cuda.tag={found[1]} (package {stamp.get('package')}, "
                    f"release {stamp.get('release')}), but this vllm "
                    f"({reference}) is variant={expected[0]} "
                    f"cuda.tag={expected[1]}. Install the {expected[0]} "
                    f"{expected[1]} build of {stamp.get('package')}, or remove "
                    "the path that put this copy first (LD_LIBRARY_PATH, a "
                    "shared install directory)."
                )

    @staticmethod
    def run() -> None:
        """Run the check for this process (Linux only; a no-op elsewhere or
        in a source checkout).

        Raises:
            ImportError: A stamped library disagrees with vllm's stamp.

        """
        if not sys.platform.startswith("linux"):
            return
        start = time.perf_counter()
        try:
            spec = importlib.util.find_spec(TunedBuildCheck.REFERENCE_MODULE)
        except (ImportError, ValueError):
            return
        if spec is None or spec.origin is None:
            return
        reference = Path(spec.origin)
        candidates = TunedBuildCheck._mapped_files(TunedBuildCheck.PROC_MAPS)
        for package in TunedBuildCheck.LATE_PACKAGES:
            candidates.extend(TunedBuildCheck._package_extensions(package))
        TunedBuildCheck.check(reference, candidates)
        logger.debug(
            "tuned build check: %d files in %.1f ms",
            len(candidates),
            (time.perf_counter() - start) * 1e3,
        )
