"""GB10 local-version suffix for vllm's version string.

Fork-local (zbrad/vllm, tuned-builds). Called from one short call site in
``setup.py``'s ``get_vllm_version()`` and kept in its own file so upstream merges
of ``setup.py`` do not collide with it.

The suffix itself is built by tuned-common's ``gpu_tuned_local_version_dated`` in
``tuned/wheel.sh`` (the only place a label is allocated), and handed over in the
``VLLM_GB10_LOCAL_VERSION`` environment variable. This module only receives it and
checks it, so the label has exactly one author (see ``tuned/common.sh`` and
``docs/VERSIONING.md`` in zbrad/tuned-common). It replaces the earlier
``tuning.<N>`` commit counter that ``setup.py`` computed itself.
"""

import datetime
import logging
import os
import re
import runpy
from collections.abc import Mapping
from typing import Optional, Union

from packaging.version import Version

logger = logging.getLogger(__name__)

ENV_BUILD = "VLLM_GB10_BUILD"
ENV_LOCAL_VERSION = "VLLM_GB10_LOCAL_VERSION"

# <variant>.cu<cuda>.<YYYYMMDD>[.<seq>], the sequence always written (".0" for the first build of a
# date, docs/ARCHITECTURE.md E1); a zero-led date or sequence is not canonical (PEP 440 rewrites it).
LABEL_RE = re.compile(
    r"^(?P<variant>[a-z][a-z0-9]*)\.cu(?P<cuda>[0-9]+)"
    r"\.(?P<date>[1-9][0-9]{7})(?:\.(?P<seq>0|[1-9][0-9]*))?$"
)


class Gb10Version:
    """Validate and return the tuned local-version suffix for a GB10 vllm build.

    Fork-local; there is no upstream counterpart.
    """

    @staticmethod
    def local_suffix(
        cuda_release: str, environ: Optional[Mapping[str, str]] = None
    ) -> str:
        """Return the suffix to append to the version, e.g. ``gb10.cu134.20261011``.

        Args:
            cuda_release: the CUDA release nvcc reports, as ``"13.4"``.
            environ: the environment to read (default ``os.environ``); a parameter so
                tests need not touch the real one.

        Returns:
            The value of ``VLLM_GB10_LOCAL_VERSION``, once checked.

        Raises:
            ValueError: if the variable is missing or malformed, the date is not a
                real calendar day, or its ``cu<NNN>`` is not the nvcc release the
                build is compiling with. A GB10 build never falls back to an
                unlabelled version: the label is the build's identity.
        """
        env = os.environ if environ is None else environ
        value = env.get(ENV_LOCAL_VERSION, "")
        if not value:
            raise ValueError(
                f"{ENV_BUILD} is set but {ENV_LOCAL_VERSION} is not. Build with "
                "tuned/wheel.sh or tuned/build.sh, which compute the label with "
                "tuned-common's gpu_tuned_local_version_dated."
            )
        match = LABEL_RE.match(value)
        if match is None:
            raise ValueError(
                f"{ENV_LOCAL_VERSION}={value!r} is not <variant>.cu<cuda>."
                "<YYYYMMDD>[.<seq>] (for example gb10.cu134.20261011)."
            )
        if not Gb10Version._is_calendar_date(match.group("date")):
            raise ValueError(
                f"{ENV_LOCAL_VERSION}={value!r} carries {match.group('date')}, "
                "which is not a real calendar date."
            )
        expected = Gb10Version._cuda_compact(cuda_release)
        if match.group("cuda") != expected:
            raise ValueError(
                f"{ENV_LOCAL_VERSION}={value!r} says cu{match.group('cuda')} but "
                f"nvcc reports CUDA {cuda_release} (cu{expected}); the label would "
                "not match the toolkit this wheel is compiled with."
            )
        logger.debug("GB10 local version suffix: %s", value)
        return value

    @staticmethod
    def version_tuple(version: str) -> tuple[Union[int, str], ...]:
        """Return ``version`` as setuptools_scm's ``__version_tuple__``.

        The release numbers, then the pre, post and dev parts and the local label as
        strings: ``"0.33.1.dev180+gb10.cu134.20261113.0"`` becomes
        ``(0, 33, 1, "dev180", "gb10.cu134.20261113.0")``.

        Raises:
            packaging.version.InvalidVersion: if ``version`` is not PEP 440.
        """
        parsed = Version(version)
        fields: list[Union[int, str]] = list(parsed.release)
        if parsed.pre is not None:
            fields.append(f"{parsed.pre[0]}{parsed.pre[1]}")
        if parsed.post is not None:
            fields.append(f"post{parsed.post}")
        if parsed.dev is not None:
            fields.append(f"dev{parsed.dev}")
        if parsed.local is not None:
            fields.append(parsed.local)
        return tuple(fields)

    @staticmethod
    def write_version_file(path: str, version: str) -> None:
        """Rewrite setuptools_scm's ``vllm/_version.py`` to carry ``version``.

        setuptools_scm writes the file with its own local part (``+g<sha>``, and
        ``.d<date>`` from a dirty tree) before the tuned label replaces that part in
        the distribution's version, so without this ``vllm.__version__`` is not the
        version pip reports. The commit setuptools_scm recorded is kept as
        ``__commit_id__``.

        Args:
            path: the ``_version.py`` setuptools_scm has just written.
            version: the full tuned version, as ``0.33.1.dev170+gb10.cu134.20261113.0``.

        Raises:
            OSError: if the file cannot be read or written.
        """
        commit_id = runpy.run_path(path).get("commit_id")
        with open(path, "w", encoding="utf-8") as f:
            f.write(
                "# file generated by setup.py for a tuned build (zbrad/vllm\n"
                "# tuned/gb10_version.py): setuptools_scm's version with the tuned\n"
                "# local label. Don't change, don't track in version control.\n"
                f"__version__ = version = {version!r}\n"
                "__version_tuple__ = version_tuple = "
                f"{Gb10Version.version_tuple(version)!r}\n"
                f"__commit_id__ = commit_id = {commit_id!r}\n"
            )
        logger.debug("wrote %s: %s (commit %s)", path, version, commit_id)

    @staticmethod
    def _cuda_compact(cuda_release: str) -> str:
        """Return the release without its dot: ``"13.4"`` becomes ``"134"``.

        Not truncated: ``"13.10"`` is ``"1310"``, the same as tuned-common's
        ``gpu_tuned_nvcc_compact``.
        """
        return cuda_release.replace(".", "")

    @staticmethod
    def _is_calendar_date(label: str) -> bool:
        """Return whether an 8-digit ``YYYYMMDD`` string is a real calendar day."""
        try:
            datetime.datetime.strptime(label, "%Y%m%d")
        except ValueError:
            return False
        return True
