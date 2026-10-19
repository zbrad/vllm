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
from collections.abc import Mapping
from typing import Optional

logger = logging.getLogger(__name__)

ENV_BUILD = "VLLM_GB10_BUILD"
ENV_LOCAL_VERSION = "VLLM_GB10_LOCAL_VERSION"

# <variant>.cu<cuda>.<YYYYMMDD>[.<seq>]; a ".0" sequence and a zero-led date are not
# canonical (PEP 440 rewrites or reorders them), so they are not accepted.
LABEL_RE = re.compile(
    r"^(?P<variant>[a-z][a-z0-9]*)\.cu(?P<cuda>[0-9]+)"
    r"\.(?P<date>[1-9][0-9]{7})(?:\.(?P<seq>[1-9][0-9]*))?$"
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
