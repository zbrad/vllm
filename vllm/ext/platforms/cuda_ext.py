# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""tuned-builds extension for vllm/platforms/cuda.py: the FLASH_ATTN PTX/driver
compatibility check. Kept out of cuda.py so upstream merges of that file stay
conflict-free; cuda.py only imports `CudaExt` and calls
`CudaExt.check_flash_attn_ptx_compat`.
"""

from __future__ import annotations

import importlib.metadata

import regex as re

import vllm.envs as envs
from vllm.ext.tuned_flags import TunedFlags
from vllm.logger import init_logger
from vllm.platforms.interface import DeviceCapability
from vllm.utils.import_utils import import_pynvml

logger = init_logger(__name__)

pynvml = import_pynvml()


class CudaExt:
    """FLASH_ATTN PTX/driver compatibility check for vllm/platforms/cuda.py."""

    # Consumer Blackwell (sm_120/121, e.g. RTX Pro 6000, GB10/DGX Spark).
    # Unlike datacenter Blackwell, `_vllm_fa2_C` ships no native cubin for
    # these and always falls back to PTX JIT.
    BLACKWELL_CONSUMER_CAPABILITIES: tuple[DeviceCapability, ...] = (
        DeviceCapability(12, 0),
        DeviceCapability(12, 1),
    )

    LOCAL_VERSION_CUDA_RE = re.compile(r"\bcu(\d{3})\b")

    @staticmethod
    def _flash_attn_no_ptx_asserted() -> bool:
        """True when this install carries the packaging-time assertion that
        `_vllm_fa2_C` holds native cubins and no PTX.

        The assertion is `TunedFlags.FLASH_ATTN_FA2_NATIVE_ONLY` (vllm/ext/
        tuned_flags.py). It is False in a source checkout. `tuned/package.sh`
        sets it inside the wheel, and only after running cuobjdump on
        `vllm_flash_attn/_vllm_fa2_C.abi3.so` and finding native cubins for the
        wheel's arch and no PTX. It does not depend on how the process was
        started, and the build's CUDA version says nothing about that library,
        which is built separately (zbrad/flash-attention-vllm).

        Returns:
            True when the wheel asserts native-only FA2; the guard then has
            nothing to check.

        """
        return TunedFlags.FLASH_ATTN_FA2_NATIVE_ONLY

    @staticmethod
    def _driver_max_cuda_version() -> tuple[int, int] | None:
        """Highest CUDA version the installed driver can PTX-JIT for.

        Deliberately doesn't use `@with_nvml_context`: that decorator lets
        `nvmlInit()` failures propagate, which would crash attention-backend
        selection instead of letting this optional check degrade gracefully.

        Returns:
            (major, minor), or None if it can't be determined (e.g. NVML
            unavailable).

        """
        try:
            pynvml.nvmlInit()
        except pynvml.NVMLError:
            return None
        try:
            raw = pynvml.nvmlSystemGetCudaDriverVersion_v2()
            return (raw // 1000, (raw % 1000) // 10)
        except pynvml.NVMLError:
            return None
        finally:
            pynvml.nvmlShutdown()

    @classmethod
    def _build_cuda_version(cls) -> tuple[int, int] | None:
        """CUDA toolkit version vLLM's own C++/CUDA extensions were built with.

        Read off the installed package's local version label (e.g.
        `0.1.dev1+g491f075.cu133` -> (13, 3)), the same `cu\\d{3}` convention
        `setup.py`'s `get_vllm_version()` writes at build time. That suffix is
        only appended when the build's CUDA version differs from vLLM's pinned
        main version (`envs.VLLM_MAIN_CUDA_VERSION`) -- when it matches
        exactly, no suffix is written, so fall back to the main version in
        that case.

        Deliberately not `torch.version.cuda`: that reflects the CUDA version
        the pre-built torch *wheel* was compiled with, which is a separate
        build step from vLLM's own from-source extension compilation and can
        disagree with it (e.g. a from-source vLLM build using a newer local
        `nvcc` than the torch wheel it links against).

        Returns:
            (major, minor), or None if it can't be determined.

        """
        try:
            version_str = importlib.metadata.version("vllm")
        except importlib.metadata.PackageNotFoundError:
            version_str = ""
        match = cls.LOCAL_VERSION_CUDA_RE.search(version_str)
        if match is not None:
            digits = match.group(1)
            return (int(digits[:2]), int(digits[2]))
        try:
            major_str, minor_str = envs.VLLM_MAIN_CUDA_VERSION.split(".")[:2]
            return (int(major_str), int(minor_str))
        except ValueError:
            return None

    @classmethod
    def check_flash_attn_ptx_compat(cls, device_capability: DeviceCapability) -> None:
        """Fail fast when the FLASH_ATTN PTX JIT cannot work on this driver.

        `_vllm_fa2_C` has no native cubin on consumer Blackwell, so it PTX-JITs
        at kernel-launch time. If vLLM was built with a newer CUDA toolkit than
        the installed driver supports, that JIT fails with
        `cudaErrorUnsupportedPtxVersion` deep inside CUDA graph capture, often
        after minutes of weight loading. Fail fast here instead.
        See https://github.com/vllm-project/vllm/issues/47397.

        Args:
            device_capability: Compute capability of the selected device.

        Raises:
            RuntimeError: If the build's CUDA version exceeds the driver's.

        """
        if device_capability not in cls.BLACKWELL_CONSUMER_CAPABILITIES:
            return
        if cls._flash_attn_no_ptx_asserted():
            logger.info_once(
                "Skipping the FLASH_ATTN PTX/driver compatibility check: this "
                "GB10 tuned build asserts (TunedFlags.FLASH_ATTN_FA2_NATIVE_ONLY, "
                "written by tuned/package.sh) that its _vllm_fa2_C was compiled "
                "with a real native sm_121a cubin (CUDA_ARCHS=12.1a in "
                "zbrad/flash-attention-vllm's tuned/build.sh) rather than "
                "upstream's PTX-only consumer-Blackwell build -- verified via "
                "cuobjdump (native cubins, no embedded PTX) when the wheel was "
                "packaged, so it can't hit cudaErrorUnsupportedPtxVersion "
                "regardless of driver/toolkit skew."
            )
            return
        build_cuda = cls._build_cuda_version()
        if build_cuda is None:
            return
        driver_cuda = cls._driver_max_cuda_version()
        if driver_cuda is None:
            return
        if build_cuda > driver_cuda:
            raise RuntimeError(
                f"vLLM was built with CUDA {build_cuda[0]}.{build_cuda[1]}, but "
                f"the installed driver only supports CUDA {driver_cuda[0]}."
                f"{driver_cuda[1]} for runtime PTX compilation. The FLASH_ATTN "
                f"backend (_vllm_fa2_C) has no native cubin for this GPU "
                f"(sm_{device_capability.major}{device_capability.minor}) and "
                "requires PTX JIT, which will fail with "
                "cudaErrorUnsupportedPtxVersion. Fix: update your driver, "
                "rebuild vLLM against a CUDA toolkit <= the driver's supported "
                "version, or pass --attention-backend flashinfer (or "
                "TRITON_ATTN) to avoid this kernel."
            )
