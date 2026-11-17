# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Read safetensors checkpoints with ``O_DIRECT``, bypassing the page cache.

The default (lazy) strategy memory-maps each checkpoint file, so every byte of
the checkpoint passes through the OS page cache. On integrated GPUs, where the
page cache and device memory share the same physical RAM, that leaves a
checkpoint-sized copy of the weights resident until the kernel reclaims it, and
the first host-to-device copy from an untouched mapping takes its page faults
inside the driver.

This reader parses the safetensors header itself and reads tensor data with
``O_DIRECT`` into ordinary page-aligned host buffers, so no page cache is
filled and every source page is already resident when it is copied to the
device. Files or tensors it cannot handle (no ``O_DIRECT`` support on the
filesystem, an unknown dtype) fall back to ``safetensors.safe_open``.
"""

import errno
import json
import mmap
import os
import struct
from collections.abc import Callable, Generator

import torch
from safetensors.torch import safe_open

from vllm.logger import init_logger

logger = init_logger(__name__)

# Largest host buffer one read fills. Tensors are yielded as views into it, so
# this bounds the reader's own peak memory when the consumer copies each tensor
# and drops it. A single tensor larger than this gets a buffer of its own.
DEFAULT_DIRECT_IO_BUFFER_BYTES = 1 << 30
# Skipped tensors (e.g. other ranks' experts) separated by more than this many
# bytes split a read in two instead of being read and discarded.
DEFAULT_DIRECT_IO_MAX_GAP_BYTES = 16 << 20

_HEADER_LEN_BYTES = 8
# O_DIRECT needs the file offset, length and buffer address aligned to the
# logical block size. The page size is a multiple of it on Linux block devices.
_ALIGN = mmap.PAGESIZE


def _safetensors_dtypes() -> dict[str, torch.dtype]:
    names = {
        "BOOL": "bool",
        "U8": "uint8",
        "I8": "int8",
        "U16": "uint16",
        "I16": "int16",
        "U32": "uint32",
        "I32": "int32",
        "U64": "uint64",
        "I64": "int64",
        "F16": "float16",
        "BF16": "bfloat16",
        "F32": "float32",
        "F64": "float64",
        "C64": "complex64",
        "F8_E4M3": "float8_e4m3fn",
        "F8_E5M2": "float8_e5m2",
        "F8_E8M0": "float8_e8m0fnu",
    }
    return {
        key: getattr(torch, name)
        for key, name in names.items()
        if isinstance(getattr(torch, name, None), torch.dtype)
    }


_DTYPES = _safetensors_dtypes()


class _Entry:
    __slots__ = ("name", "dtype", "shape", "begin", "end")

    def __init__(
        self, name: str, dtype: torch.dtype, shape: list[int], begin: int, end: int
    ) -> None:
        self.name = name
        self.dtype = dtype
        self.shape = shape
        self.begin = begin
        self.end = end


class _UnsupportedFile(Exception):
    """The file cannot be read by this reader; use safe_open instead."""


def _aligned_buffer(nbytes: int) -> torch.Tensor:
    """A uint8 CPU tensor of ``nbytes`` whose data pointer is ``_ALIGN``-aligned.

    Plain (pageable) memory on purpose: pinned memory is allocated through the
    GPU driver, which is the allocation this reader exists to keep small.
    """
    raw = torch.empty(nbytes + _ALIGN, dtype=torch.uint8)
    offset = (-raw.data_ptr()) % _ALIGN
    return raw[offset : offset + nbytes]


class DirectIOSafetensorsReader:
    """Yield ``(name, tensor)`` from safetensors files read with ``O_DIRECT``.

    The counters record which path each file took, so callers and tests can
    tell a direct read from a fallback.
    """

    def __init__(
        self,
        max_buffer_bytes: int = DEFAULT_DIRECT_IO_BUFFER_BYTES,
        max_gap_bytes: int = DEFAULT_DIRECT_IO_MAX_GAP_BYTES,
    ) -> None:
        if max_buffer_bytes < _ALIGN:
            raise ValueError(f"max_buffer_bytes must be >= {_ALIGN}")
        if max_gap_bytes < 0:
            raise ValueError("max_gap_bytes must be >= 0")
        self.max_buffer_bytes = max_buffer_bytes
        self.max_gap_bytes = max_gap_bytes
        self.direct_files = 0
        self.fallback_files = 0
        self.bytes_read = 0
        # Absolute (aligned_start, aligned_end) byte ranges read with O_DIRECT.
        self.read_ranges: list[tuple[int, int]] = []

    def iter_file(
        self,
        path: str,
        skip: Callable[[str], bool] | None = None,
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        """Yield the tensors of one file in file order, skipping ``skip(name)``."""
        try:
            data_start, entries = self._read_header(path)
            fd = self._open_direct(path)
        except _UnsupportedFile as exc:
            yield from self._fallback(path, skip, str(exc))
            return
        wanted = [e for e in entries if skip is None or not skip(e.name)]
        try:
            batches = self._plan(wanted)
            first = True
            for batch in batches:
                try:
                    buffer, base = self._read_batch(fd, data_start, batch)
                except _UnsupportedFile as exc:
                    if not first:
                        raise OSError(
                            errno.EIO, f"O_DIRECT read failed part-way: {exc}", path
                        ) from exc
                    os.close(fd)
                    fd = -1
                    yield from self._fallback(path, skip, str(exc))
                    return
                if first:
                    self.direct_files += 1
                    first = False
                for entry in batch:
                    yield entry.name, self._tensor(buffer, base, data_start, entry)
                del buffer
            if first:
                # Nothing wanted from this file; it still counts as direct.
                self.direct_files += 1
        finally:
            if fd >= 0:
                os.close(fd)

    @staticmethod
    def _read_header(path: str) -> tuple[int, list[_Entry]]:
        with open(path, "rb") as f:
            raw_len = f.read(_HEADER_LEN_BYTES)
            if len(raw_len) != _HEADER_LEN_BYTES:
                raise ValueError(f"{path}: file too short for a safetensors header")
            (header_len,) = struct.unpack("<Q", raw_len)
            header_bytes = f.read(header_len)
            if len(header_bytes) != header_len:
                raise ValueError(f"{path}: truncated safetensors header")
        header = json.loads(header_bytes)
        entries = []
        for name, info in header.items():
            if name == "__metadata__":
                continue
            dtype = _DTYPES.get(info["dtype"])
            if dtype is None:
                raise _UnsupportedFile(f"dtype {info['dtype']} of {name}")
            begin, end = info["data_offsets"]
            entries.append(_Entry(name, dtype, list(info["shape"]), begin, end))
        entries.sort(key=lambda e: e.begin)
        return _HEADER_LEN_BYTES + header_len, entries

    @staticmethod
    def _open_direct(path: str) -> int:
        flag = getattr(os, "O_DIRECT", None)
        if flag is None:
            raise _UnsupportedFile("O_DIRECT is not available on this platform")
        try:
            return os.open(path, os.O_RDONLY | flag)
        except OSError as exc:
            if exc.errno in (errno.EINVAL, errno.EOPNOTSUPP):
                raise _UnsupportedFile(f"open with O_DIRECT: {exc}") from exc
            raise

    def _plan(self, entries: list[_Entry]) -> list[list[_Entry]]:
        """Group tensors into reads of at most ``max_buffer_bytes`` each."""
        batches: list[list[_Entry]] = []
        for entry in entries:
            if batches:
                batch = batches[-1]
                gap = entry.begin - batch[-1].end
                span = entry.end - batch[0].begin
                if gap <= self.max_gap_bytes and span + 2 * _ALIGN <= (
                    self.max_buffer_bytes
                ):
                    batch.append(entry)
                    continue
            batches.append([entry])
        return batches

    def _read_batch(
        self, fd: int, data_start: int, batch: list[_Entry]
    ) -> tuple[torch.Tensor, int]:
        start = data_start + batch[0].begin
        end = data_start + batch[-1].end
        base = start - start % _ALIGN
        length = -(-(end - base) // _ALIGN) * _ALIGN
        buffer = _aligned_buffer(length)
        view = memoryview(buffer.numpy())
        needed = end - base
        done = 0
        while done < needed:
            try:
                n = os.preadv(fd, [view[done:]], base + done)
            except OSError as exc:
                if exc.errno in (errno.EINVAL, errno.EOPNOTSUPP):
                    raise _UnsupportedFile(f"read with O_DIRECT: {exc}") from exc
                raise
            if n == 0:
                raise OSError(
                    errno.EIO,
                    f"unexpected end of file at byte {base + done}, needed {end}",
                )
            done += n
        self.bytes_read += done
        self.read_ranges.append((base, base + length))
        return buffer, base

    @staticmethod
    def _tensor(
        buffer: torch.Tensor, base: int, data_start: int, entry: _Entry
    ) -> torch.Tensor:
        nbytes = entry.end - entry.begin
        if nbytes == 0:
            return torch.empty(entry.shape, dtype=entry.dtype)
        offset = data_start + entry.begin - base
        chunk = buffer[offset : offset + nbytes]
        if chunk.data_ptr() % entry.dtype.itemsize:
            # The format does not require offsets aligned to the dtype; copy
            # such a tensor out so the view below is legal.
            chunk = chunk.clone()
        return chunk.view(entry.dtype).reshape(entry.shape)

    def _fallback(
        self,
        path: str,
        skip: Callable[[str], bool] | None,
        reason: str,
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        logger.warning_once(
            "Direct I/O is not usable for %s (%s); reading it with safe_open "
            "instead. Later files that hit the same problem fall back silently.",
            path,
            reason,
        )
        self.fallback_files += 1
        with safe_open(path, framework="pt") as f:
            for name in f.keys():  # noqa: SIM118
                if skip is not None and skip(name):
                    continue
                yield name, f.get_tensor(name)
