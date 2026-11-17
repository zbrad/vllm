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
import threading
from collections import deque
from collections.abc import Callable, Generator, Iterable
from concurrent.futures import Future, ThreadPoolExecutor
from queue import Full, Queue
from typing import Any

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
# Device-staged reads (``device`` given): each batch is read in chunks of this
# size through a ring of this many pinned host buffers, so the O_DIRECT read of
# one chunk overlaps the host-to-device copy of the previous ones (the scheme of
# llama.cpp's ``--load-mode dio``: 4 x 64 MiB).
DEFAULT_DIRECT_IO_CHUNK_BYTES = 64 << 20
DEFAULT_DIRECT_IO_RING_SIZE = 4
# O_DIRECT reads kept in flight at once on the device-staged path. 1 reads inline
# on the read-ahead thread; more use worker threads (os.preadv releases the GIL).
# A single 64 MiB O_DIRECT read already keeps an NVMe queue deep (the block layer
# splits it into many commands); on a GB10 node 1 measured fastest (2-4 did not
# help and added thread hand-offs).
DEFAULT_DIRECT_IO_READS_IN_FLIGHT = 1
# Batches a background thread reads and stages ahead of the consumer on the
# device-staged path, across file boundaries. Without read-ahead the disk idles
# while the model's weight loaders process the previous batch (about 40% of the
# load, by profile); 4 or 8 measured no better than 2.
DEFAULT_DIRECT_IO_PREFETCH_BATCHES = 2

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

    With ``device`` (a CUDA device), each batch is staged in a device buffer and
    tensors are yielded as views of it: the file is read in ``chunk_bytes``
    pieces into a ring of ``ring_size`` pinned host buffers, each copied to the
    device with a non-blocking copy on the current stream, and a buffer is
    reused only after its copy's event has completed. ``reads_in_flight`` reads
    run at once on worker threads; chunks are copied in file order as their
    reads finish, so ``ring_size`` must exceed ``reads_in_flight``. A background
    thread stages up to ``prefetch_batches`` batches ahead of the consumer (0
    reads in the caller's thread). Without ``device`` (the default),
    tensors are views of a pageable host buffer, read one batch at a time.
    """

    def __init__(
        self,
        max_buffer_bytes: int = DEFAULT_DIRECT_IO_BUFFER_BYTES,
        max_gap_bytes: int = DEFAULT_DIRECT_IO_MAX_GAP_BYTES,
        device: torch.device | None = None,
        chunk_bytes: int = DEFAULT_DIRECT_IO_CHUNK_BYTES,
        ring_size: int = DEFAULT_DIRECT_IO_RING_SIZE,
        reads_in_flight: int = DEFAULT_DIRECT_IO_READS_IN_FLIGHT,
        prefetch_batches: int = DEFAULT_DIRECT_IO_PREFETCH_BATCHES,
    ) -> None:
        if max_buffer_bytes < _ALIGN:
            raise ValueError(f"max_buffer_bytes must be >= {_ALIGN}")
        if max_gap_bytes < 0:
            raise ValueError("max_gap_bytes must be >= 0")
        if chunk_bytes < _ALIGN or chunk_bytes % _ALIGN:
            raise ValueError(f"chunk_bytes must be a positive multiple of {_ALIGN}")
        if reads_in_flight < 1:
            raise ValueError("reads_in_flight must be >= 1")
        if ring_size <= reads_in_flight:
            raise ValueError("ring_size must be > reads_in_flight")
        if prefetch_batches < 0:
            raise ValueError("prefetch_batches must be >= 0")
        if device is not None and device.type != "cuda":
            raise ValueError(f"device must be a CUDA device, got {device}")
        self.max_buffer_bytes = max_buffer_bytes
        self.max_gap_bytes = max_gap_bytes
        self.device = device
        self.chunk_bytes = chunk_bytes
        self.ring_size = ring_size
        self.reads_in_flight = reads_in_flight
        self.prefetch_batches = prefetch_batches
        self._ring: list[torch.Tensor] = []
        self._events: list[torch.cuda.Event] = []
        self._ring_next = 0
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
        yield from self.iter_files([path], skip)

    def iter_files(
        self,
        paths: Iterable[str],
        skip: Callable[[str], bool] | None = None,
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        """Yield each file's tensors in turn, in file order, skipping ``skip(name)``.

        With device staging and read-ahead, one background thread reads across
        file boundaries, so the disk does not idle while the consumer finishes a
        file. A file whose first read (or open) rejects O_DIRECT is read with
        ``safe_open`` instead; a failure after its first read raises.
        """
        if self.device is None or self.prefetch_batches == 0:
            for path in paths:
                yield from self._iter_file_serial(path, skip)
            return
        yield from self._iter_files_read_ahead(paths, skip)

    def _iter_file_serial(
        self, path: str, skip: Callable[[str], bool] | None
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        """One file, each batch read in the caller's thread."""
        try:
            data_start, entries = self._read_header(path)
            fd = self._open_direct(path)
        except _UnsupportedFile as exc:
            yield from self._fallback(path, skip, str(exc))
            return
        wanted = [e for e in entries if skip is None or not skip(e.name)]
        try:
            first = True
            for batch in self._plan(wanted):
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

    def _iter_files_read_ahead(
        self, paths: Iterable[str], skip: Callable[[str], bool] | None
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        """Read and stage every file's batches on one background thread.

        The thread runs up to ``prefetch_batches`` batches ahead of the consumer,
        across file boundaries, and issues its copies on the consumer's current
        stream, so the consumer's own work on a staged buffer is ordered after
        them. It hands over ``("batch", ...)`` items, a ``("fallback", path,
        reason)`` marker for a file to read with ``safe_open``, an exception to
        re-raise here, or None at the end. Closing this generator stops the
        thread and waits for it.
        """
        assert self.device is not None
        device = self.device
        stream = torch.cuda.current_stream(device)
        out: Queue[tuple[Any, ...] | BaseException | None] = Queue(
            maxsize=self.prefetch_batches
        )
        stop = threading.Event()

        def put(item: tuple[Any, ...] | BaseException | None) -> None:
            while not stop.is_set():
                try:
                    out.put(item, timeout=0.1)
                    return
                except Full:
                    continue

        def produce_file(path: str) -> None:
            try:
                data_start, entries = self._read_header(path)
                fd = self._open_direct(path)
            except _UnsupportedFile as exc:
                put(("fallback", path, str(exc)))
                return
            wanted = [e for e in entries if skip is None or not skip(e.name)]
            try:
                first = True
                for batch in self._plan(wanted):
                    if stop.is_set():
                        return
                    try:
                        buffer, base = self._read_batch(fd, data_start, batch)
                    except _UnsupportedFile as exc:
                        if not first:
                            raise OSError(
                                errno.EIO, f"O_DIRECT read failed part-way: {exc}", path
                            ) from exc
                        put(("fallback", path, str(exc)))
                        return
                    if first:
                        self.direct_files += 1
                        first = False
                    put(("batch", batch, buffer, base, data_start))
                    del buffer
                if first:
                    # Nothing wanted from this file; it still counts as direct.
                    self.direct_files += 1
            finally:
                os.close(fd)

        def produce() -> None:
            try:
                torch.cuda.set_device(device)
                with torch.cuda.stream(stream):
                    for path in paths:
                        if stop.is_set():
                            return
                        produce_file(path)
            except BaseException as exc:  # handed to the consumer, re-raised there
                put(exc)
            finally:
                put(None)

        thread = threading.Thread(
            target=produce, name="direct-io-read-ahead", daemon=True
        )
        thread.start()
        try:
            while True:
                item = out.get()
                if item is None:
                    return
                if isinstance(item, BaseException):
                    raise item
                if item[0] == "fallback":
                    yield from self._fallback(item[1], skip, item[2])
                    continue
                _, batch, buffer, base, data_start = item
                del item
                for entry in batch:
                    yield entry.name, self._tensor(buffer, base, data_start, entry)
                del buffer
        finally:
            stop.set()
            thread.join()

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
        if self.device is not None:
            return self._read_batch_to_device(fd, base, end, length), base
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

    def _ring_buffers(self) -> None:
        """Allocate the pinned ring and its events once, on first use."""
        if self._ring:
            return
        for _ in range(self.ring_size):
            raw = torch.empty(
                self.chunk_bytes + _ALIGN, dtype=torch.uint8, pin_memory=True
            )
            offset = (-raw.data_ptr()) % _ALIGN
            self._ring.append(raw[offset : offset + self.chunk_bytes])
            self._events.append(torch.cuda.Event())

    def _read_batch_to_device(
        self, fd: int, base: int, end: int, length: int
    ) -> torch.Tensor:
        """Read ``[base, end)`` into a new device buffer through the pinned ring.

        Up to ``reads_in_flight`` chunk reads run at once; each finished chunk is
        copied to the device with a non-blocking copy, in file order. A ring slot
        is handed to a new read only after the copy out of it has completed, and
        ``ring_size > reads_in_flight`` keeps a slot from being reused while its
        own read is still pending.
        """
        assert self.device is not None
        self._ring_buffers()
        staged = torch.empty(length, dtype=torch.uint8, device=self.device)
        needed = end - base
        chunks = []
        for offset in range(0, needed, self.chunk_bytes):
            want = min(self.chunk_bytes, length - offset)
            chunks.append((offset, want, min(want, needed - offset)))
        pending: deque[tuple[int, int, Future[int]]] = deque()
        done = 0
        # One read in flight: read inline, no worker thread or futures in between.
        pool = (
            ThreadPoolExecutor(max_workers=self.reads_in_flight)
            if self.reads_in_flight > 1
            else None
        )
        try:
            next_chunk = 0
            while next_chunk < len(chunks) or pending:
                while len(pending) < self.reads_in_flight and next_chunk < len(chunks):
                    offset, want, useful = chunks[next_chunk]
                    slot = self._ring_next
                    self._ring_next = (slot + 1) % self.ring_size
                    # The copy out of this host buffer must finish before a read
                    # overwrites it.
                    self._events[slot].synchronize()
                    if pool is None:
                        future: Future[int] = Future()
                        future.set_result(
                            self._pread_full(
                                fd, self._ring[slot], base + offset, want, useful, end
                            )
                        )
                    else:
                        future = pool.submit(
                            self._pread_full,
                            fd,
                            self._ring[slot],
                            base + offset,
                            want,
                            useful,
                            end,
                        )
                    pending.append((offset, slot, future))
                    next_chunk += 1
                offset, slot, future = pending.popleft()
                got = future.result()
                staged[offset : offset + got].copy_(
                    self._ring[slot][:got], non_blocking=True
                )
                self._events[slot].record()
                done += got
        finally:
            if pool is not None:
                pool.shutdown(wait=True)
        self.bytes_read += done
        self.read_ranges.append((base, base + length))
        return staged

    @staticmethod
    def _pread_full(
        fd: int, host: torch.Tensor, file_offset: int, want: int, useful: int, end: int
    ) -> int:
        """Read at least ``useful`` bytes (up to ``want``) at ``file_offset``."""
        view = memoryview(host.numpy())[:want]
        got = 0
        while got < useful:
            try:
                n = os.preadv(fd, [view[got:]], file_offset + got)
            except OSError as exc:
                if exc.errno in (errno.EINVAL, errno.EOPNOTSUPP):
                    raise _UnsupportedFile(f"read with O_DIRECT: {exc}") from exc
                raise
            if n == 0:
                raise OSError(
                    errno.EIO,
                    f"unexpected end of file at byte {file_offset + got}, needed {end}",
                )
            got += n
        return got

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
