# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the O_DIRECT safetensors reader (safetensors_load_strategy="direct")."""

import errno
import json
import os
import struct

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from vllm.config.load import LoadConfig
from vllm.model_executor.model_loader import direct_io
from vllm.model_executor.model_loader.direct_io import (
    DirectIOSafetensorsReader,
)
from vllm.model_executor.model_loader.weight_utils import (
    safetensors_weights_iterator,
)

PAGE = direct_io._ALIGN


def _supports_o_direct(directory) -> bool:
    flag = getattr(os, "O_DIRECT", None)
    if flag is None:
        return False
    probe = os.path.join(directory, "o_direct_probe")
    with open(probe, "wb") as f:
        f.write(b"\0" * PAGE)
    try:
        fd = os.open(probe, os.O_RDONLY | flag)
    except OSError:
        return False
    os.close(fd)
    return True


@pytest.fixture
def direct_dir(tmp_path):
    """A directory whose filesystem accepts O_DIRECT, or skip."""
    if not _supports_o_direct(tmp_path):
        pytest.skip(f"{tmp_path} does not support O_DIRECT")
    return tmp_path


def _reference(path) -> dict[str, torch.Tensor]:
    with safe_open(path, framework="pt") as f:
        return {name: f.get_tensor(name) for name in f.keys()}  # noqa: SIM118


def _assert_same(got: dict[str, torch.Tensor], want: dict[str, torch.Tensor]):
    assert sorted(got) == sorted(want)
    for name, tensor in want.items():
        assert got[name].dtype == tensor.dtype, name
        assert got[name].shape == tensor.shape, name
        assert torch.equal(
            got[name].reshape(-1).view(torch.uint8),
            tensor.reshape(-1).view(torch.uint8),
        ), name


def _all_dtype_tensors() -> dict[str, torch.Tensor]:
    gen = torch.Generator().manual_seed(0)
    tensors: dict[str, torch.Tensor] = {}
    for key, dtype in direct_io._DTYPES.items():
        raw = torch.randint(
            0, 256, (37 * dtype.itemsize,), dtype=torch.uint8, generator=gen
        )
        tensors[f"t_{key}"] = raw.view(dtype).reshape(37)
    return tensors


def _savable(tensors: dict[str, torch.Tensor], path) -> dict[str, torch.Tensor]:
    """Save what this safetensors build can save; return what was saved."""
    probe = str(path) + ".probe"
    saved = {}
    for name, tensor in tensors.items():
        try:
            save_file({name: tensor}, probe)
        except Exception:
            continue
        saved[name] = tensor
    if os.path.exists(probe):
        os.remove(probe)
    save_file(saved, str(path))
    return saved


def _write_raw_safetensors(path, entries, header_pad: int = 0) -> None:
    """Write a safetensors file by hand, with exactly the given byte layout.

    ``entries`` is a list of (name, dtype string, shape, bytes); tensors are
    stored back to back in that order, so offsets need not be dtype-aligned.
    """
    header = {}
    data = b""
    for name, dtype, shape, payload in entries:
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [len(data), len(data) + len(payload)],
        }
        data += payload
    header_bytes = json.dumps(header).encode() + b" " * header_pad
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        f.write(data)


def test_direct_io_all_dtypes_match_safe_open(direct_dir):
    path = direct_dir / "dtypes.safetensors"
    saved = _savable(_all_dtype_tensors(), path)
    assert len(saved) >= 10
    reader = DirectIOSafetensorsReader()
    got = dict(reader.iter_file(str(path)))
    assert reader.direct_files == 1 and reader.fallback_files == 0
    _assert_same(got, _reference(path))


def test_direct_io_misaligned_offsets_zero_size_and_scalar(direct_dir):
    path = direct_dir / "layout.safetensors"
    f32 = torch.arange(5, dtype=torch.float32)
    bf16 = torch.arange(3, dtype=torch.bfloat16)
    entries = [
        ("u8_odd", "U8", [3], bytes([1, 2, 3])),
        # Starts at data offset 3: not 4-byte aligned.
        ("f32_misaligned", "F32", [5], f32.numpy().tobytes()),
        ("empty", "F16", [0, 4], b""),
        ("scalar", "I64", [], struct.pack("<q", -7)),
        ("bf16_odd", "BF16", [3], bf16.view(torch.int16).numpy().tobytes()),
    ]
    # Header padding chosen so the data region starts off a page boundary.
    _write_raw_safetensors(path, entries, header_pad=3)
    assert os.path.getsize(path) % PAGE != 0
    reader = DirectIOSafetensorsReader()
    got = dict(reader.iter_file(str(path)))
    assert reader.direct_files == 1
    _assert_same(got, _reference(path))
    assert got["scalar"].shape == () and got["scalar"].item() == -7
    assert got["empty"].shape == (0, 4)


def test_direct_io_splits_reads_and_large_tensor_gets_own_buffer(direct_dir):
    path = direct_dir / "split.safetensors"
    tensors = {f"w{i}": torch.full((PAGE // 4,), float(i)) for i in range(8)}
    tensors["big"] = torch.ones(4 * PAGE)  # 4 pages of float32 = 16 pages
    save_file(tensors, str(path))
    reader = DirectIOSafetensorsReader(max_buffer_bytes=4 * PAGE)
    got = dict(reader.iter_file(str(path)))
    _assert_same(got, _reference(path))
    assert len(reader.read_ranges) > 2
    big_reads = [e - s for s, e in reader.read_ranges if e - s > 4 * PAGE]
    assert len(big_reads) == 1


def test_direct_io_skipped_ranges_are_not_read(direct_dir):
    path = direct_dir / "skip.safetensors"
    gap = 4 * PAGE
    entries = [
        ("keep_a", "U8", [PAGE], bytes(PAGE)),
        ("skip_me", "U8", [8 * gap], bytes([0xAB]) * (8 * gap)),
        ("keep_b", "U8", [PAGE], bytes([1]) * PAGE),
    ]
    _write_raw_safetensors(path, entries)
    reader = DirectIOSafetensorsReader(max_gap_bytes=gap)
    got = dict(reader.iter_file(str(path), skip=lambda n: n == "skip_me"))
    assert sorted(got) == ["keep_a", "keep_b"]
    with open(path, "rb") as f:
        (header_len,) = struct.unpack("<Q", f.read(8))
    data_start = 8 + header_len
    skip_start = data_start + PAGE
    skip_end = skip_start + 8 * gap
    for start, end in reader.read_ranges:
        # Only the pages shared with neighbouring tensors may overlap.
        overlap = min(end, skip_end) - max(start, skip_start)
        assert overlap <= PAGE, (start, end)
    assert reader.bytes_read < 8 * gap


def test_direct_io_falls_back_when_open_rejects_o_direct(tmp_path, monkeypatch):
    path = tmp_path / "fallback_open.safetensors"
    save_file({"a": torch.arange(10)}, str(path))
    real_open = os.open

    def fake_open(p, flags, *args):
        if flags & getattr(os, "O_DIRECT", 0):
            raise OSError(errno.EINVAL, "Invalid argument")
        return real_open(p, flags, *args)

    monkeypatch.setattr(direct_io.os, "open", fake_open)
    reader = DirectIOSafetensorsReader()
    got = dict(reader.iter_file(str(path)))
    assert reader.fallback_files == 1 and reader.direct_files == 0
    _assert_same(got, _reference(path))


def test_direct_io_falls_back_when_first_read_rejects(direct_dir, monkeypatch):
    path = direct_dir / "fallback_read.safetensors"
    save_file({"a": torch.arange(10), "b": torch.ones(3)}, str(path))

    def fake_preadv(*args):
        raise OSError(errno.EINVAL, "Invalid argument")

    monkeypatch.setattr(direct_io.os, "preadv", fake_preadv)
    reader = DirectIOSafetensorsReader()
    got = dict(reader.iter_file(str(path)))
    assert reader.fallback_files == 1 and reader.direct_files == 0
    _assert_same(got, _reference(path))


def test_direct_io_unknown_dtype_falls_back(tmp_path, monkeypatch):
    path = tmp_path / "dtype.safetensors"
    save_file({"a": torch.arange(4, dtype=torch.int32)}, str(path))
    monkeypatch.setattr(direct_io, "_DTYPES", {})
    reader = DirectIOSafetensorsReader()
    got = dict(reader.iter_file(str(path)))
    assert reader.fallback_files == 1
    _assert_same(got, _reference(path))


def test_direct_io_truncated_file_raises(direct_dir):
    path = direct_dir / "truncated.safetensors"
    save_file({"a": torch.ones(4 * PAGE)}, str(path))
    size = os.path.getsize(path)
    with open(path, "r+b") as f:
        f.truncate(size - 100)
    reader = DirectIOSafetensorsReader()
    with pytest.raises(OSError):
        dict(reader.iter_file(str(path)))


def test_direct_io_rejects_bad_sizes():
    with pytest.raises(ValueError):
        DirectIOSafetensorsReader(max_buffer_bytes=1)
    with pytest.raises(ValueError):
        DirectIOSafetensorsReader(max_gap_bytes=-1)


def test_safetensors_weights_iterator_direct_matches_lazy(direct_dir):
    files = []
    for i in range(3):
        path = direct_dir / f"model-0000{i}-of-00003.safetensors"
        save_file(
            {
                f"layers.{i}.mlp.experts.{e}.weight": torch.full(
                    (17,), float(i * 10 + e)
                )
                for e in range(4)
            }
            | {f"layers.{i}.norm": torch.arange(7, dtype=torch.bfloat16)},
            str(path),
        )
        files.append(str(path))
    lazy = dict(safetensors_weights_iterator(files, False, "lazy"))
    direct = dict(safetensors_weights_iterator(files, False, "direct"))
    _assert_same(direct, lazy)
    # Expert filtering applies to the direct path too.
    local = dict(
        safetensors_weights_iterator(files, False, "direct", local_expert_ids={1})
    )
    assert all(".experts.1." in n or ".experts." not in n for n in local)
    assert any(".experts.1." in n for n in local)


def test_load_config_accepts_direct():
    assert LoadConfig(safetensors_load_strategy="direct").safetensors_load_strategy == (
        "direct"
    )
