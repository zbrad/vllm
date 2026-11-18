"""reshape_and_cache_flash with fewer than 32 K elements per token (num_heads * head_size).

The launcher sized the block as min(num_heads * head_size, 512) threads; with a packed K/V
cache (head stride != head size) the kernel steps its head loop by blockDim.x >> 5, which is 0
under 32 threads, so it spun forever. yujiepan/qwen2-vl-tiny-random (1 KV head x 8) hung the
acceptance test there. Needs a GPU and the installed wheel: the check runs in a child process
started from / with -I (the repo root on sys.path would import the source tree), under a timeout
so a regression fails instead of hanging pytest.

Run from the repo root with the build venv's python:
    python -m pytest tuned/tests/test_reshape_and_cache_flash.py
"""

import subprocess
import sys

import pytest

CHILD_TIMEOUT_S = 120

CHILD = r"""
import torch
from vllm import _custom_ops as ops

torch.manual_seed(0)
num_blocks, block_size, num_tokens = 4, 16, 37
for num_heads, head_size in ((1, 8), (2, 8), (1, 64), (4, 64)):
    for packed in (False, True):
        kw = dict(device="cuda", dtype=torch.bfloat16)
        key = torch.randn(num_tokens, num_heads, head_size, **kw)
        value = torch.randn(num_tokens, num_heads, head_size, **kw)
        if packed:
            # K and V share one buffer, as vllm's FlashAttention backend lays them out:
            # head stride 2 * head_size, so the kernel takes its strided-head path.
            kv = torch.zeros(num_blocks, block_size, num_heads, 2, head_size, **kw)
            key_cache, value_cache = kv[..., 0, :], kv[..., 1, :]
        else:
            key_cache = torch.zeros(num_blocks, block_size, num_heads, head_size, **kw)
            value_cache = torch.zeros_like(key_cache)
        slots = torch.randperm(num_blocks * block_size, device="cuda")[:num_tokens]
        scale = torch.ones(1, device="cuda", dtype=torch.float32)
        ops.reshape_and_cache_flash(key, value, key_cache, value_cache, slots, "auto", scale, scale)
        torch.cuda.synchronize()
        flat_k = key_cache.reshape(-1, num_heads, head_size)[slots]
        flat_v = value_cache.reshape(-1, num_heads, head_size)[slots]
        assert torch.equal(flat_k, key) and torch.equal(flat_v, value), (num_heads, head_size, packed)
print("RESHAPE_AND_CACHE_OK")
"""


@pytest.fixture(scope="module")
def cuda_available() -> None:
    probe = subprocess.run(
        [sys.executable, "-I", "-c", "import torch; print(torch.cuda.is_available())"],
        cwd="/",
        capture_output=True,
        text=True,
        timeout=CHILD_TIMEOUT_S,
    )
    if probe.stdout.strip() != "True":
        pytest.skip("needs a CUDA device and torch")


def test_reshape_and_cache_flash_small_heads_completes(cuda_available: None) -> None:
    try:
        run = subprocess.run(
            [sys.executable, "-I", "-c", CHILD],
            cwd="/",
            capture_output=True,
            text=True,
            timeout=CHILD_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        pytest.fail(f"reshape_and_cache_flash did not finish in {CHILD_TIMEOUT_S} s (kernel hang)")
    assert run.returncode == 0, run.stderr[-4000:]
    assert "RESHAPE_AND_CACHE_OK" in run.stdout
