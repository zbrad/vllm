# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare BlockHashToBlockMap (dict) with ExtendibleBlockHashMap (research).

Run: PYTHONPATH=. VLLM_TARGET_DEVICE=empty python benchmarks/benchmark_prefix_index.py
Pure-Python timings of the map alone, per operation, best of several repeats.
"""

import hashlib
import random
import time

from vllm.v1.core.block_pool import BlockHashToBlockMap
from vllm.v1.core.extendible_block_hash_map import ExtendibleBlockHashMap
from vllm.v1.core.kv_cache_utils import KVCacheBlock, make_block_hash_with_group_id

SIZES = (1_000, 10_000, 100_000)
REPEATS = 5


def make_keys(count: int, salt: str = ""):
    return [
        make_block_hash_with_group_id(hashlib.sha256(f"{salt}{i}".encode()).digest(), 0)
        for i in range(count)
    ]


def best_ns_per_op(fn, ops: int) -> float:
    best = float("inf")
    for _ in range(REPEATS):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best / ops * 1e9


def bench(make_map, size: int) -> dict[str, float]:
    keys = make_keys(size)
    misses = make_keys(size, "miss")
    blocks = [KVCacheBlock(i) for i in range(size)]
    order = list(range(size))
    random.Random(0).shuffle(order)

    def fill(m):
        for i in order:
            m.insert_if_missing(keys[i], blocks[i])

    def time_insert():
        best = float("inf")
        for _ in range(REPEATS):
            m = make_map()
            start = time.perf_counter()
            fill(m)
            best = min(best, time.perf_counter() - start)
        return best / size * 1e9

    m = make_map()
    fill(m)

    def hit():
        get = m.get_one_block
        for i in order:
            get(keys[i])

    def miss():
        get = m.get_one_block
        for key in misses:
            get(key)

    def time_pop_insert():
        def run():
            for i in order:
                m.pop(keys[i], i)
            for i in order:
                m.insert_if_missing(keys[i], blocks[i])

        return best_ns_per_op(run, 2 * size)

    return {
        "insert": time_insert(),
        "get hit": best_ns_per_op(hit, size),
        "get miss": best_ns_per_op(miss, size),
        "pop+insert": time_pop_insert(),
    }


def main() -> None:
    print(f"{'size':>8} {'op':<11} {'dict ns':>9} {'ext ns':>9} {'ratio':>6}")
    for size in SIZES:
        base = bench(BlockHashToBlockMap, size)
        ext = bench(ExtendibleBlockHashMap, size)
        for op in base:
            print(
                f"{size:>8} {op:<11} {base[op]:>9.0f} {ext[op]:>9.0f} "
                f"{ext[op] / base[op]:>5.1f}x"
            )


if __name__ == "__main__":
    main()
