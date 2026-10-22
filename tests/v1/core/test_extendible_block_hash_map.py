# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the extendible-hash prefix-cache index (research branch)."""

import hashlib
import random

from vllm.v1.core.extendible_block_hash_map import (
    MAX_GLOBAL_DEPTH,
    ExtendibleBlockHashMap,
    short_key,
)
from vllm.v1.core.kv_cache_utils import KVCacheBlock, make_block_hash_with_group_id


def make_key(ident: int, group_id: int = 0):
    digest = hashlib.sha256(str(ident).encode()).digest()
    return make_block_hash_with_group_id(digest, group_id)


def test_short_key_is_fnv1a_64_of_group_id_then_digest():
    # key = digest + 4-byte group id; b"ar" + b"foob" hashes as b"foob" + b"ar".
    assert short_key(b"ar" + b"foob") == 0x85944171F73967E8


def test_insert_if_missing_first_writer_wins():
    index = ExtendibleBlockHashMap()
    first, second = KVCacheBlock(1), KVCacheBlock(2)
    key = make_key(1)
    assert index.insert_if_missing(key, first) is first
    assert index.insert_if_missing(key, second) is first
    assert index.get_one_block(key) is first
    assert index.contain(key, 1) and not index.contain(key, 2)
    assert len(index) == 1


def test_pop_requires_matching_block_id_and_frees_the_key():
    index = ExtendibleBlockHashMap()
    block = KVCacheBlock(3)
    key = make_key(1)
    index.insert_if_missing(key, block)
    assert index.pop(key, 4) is None
    assert index.pop(key, 3) is block
    assert index.get_one_block(key) is None
    assert len(index) == 0
    other = KVCacheBlock(5)
    assert index.insert_if_missing(key, other) is other


def test_group_ids_are_distinct_keys():
    index = ExtendibleBlockHashMap()
    a, b = KVCacheBlock(1), KVCacheBlock(2)
    index.insert_if_missing(make_key(1, 0), a)
    index.insert_if_missing(make_key(1, 1), b)
    assert index.get_one_block(make_key(1, 0)) is a
    assert index.get_one_block(make_key(1, 1)) is b


def test_random_operations_match_a_dict():
    rng = random.Random(0)
    index = ExtendibleBlockHashMap()
    blocks = [KVCacheBlock(i) for i in range(2000)]
    model: dict[bytes, KVCacheBlock] = {}
    for _ in range(20000):
        key = make_key(rng.randrange(1500), rng.randrange(2))
        roll = rng.random()
        if roll < 0.5:
            block = rng.choice(blocks)
            got = index.insert_if_missing(key, block)
            assert got is model.setdefault(key, block)
        elif roll < 0.8:
            held = model.get(key)
            block = held or rng.choice(blocks)
            popped = index.pop(key, block.block_id)
            assert (popped is not None) == (held is not None)
            model.pop(key, None)
        else:
            assert index.get_one_block(key) is model.get(key)
    assert len(index) == len(model)
    assert index.global_depth <= MAX_GLOBAL_DEPTH


def test_clear_empties_the_index_and_resets_the_depth():
    index = ExtendibleBlockHashMap()
    for ident in range(500):
        index.insert_if_missing(make_key(ident), KVCacheBlock(ident))
    assert index.global_depth > 0
    index.clear()
    assert len(index) == 0 and index.global_depth == 0
    assert index.get_one_block(make_key(1)) is None


def test_insert_leaves_a_duplicate_key_unindexed():
    index = ExtendibleBlockHashMap()
    first, second = KVCacheBlock(1), KVCacheBlock(2)
    index.insert(make_key(1), first)
    index.insert(make_key(1), second)
    assert index.get_one_block(make_key(1)) is first
    assert len(index) == 1
