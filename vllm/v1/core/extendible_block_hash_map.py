# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Extendible-hash prefix-cache index, a drop-in for `BlockHashToBlockMap`.

Research branch. Pure Python, so it validates behavior, not speed: it will be
slower than the dict it replaces. Design, measurements and the executable model
this was ported from: github.com/zbrad/kvcache-research
(design/extendible-hash-prefix-index.md, docs/short-key-hash.md).

Differences from `BlockHashToBlockMap` that callers must know about:
- One block per key, first writer wins (`insert_if_missing`). A block refused
  because its key is held by another block stays unindexed.
- A full bucket that cannot split because the directory is at
  `MAX_GLOBAL_DEPTH` refuses the insert as well (returns None).
"""

from vllm.v1.core.kv_cache_utils import BlockHashWithGroupId, KVCacheBlock

SLOTS_PER_BUCKET = 32
MAX_USED = 24  # LIVE + DEAD slots at which a bucket is rebuilt or split
REBUILD_DEAD = 8  # tombstones at which a rebuild beats a split
HASH_BITS = 64
MAX_GLOBAL_DEPTH = 24
GROUP_ID_BYTES = 4

EMPTY, LIVE, DEAD = 0, 1, 2

FNV64_OFFSET_BASIS = 0xCBF29CE484222325
FNV64_PRIME = 0x100000001B3
MASK_64 = (1 << 64) - 1


def short_key(key: BlockHashWithGroupId) -> int:
    """FNV-1a 64 over the group id, then the digest.

    `key` is the digest followed by a 4-byte group id. The group id is hashed
    first: appended last, its variants of one digest share a directory slot.
    """
    value = FNV64_OFFSET_BASIS
    for byte in key[-GROUP_ID_BYTES:] + key[:-GROUP_ID_BYTES]:
        value = ((value ^ byte) * FNV64_PRIME) & MASK_64
    return value


class _Bucket:
    """A page of 32 slots sharing the top `local_depth` bits of their short keys."""

    __slots__ = ("local_depth", "state", "shortkeys", "keys", "blocks", "live", "dead")

    def __init__(self, local_depth: int):
        self.local_depth = local_depth
        self.state = bytearray(SLOTS_PER_BUCKET)
        self.shortkeys = [0] * SLOTS_PER_BUCKET
        self.keys: list[BlockHashWithGroupId | None] = [None] * SLOTS_PER_BUCKET
        self.blocks: list[KVCacheBlock | None] = [None] * SLOTS_PER_BUCKET

    @property
    def used(self) -> int:
        return self.live + self.dead


def _new_bucket(local_depth: int) -> _Bucket:
    bucket = _Bucket(local_depth)
    bucket.live = 0
    bucket.dead = 0
    return bucket


class ExtendibleBlockHashMap:
    """Maps `BlockHashWithGroupId` to one `KVCacheBlock` (extendible hashing).

    A single writer is assumed, as for the scheduler's block pool. Only an
    EMPTY slot ends a probe; a DEAD slot (tombstone) never does.
    """

    def __init__(self):
        self._global_depth = 0
        self._directory: list[_Bucket] = [_new_bucket(0)]
        self._live = 0

    def get_one_block(self, key: BlockHashWithGroupId) -> KVCacheBlock | None:
        """Gets the block indexed under the key."""
        shortkey = short_key(key)
        bucket, index = self._find(shortkey, key)
        return None if bucket is None else bucket.blocks[index]

    def contain(self, key: BlockHashWithGroupId, block_id: int) -> bool:
        """Checks whether the key maps to the given block ID."""
        block = self.get_one_block(key)
        return block is not None and block.block_id == block_id

    def insert_if_missing(
        self, key: BlockHashWithGroupId, block: KVCacheBlock
    ) -> KVCacheBlock | None:
        """Index `block` under `key` unless a LIVE entry already holds the key.

        Returns:
            `block` if it was inserted or was already there, the other block
            if one holds the key, or None if a full bucket cannot split.
        """
        shortkey = short_key(key)
        while True:
            bucket = self._bucket_for(shortkey)
            first_dead = -1
            empty = -1
            for index in self._probe(shortkey):
                state = bucket.state[index]
                if state == EMPTY:
                    empty = index
                    break
                if state == LIVE:
                    if (
                        bucket.shortkeys[index] == shortkey
                        and bucket.keys[index] == key
                    ):
                        return bucket.blocks[index]
                elif first_dead < 0:
                    first_dead = index
            assert empty >= 0, "a full probe cycle found no EMPTY slot"
            if first_dead < 0 and bucket.used >= MAX_USED:
                if not self._restructure(bucket):
                    return None
                continue
            target = empty if first_dead < 0 else first_dead
            if first_dead >= 0:
                bucket.dead -= 1
            self._write(bucket, target, shortkey, key, block)
            bucket.live += 1
            self._live += 1
            return block

    def pop(self, key: BlockHashWithGroupId, block_id: int) -> KVCacheBlock | None:
        """Tombstones the entry for `key` if it points at `block_id`."""
        shortkey = short_key(key)
        bucket, index = self._find(shortkey, key)
        if bucket is None:
            return None
        block = bucket.blocks[index]
        if block is None or block.block_id != block_id:
            return None
        bucket.state[index] = DEAD
        bucket.blocks[index] = None
        bucket.live -= 1
        bucket.dead += 1
        self._live -= 1
        return block

    def __len__(self) -> int:
        return self._live

    @property
    def global_depth(self) -> int:
        return self._global_depth

    @staticmethod
    def _probe(shortkey: int):
        index = shortkey & (SLOTS_PER_BUCKET - 1)
        step = ((shortkey >> 5) & (SLOTS_PER_BUCKET - 1)) | 1
        for _ in range(SLOTS_PER_BUCKET):
            yield index
            index = (index + step) & (SLOTS_PER_BUCKET - 1)

    def _bucket_for(self, shortkey: int) -> _Bucket:
        depth = self._global_depth
        return self._directory[shortkey >> (HASH_BITS - depth) if depth else 0]

    def _find(
        self, shortkey: int, key: BlockHashWithGroupId
    ) -> tuple[_Bucket | None, int]:
        bucket = self._bucket_for(shortkey)
        for index in self._probe(shortkey):
            state = bucket.state[index]
            if state == EMPTY:
                break
            if (
                state == LIVE
                and bucket.shortkeys[index] == shortkey
                and bucket.keys[index] == key
            ):
                return bucket, index
        return None, -1

    @staticmethod
    def _write(
        bucket: _Bucket,
        index: int,
        shortkey: int,
        key: BlockHashWithGroupId,
        block: KVCacheBlock,
    ) -> None:
        bucket.shortkeys[index] = shortkey
        bucket.keys[index] = key
        bucket.blocks[index] = block
        bucket.state[index] = LIVE

    def _place_fresh(self, bucket: _Bucket, source: _Bucket, src_index: int) -> None:
        shortkey = source.shortkeys[src_index]
        for index in self._probe(shortkey):
            if bucket.state[index] == EMPTY:
                key = source.keys[src_index]
                block = source.blocks[src_index]
                assert key is not None and block is not None
                self._write(bucket, index, shortkey, key, block)
                bucket.live += 1
                return
        raise AssertionError("a fresh bucket has no EMPTY slot")

    def _restructure(self, bucket: _Bucket) -> bool:
        """Rebuilds in place if tombstones dominate, else splits.

        Returns False if the bucket cannot split (directory at the depth cap).
        """
        live = [i for i in range(SLOTS_PER_BUCKET) if bucket.state[i] == LIVE]
        if bucket.dead >= REBUILD_DEAD:
            fresh = _new_bucket(bucket.local_depth)
            for i in live:
                self._place_fresh(fresh, bucket, i)
            self._directory = [fresh if b is bucket else b for b in self._directory]
            return True
        if bucket.local_depth == self._global_depth:
            if self._global_depth >= MAX_GLOBAL_DEPTH:
                return False
            self._directory = [b for b in self._directory for _ in range(2)]
            self._global_depth += 1
        low = _new_bucket(bucket.local_depth + 1)
        high = _new_bucket(bucket.local_depth + 1)
        shift = HASH_BITS - 1 - bucket.local_depth
        for i in live:
            self._place_fresh(
                high if (bucket.shortkeys[i] >> shift) & 1 else low, bucket, i
            )
        prefix_shift = self._global_depth - bucket.local_depth - 1
        for index, current in enumerate(self._directory):
            if current is bucket:
                self._directory[index] = high if (index >> prefix_shift) & 1 else low
        return True
