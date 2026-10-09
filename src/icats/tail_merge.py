"""Tail merging of query clusters (section 4.3, Appendix E.1 Algorithm 1).

The routine is one head. Heads are independent: call it once per head, which
is the paper's "same head" restriction on identical-mask merges.

Steps, matching Algorithm 1:

1. Merge query clusters that share an identical key-cluster mask. Sizes are
   summed and token labels are rewritten. The attended key set is unchanged
   because the union of identical masks is that mask. If there are no
   duplicates, labels and masks are left as they are (the paper's else branch).
2. Tail length is cluster size mod block size. Tail tokens are the last that
   many members of the cluster in increasing token-index order (the incomplete
   final block when members are packed in index order).
3. Compatibility is Hamming distance between binary key-cluster masks, the
   popcount of the XOR. That is the same distance as PackToInt64 followed by
   an integer Hamming distance, without truncating masks wider than 64 bits.
   Self-pairs, clusters with no tail, and pairs whose tails sum to more than
   the block size are forbidden.
4. Two rounds. A round pairs mutual nearest neighbors with i < j, unions the
   masks onto i, and moves only j's tail tokens onto i. The lower index is the
   target because the paper writes the pair as (i, j) with i < j.
5. Empty clusters are removed and the remaining clusters are reindexed.

The merge does not drop or duplicate query tokens: every token keeps exactly
one label. Padding blocks are sum_c ceil(size_c / block_size).
"""

from __future__ import annotations

import numpy as np


def padding_block_count(sizes, block_size):
    """Sum of ceil(size / block_size) over clusters. Empty clusters add nothing."""
    sizes = np.asarray(sizes, dtype=np.int64)
    block_size = int(block_size)
    if block_size < 1:
        raise ValueError("block_size must be positive")
    return int(np.sum((sizes + block_size - 1) // block_size))


def compute_is_tail(labels, sizes, block_size):
    """Flag the incomplete final block of each cluster.

    Members are taken in increasing token index. The last (size mod block_size)
    members are the tail. A cluster whose size is a multiple of the block size
    has no tail tokens.
    """
    labels = np.asarray(labels)
    sizes = np.asarray(sizes)
    is_tail = np.zeros(labels.shape[0], dtype=bool)
    for cluster, size in enumerate(sizes):
        tail = int(size) % int(block_size)
        if tail <= 0:
            continue
        members = np.flatnonzero(labels == cluster)
        if members.size != int(size):
            raise ValueError(
                f"cluster {cluster} size {int(size)} does not match {members.size} tokens"
            )
        is_tail[members[-tail:]] = True
    return is_tail


def _hamming(masks):
    bits = masks.astype(np.int16)
    return np.sum(bits[:, None, :] != bits[None, :, :], axis=-1).astype(np.float64)


def _constrain_distances(distances, tail_lengths, block_size):
    distances = np.array(distances, dtype=np.float64, copy=True)
    np.fill_diagonal(distances, np.inf)
    no_tail = tail_lengths <= 0
    distances[no_tail, :] = np.inf
    distances[:, no_tail] = np.inf
    overflow = tail_lengths[:, None] + tail_lengths[None, :] > int(block_size)
    distances[overflow] = np.inf
    return distances


def _merge_identical_masks(labels, masks, sizes):
    """Algorithm 1 lines 1-6. Remap only when a duplicate mask exists."""
    keys = [row.tobytes() for row in masks]
    first = {}
    mapping = np.empty(masks.shape[0], dtype=np.int32)
    representatives = []
    duplicate = False
    for cluster, key in enumerate(keys):
        if key in first:
            duplicate = True
            mapping[cluster] = first[key]
        else:
            first[key] = len(representatives)
            mapping[cluster] = len(representatives)
            representatives.append(cluster)
    if not duplicate:
        return labels, masks, sizes
    new_masks = masks[np.asarray(representatives)]
    new_labels = mapping[labels]
    new_sizes = np.bincount(new_labels, minlength=len(representatives)).astype(np.int32)
    return new_labels, new_masks, new_sizes


def _mutual_pairs(distances):
    nearest = np.argmin(distances, axis=1)
    pairs = []
    for i, j in enumerate(nearest):
        j = int(j)
        if i < j and int(nearest[j]) == i and np.isfinite(distances[i, j]):
            pairs.append((i, j))
    return pairs


def _compact(labels, masks, sizes):
    keep = np.asarray(sizes) > 0
    old_ids = np.flatnonzero(keep)
    remap = np.full(sizes.shape[0], -1, dtype=np.int32)
    remap[old_ids] = np.arange(old_ids.shape[0], dtype=np.int32)
    new_labels = remap[labels]
    if np.any(new_labels < 0):
        raise RuntimeError("a token was assigned to an empty cluster")
    return new_labels, sizes[keep].astype(np.int32), masks[keep]


def tail_merge(labels, masks, block_size):
    """Run Algorithm 1. Return (labels, sizes, masks) with empty clusters removed.

    labels: (N,) cluster id per query token, in 0 .. C-1.
    masks: (C, W) binary key-cluster mask per query cluster.
    block_size: query block size. Tests use a small block; the config default
    is 128.
    """
    block_size = int(block_size)
    if block_size < 1:
        raise ValueError("block_size must be positive")
    labels = np.array(np.asarray(labels), dtype=np.int32, copy=True)
    masks = np.array(np.asarray(masks), dtype=bool, copy=True)
    if labels.ndim != 1:
        raise ValueError("labels must be a 1-D per-head array; run one head at a time")
    if masks.ndim != 2:
        raise ValueError("masks must have shape (clusters, key_clusters)")
    if labels.size == 0:
        raise ValueError("labels must be non-empty")
    n_clusters = masks.shape[0]
    if int(labels.max()) >= n_clusters or int(labels.min()) < 0:
        raise ValueError("label ids must lie in 0 .. masks.shape[0]-1")
    sizes = np.bincount(labels, minlength=n_clusters).astype(np.int32)

    labels, masks, sizes = _merge_identical_masks(labels, masks, sizes)
    tail_lengths = sizes % block_size
    if np.any(tail_lengths > 0):
        is_tail = compute_is_tail(labels, sizes, block_size)
        distances = _constrain_distances(_hamming(masks), tail_lengths, block_size)
        for round_index in range(2):
            pairs = _mutual_pairs(distances)
            if not pairs:
                break
            for i, j in pairs:
                masks[i] = masks[i] | masks[j]
                sizes[i] = sizes[i] + tail_lengths[j]
                sizes[j] = sizes[j] - tail_lengths[j]
                move = (labels == j) & is_tail
                labels[move] = i
                distances[j, :] = np.inf
                distances[:, j] = np.inf
            if round_index < 1:
                tail_lengths = sizes % block_size
                is_tail = compute_is_tail(labels, sizes, block_size)
                distances = _constrain_distances(_hamming(masks), tail_lengths, block_size)

    labels, sizes, masks = _compact(labels, masks, sizes)
    if int(sizes.sum()) != int(labels.shape[0]):
        raise RuntimeError("tail merge dropped or duplicated query tokens")
    return labels, sizes, masks
