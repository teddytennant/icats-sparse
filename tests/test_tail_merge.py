"""Algorithm 1: identical-mask merge, tail reassignment, token and padding checks."""

import numpy as np

from icats.tail_merge import padding_block_count, tail_merge


def test_identical_masks_merge_without_changing_key_set():
    block_size = 8
    labels = np.array([0] * 5 + [1] * 3 + [2] * 8, dtype=np.int32)
    masks = np.array(
        [
            [1, 0, 1],
            [1, 0, 1],
            [0, 1, 0],
        ],
        dtype=bool,
    )
    before = padding_block_count(np.bincount(labels), block_size)
    merged_labels, merged_sizes, merged_masks = tail_merge(labels, masks, block_size)

    assert merged_labels.shape == labels.shape
    assert int(merged_sizes.sum()) == labels.shape[0]
    np.testing.assert_array_equal(
        np.bincount(merged_labels, minlength=merged_sizes.shape[0]),
        merged_sizes,
    )
    for token in range(labels.shape[0]):
        np.testing.assert_array_equal(merged_masks[merged_labels[token]], masks[labels[token]])
    after = padding_block_count(merged_sizes, block_size)
    assert after <= before
    assert after < before
    assert merged_sizes.shape[0] == 2
    np.testing.assert_array_equal(np.sort(merged_sizes), np.array([8, 8]))


def test_tail_is_reassigned_and_tokens_and_padding_are_preserved():
    block_size = 8
    # Clusters 0 and 1 share a mask and merge into a block-aligned cluster.
    # Clusters 2 and 3 keep distinct masks and a non-aligned tail, so the tail
    # of the higher index is reassigned.
    labels = np.array(
        [0] * 5 + [1] * 3 + [2] * 10 + [3] * 9,
        dtype=np.int32,
    )
    mask_a = np.array([1, 0, 1, 0], dtype=bool)
    # B is not a subset of C, so the merged mask is a strict superset of both.
    mask_b = np.array([1, 0, 0, 1], dtype=bool)
    mask_c = np.array([0, 1, 1, 0], dtype=bool)
    masks = np.stack([mask_a, mask_a, mask_b, mask_c])
    before = padding_block_count(np.bincount(labels), block_size)

    merged_labels, merged_sizes, merged_masks = tail_merge(labels, masks, block_size)

    assert merged_labels.shape == labels.shape
    assert int(merged_sizes.sum()) == labels.shape[0]
    assert np.all(merged_labels >= 0)
    assert np.all(merged_labels < merged_sizes.shape[0])
    np.testing.assert_array_equal(
        np.bincount(merged_labels, minlength=merged_sizes.shape[0]),
        merged_sizes,
    )

    for token in range(8):
        np.testing.assert_array_equal(merged_masks[merged_labels[token]], mask_a)

    # Token 26 is the sole tail token of the size-9 cluster and must move.
    # Token 25 is not a tail token and keeps mask C.
    assert not np.array_equal(merged_masks[merged_labels[26]], mask_c)
    np.testing.assert_array_equal(merged_masks[merged_labels[26]], mask_b | mask_c)
    np.testing.assert_array_equal(merged_masks[merged_labels[25]], mask_c)
    assert merged_labels[26] != merged_labels[25]

    after = padding_block_count(merged_sizes, block_size)
    assert after <= before
    assert after < before
