"""Sparse attention matches dense when every key is kept, and changes if a key is dropped."""

import numpy as np

from icats.attention import dense_attention, sparse_attention


def test_full_mask_matches_dense_and_dropped_key_changes_output():
    rng = np.random.default_rng(1)
    n_tokens = 8
    width = 4
    queries = rng.normal(size=(n_tokens, width))
    keys = rng.normal(size=(n_tokens, width))
    values = rng.normal(size=(n_tokens, width))
    query_labels = np.array([0, 0, 1, 1, 0, 1, 1, 0], dtype=np.int32)
    key_labels = np.array([0, 1, 0, 1, 1, 0, 1, 0], dtype=np.int32)
    full_mask = np.ones((2, 2), dtype=bool)

    sparse = np.asarray(
        sparse_attention(queries, keys, values, query_labels, key_labels, full_mask)
    )
    dense = np.asarray(dense_attention(queries, keys, values))
    np.testing.assert_allclose(sparse, dense, rtol=1e-4, atol=1e-4)

    # One key per cluster so dropping a cluster drops that key. Align key 0
    # with query 0 so the key has nonzero attention, and give it a unique value.
    queries = np.zeros((2, width))
    queries[0, 0] = 4.0
    queries[1, 1] = 4.0
    keys = np.eye(width)
    values = np.eye(width)
    query_labels = np.zeros(2, dtype=np.int32)
    key_labels = np.arange(width, dtype=np.int32)
    dense = np.asarray(dense_attention(queries, keys, values))
    scale = width ** -0.5
    weights = np.exp(queries @ keys.T * scale)
    weights = weights / weights.sum(axis=-1, keepdims=True)
    assert weights[0, 0] > 0.0

    dropped = np.ones((1, width), dtype=bool)
    dropped[0, 0] = False
    sparse_dropped = np.asarray(
        sparse_attention(queries, keys, values, query_labels, key_labels, dropped)
    )
    assert np.max(np.abs(sparse_dropped - dense)) > 1e-3
