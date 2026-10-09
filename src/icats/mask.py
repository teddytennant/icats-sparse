"""Cluster-level key mask from centroid softmax scores (section 4.2).

The paper says query and key centroids approximate the block softmax scores
and that figure rows sum to 1, but it does not write the temperature. Scores
here are

    softmax_t(mu_r^Q · mu_t^K / sqrt(d))

with temperature 1/sqrt(d), the usual attention scale. Rows sum to 1.

For each query cluster, key clusters are taken in descending score until the
cumulative mass is at least top-p, and at least ceil(min-p * P) key clusters
are kept. min-p is a fraction of key clusters, not a probability mass. Every
query cluster receives a mask, so no query token is dropped when the mask is
applied through query labels.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


def cluster_softmax_scores(query_centroids, key_centroids):
    """Softmax over key clusters of centroid dots scaled by 1/sqrt(d)."""
    queries = jnp.asarray(query_centroids)
    keys = jnp.asarray(key_centroids, dtype=queries.dtype)
    if queries.ndim != 2 or keys.ndim != 2:
        raise ValueError("centroids must have shape (clusters, d)")
    if queries.shape[-1] != keys.shape[-1]:
        raise ValueError("query and key centroids must share d")
    scale = queries.shape[-1] ** -0.5
    logits = (queries @ keys.T) * scale
    return jax.nn.softmax(logits, axis=-1)


def select_key_clusters(scores, top_p, min_p):
    """Boolean mask (query clusters, key clusters) from top-p and min-p.

    top_p >= 1 keeps every key cluster. Otherwise each row keeps keys in
    descending score until both the cumulative mass reaches top_p and the row
    has at least ceil(min_p * P) keys. At least one key cluster is kept so a
    query token is never left with an empty support.
    """
    scores_np = np.asarray(scores, dtype=np.float64)
    if scores_np.ndim != 2:
        raise ValueError("scores must have shape (query_clusters, key_clusters)")
    n_query, n_key = scores_np.shape
    if n_key < 1:
        raise ValueError("at least one key cluster is required")
    top_p = float(top_p)
    min_p = float(min_p)
    min_count = int(np.ceil(min_p * n_key))
    min_count = max(1, min(min_count, n_key))
    if top_p >= 1.0 - 1e-12:
        return np.ones((n_query, n_key), dtype=bool)
    order = np.argsort(-scores_np, axis=1)
    mask = np.zeros((n_query, n_key), dtype=bool)
    for row in range(n_query):
        cumulative = 0.0
        for rank, key_index in enumerate(order[row]):
            mask[row, int(key_index)] = True
            cumulative += scores_np[row, int(key_index)]
            if (rank + 1) >= min_count and cumulative >= top_p - 1e-12:
                break
    return mask


def build_cluster_mask(query_centroids, key_centroids, top_p, min_p):
    """Score key clusters and select a top-p / min-p mask."""
    scores = cluster_softmax_scores(query_centroids, key_centroids)
    mask = select_key_clusters(scores, top_p, min_p)
    return scores, mask
