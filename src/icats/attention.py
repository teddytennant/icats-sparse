"""Reference sparse attention. Correctness check, not a GPU kernel.

Given query labels and a per-query-cluster key-cluster mask, each query attends
only to keys whose cluster is selected:

    softmax(Q K_sel^T / sqrt(d)) V_sel

No query token is dropped: the output has one row per query. A query whose
mask selects no keys raises, rather than being skipped. When the mask keeps
every key the result matches dense attention.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


def dense_attention(queries, keys, values):
    """Standard softmax(Q K^T / sqrt(d)) V."""
    queries = jnp.asarray(queries)
    keys = jnp.asarray(keys, dtype=queries.dtype)
    values = jnp.asarray(values, dtype=queries.dtype)
    scale = queries.shape[-1] ** -0.5
    logits = (queries @ keys.T) * scale
    weights = jax.nn.softmax(logits, axis=-1)
    return weights @ values


def token_key_mask(query_labels, key_labels, cluster_mask):
    """Expand a cluster mask to a boolean (n_query, n_key) token mask."""
    query_labels = np.asarray(query_labels)
    key_labels = np.asarray(key_labels)
    cluster_mask = np.asarray(cluster_mask, dtype=bool)
    if query_labels.ndim != 1 or key_labels.ndim != 1:
        raise ValueError("labels must be 1-D")
    if int(query_labels.max()) >= cluster_mask.shape[0] or int(query_labels.min()) < 0:
        raise ValueError("query labels are outside the cluster mask")
    if int(key_labels.max()) >= cluster_mask.shape[1] or int(key_labels.min()) < 0:
        raise ValueError("key labels are outside the cluster mask")
    return cluster_mask[query_labels][:, key_labels]


def sparse_attention(queries, keys, values, query_labels, key_labels, cluster_mask):
    """Attention restricted to the key clusters selected for each query cluster."""
    queries = jnp.asarray(queries)
    keys = jnp.asarray(keys, dtype=queries.dtype)
    values = jnp.asarray(values, dtype=queries.dtype)
    keep = token_key_mask(query_labels, key_labels, cluster_mask)
    if not np.all(np.any(keep, axis=1)):
        raise ValueError("every query token must attend to at least one key")
    scale = queries.shape[-1] ** -0.5
    logits = (queries @ keys.T) * scale
    logits = jnp.where(jnp.asarray(keep), logits, -jnp.inf)
    weights = jax.nn.softmax(logits, axis=-1)
    return weights @ values
