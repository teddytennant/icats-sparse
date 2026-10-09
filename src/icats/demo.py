"""Small CPU demo of one iCATS step. Not a video generator."""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp

from icats.attention import dense_attention, sparse_attention
from icats.clustering import cluster_qk
from icats.mask import build_cluster_mask
from icats.snr import synthetic_snr_schedule
from icats.tail_merge import padding_block_count, tail_merge


def main():
    seed = 0
    n_tokens = 64
    width = 16
    n_query_clusters = 4
    n_key_clusters = 8
    block_size = 8
    num_steps = 50
    t_sparse = 40
    t_eval = 25
    n_iters = 20

    key = jax.random.PRNGKey(seed)
    key_q, key_k, key_v = jax.random.split(key, 3)
    queries = jax.random.normal(key_q, (n_tokens, width))
    keys = jax.random.normal(key_k, (n_tokens, width))
    values = jax.random.normal(key_v, (n_tokens, width))

    clustered = cluster_qk(
        queries,
        keys,
        n_query_clusters=n_query_clusters,
        n_key_clusters=n_key_clusters,
        M_K_prev=None,
        n_iters=n_iters,
        seed=seed,
    )
    schedule = synthetic_snr_schedule(num_steps, t_sparse=t_sparse, p_min=0.7, tau=0.999)
    eval_index = int(jnp.where(schedule.timesteps == t_eval)[0][0])
    top_p = float(schedule.top_p[eval_index])
    min_p = float(schedule.min_p[eval_index])

    _scores, cluster_mask = build_cluster_mask(
        clustered.query_centroids,
        clustered.key_centroids,
        top_p,
        min_p,
    )
    pad_before = padding_block_count(clustered.query_sizes, block_size)
    merged_labels, merged_sizes, merged_masks = tail_merge(
        clustered.query_labels,
        cluster_mask,
        block_size,
    )
    pad_after = padding_block_count(merged_sizes, block_size)

    sparse = sparse_attention(
        queries,
        keys,
        values,
        merged_labels,
        clustered.key_labels,
        merged_masks,
    )
    dense = dense_attention(queries, keys, values)
    dense_norm = float(jnp.linalg.norm(dense))
    if dense_norm == 0.0:
        relative_error = 0.0
    else:
        relative_error = float(jnp.linalg.norm(sparse - dense)) / dense_norm

    n_query_before = int(jnp.sum(clustered.query_sizes > 0))
    n_query_after = int(merged_sizes.shape[0])
    n_key = int(jnp.sum(clustered.key_sizes > 0))
    print(f"seed={seed} n={n_tokens} d={width} block_size={block_size}")
    print(f"query_clusters_before={n_query_before}")
    print(f"query_clusters_after={n_query_after}")
    print(f"key_clusters={n_key}")
    print(f"timestep={t_eval} t_sparse={t_sparse} T={num_steps}")
    print(f"top_p={top_p:.6f}")
    print(f"min_p={min_p:.6f}")
    print(f"padding_blocks_before={pad_before}")
    print(f"padding_blocks_after={pad_after}")
    print(f"relative_error={relative_error:.6e}")


if __name__ == "__main__":
    main()
