"""Interaction-aware Lloyd clustering (section 4.1).

Tokens are row vectors. The key-clustering objective (equation 1) is

    L = sum_t sum_{j in B_t} ||Q (k_j - mu_t)^T||_2^2
      = sum_t sum_{j in B_t} (k_j - mu_t) (Q^T Q) (k_j - mu_t)^T

Q^T Q is approximated by M_Q = sum_r |A_r| mu_r^T mu_r, and K^T K by
M_K = sum_t |B_t| mu_t^T mu_t. For a fixed SPD metric the assignment cost is
the quadratic distance (equation 2)

    (k_j - mu) M (k_j - mu)^T = k_j M k_j^T + mu M mu^T - 2 k_j M mu^T

and the centroid update is the arithmetic mean, which minimizes that cost.

Token-side terms (k M and k M k^T) are precomputed once per clustering call.
Centroid-side terms are recomputed each iteration. If a cluster is empty, those
token-side terms are recomputed after the jitter below, because M changed.

Metric handling: M is always symmetrized. If M is singular (smallest eigenvalue
<= 0) or a cluster is empty, add METRIC_JITTER * I with METRIC_JITTER = 1e-6.
The jitter is not applied to a well-conditioned metric, and it is not folded
into metric_from_centroids (singleton clusters must reproduce Q^T Q exactly).

Schedule:
    First sparse timestep (M_K_prev is None): Euclidean k-means on queries,
    build M_Q, cluster keys with M_Q, build M_K.
    Later timesteps: cluster queries with the previous M_K, build M_Q, cluster
    keys with M_Q, build the next M_K.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from icats.config import ICATSConfig

METRIC_JITTER = 1e-6


def prepare_metric(metric, d, dtype=None, jitter=METRIC_JITTER):
    """Symmetrize a metric and jitter it if it is singular.

    Returns (M, jittered). A None metric is Euclidean (the identity) and is
    not jittered. Singularity is the smallest eigenvalue being <= 0, checked
    in float64 so a tiny negative eigenvalue is not missed.
    """
    if dtype is None:
        dtype = jnp.float32
    # JAX defaults to float32. Do not ask it for float64 unless x64 is enabled.
    if jnp.dtype(dtype) == jnp.float64 and not jax.config.jax_enable_x64:
        dtype = jnp.float32
    eye = jnp.eye(d, dtype=dtype)
    if metric is None:
        return eye, False
    M64 = np.asarray(metric, dtype=np.float64)
    if M64.shape != (d, d):
        raise ValueError(f"metric must have shape ({d}, {d}), got {M64.shape}")
    M64 = 0.5 * (M64 + M64.T)
    singular = bool(np.linalg.eigvalsh(M64).min() <= 0.0)
    M = jnp.asarray(M64, dtype=dtype)
    if singular:
        M = M + jnp.asarray(jitter, dtype=dtype) * eye
        return M, True
    return M, False


def metric_from_centroids(centroids, sizes):
    """M = sum_r |cluster_r| mu_r^T mu_r.

    Singleton clusters with unit sizes reproduce X^T X when centroids are the
    rows of X. Empty clusters (size 0) do not contribute. This function does
    not add jitter.
    """
    centroids = jnp.asarray(centroids)
    sizes = jnp.asarray(sizes, dtype=centroids.dtype)
    if centroids.ndim != 2:
        raise ValueError("centroids must be (clusters, d)")
    if sizes.shape != (centroids.shape[0],):
        raise ValueError("sizes must have one entry per centroid")
    weighted = centroids * sizes[:, None]
    return weighted.T @ centroids


def _token_terms(tokens, metric):
    """Precompute k M and k M k^T."""
    KM = tokens @ metric
    quad = jnp.sum(KM * tokens, axis=-1)
    return KM, quad


def quadratic_distances(tokens, centroids, metric):
    """Equation (2) distances from each token to each centroid.

    Uses precomputed token-side terms and fresh centroid-side terms. metric
    is symmetrized. Jitter is not applied here; lloyd_cluster prepares M.
    """
    tokens = jnp.asarray(tokens)
    centroids = jnp.asarray(centroids, dtype=tokens.dtype)
    metric = jnp.asarray(metric, dtype=tokens.dtype)
    metric = 0.5 * (metric + metric.T)
    KM, token_quad = _token_terms(tokens, metric)
    cent_quad = jnp.sum((centroids @ metric) * centroids, axis=-1)
    dots = KM @ centroids.T
    return token_quad[:, None] + cent_quad[None, :] - 2.0 * dots


def interaction_objective(tokens, labels, centroids, metric):
    """Equation (1) right-hand side for a fixed metric: sum (k - mu) M (k - mu)^T."""
    tokens = jnp.asarray(tokens)
    centroids = jnp.asarray(centroids, dtype=tokens.dtype)
    metric = jnp.asarray(metric, dtype=tokens.dtype)
    metric = 0.5 * (metric + metric.T)
    labels = np.asarray(labels)
    total = jnp.zeros((), dtype=tokens.dtype)
    for t in range(int(centroids.shape[0])):
        member = labels == t
        if not np.any(member):
            continue
        resid = tokens[np.flatnonzero(member)] - centroids[t]
        total = total + jnp.sum(jnp.einsum("ij,jk,ik->i", resid, metric, resid))
    return total


def explicit_interaction_residual(queries, tokens, labels, centroids):
    """Equation (1) left-hand side: sum ||Q (k_j - mu)^T||_2^2."""
    queries = jnp.asarray(queries)
    tokens = jnp.asarray(tokens, dtype=queries.dtype)
    centroids = jnp.asarray(centroids, dtype=queries.dtype)
    labels = np.asarray(labels)
    total = jnp.zeros((), dtype=queries.dtype)
    for t in range(int(centroids.shape[0])):
        member = labels == t
        if not np.any(member):
            continue
        resid = tokens[np.flatnonzero(member)] - centroids[t]
        interaction = queries @ resid.T
        total = total + jnp.sum(interaction * interaction)
    return total


def kmeans_pp_init(tokens, n_clusters, seed):
    """Euclidean k-means++ centroids. Independent of the clustering metric.

    The same init can be passed to Euclidean and interaction-aware Lloyd runs
    so the comparison differs only in the assignment metric.
    """
    tokens = jnp.asarray(tokens)
    n, _d = tokens.shape
    if n_clusters < 1 or n_clusters > n:
        raise ValueError(f"n_clusters must be in 1..{n}, got {n_clusters}")
    key = jax.random.PRNGKey(int(seed))
    key, sub = jax.random.split(key)
    indices = [int(jax.random.randint(sub, (), 0, n))]
    chosen = set(indices)
    for _ in range(1, n_clusters):
        cents = tokens[jnp.asarray(indices)]
        diff = tokens[:, None, :] - cents[None, :, :]
        nearest = jnp.min(jnp.sum(diff * diff, axis=-1), axis=-1)
        total = jnp.sum(nearest)
        key, sub = jax.random.split(key)
        if float(total) <= 0.0:
            pick = next(i for i in range(n) if i not in chosen)
        else:
            probs = nearest / total
            pick = int(jax.random.choice(sub, n, p=probs))
            if pick in chosen:
                pick = next(i for i in range(n) if i not in chosen)
        indices.append(pick)
        chosen.add(pick)
    return tokens[jnp.asarray(indices)]


def _mean_centroids(tokens, labels, n_clusters):
    """Arithmetic mean of each cluster. Empty clusters are left at 0."""
    labels = jnp.asarray(labels)
    one_hot = jax.nn.one_hot(labels, n_clusters, dtype=tokens.dtype)
    counts = one_hot.sum(axis=0)
    sums = one_hot.T @ tokens
    safe = jnp.where(counts > 0, counts, jnp.ones_like(counts))
    return sums / safe[:, None], counts


def _reinit_empty(tokens, labels, centroids, distances, empty):
    """Move each empty centroid onto a far token so the next assignment can fill it."""
    tokens_np = np.asarray(tokens)
    centroids_np = np.array(centroids, copy=True)
    labels_np = np.asarray(labels)
    dist_np = np.asarray(distances)
    point_dist = dist_np[np.arange(tokens_np.shape[0]), labels_np]
    order = np.argsort(-point_dist)
    used = set()
    for t in np.flatnonzero(np.asarray(empty)):
        for point in order:
            point = int(point)
            if point not in used:
                centroids_np[int(t)] = tokens_np[point]
                used.add(point)
                break
    return jnp.asarray(centroids_np, dtype=centroids.dtype)


def lloyd_cluster(
    tokens,
    n_clusters,
    metric=None,
    n_iters=10,
    init_centroids=None,
    seed=0,
    jitter=METRIC_JITTER,
):
    """Lloyd clustering under a fixed SPD metric.

    metric None is Euclidean (identity). Assignment uses equation (2). The
    centroid update is the arithmetic mean. Token-side terms are precomputed
    once, unless an empty cluster forces a jitter update of M.

    Returns labels (n,), centroids (n_clusters, d), sizes (n_clusters,).
    """
    tokens = jnp.asarray(tokens)
    if tokens.ndim != 2:
        raise ValueError("tokens must have shape (n, d)")
    n, d = tokens.shape
    n_clusters = int(n_clusters)
    n_iters = int(n_iters)
    if n_clusters < 1 or n_clusters > n:
        raise ValueError(f"n_clusters must be in 1..{n}, got {n_clusters}")
    if n_iters < 1:
        raise ValueError("n_iters must be positive")

    M, jittered = prepare_metric(metric, d, dtype=tokens.dtype, jitter=jitter)
    KM, token_quad = _token_terms(tokens, M)

    if init_centroids is None:
        centroids = kmeans_pp_init(tokens, n_clusters, seed)
    else:
        centroids = jnp.asarray(init_centroids, dtype=tokens.dtype)
        if centroids.shape != (n_clusters, d):
            raise ValueError(
                f"init_centroids must have shape ({n_clusters}, {d}), got {centroids.shape}"
            )
        centroids = jnp.array(centroids)

    labels = jnp.zeros((n,), dtype=jnp.int32)
    for it in range(n_iters):
        distances = token_quad[:, None] + jnp.sum((centroids @ M) * centroids, axis=-1)[None, :]
        distances = distances - 2.0 * (KM @ centroids.T)
        new_labels = jnp.argmin(distances, axis=1).astype(jnp.int32)
        if it > 0 and bool(jnp.array_equal(new_labels, labels)):
            labels = new_labels
            break
        labels = new_labels
        centroids, counts = _mean_centroids(tokens, labels, n_clusters)
        empty = counts <= 0
        if bool(jnp.any(empty)):
            if not jittered:
                M = M + jnp.asarray(jitter, dtype=M.dtype) * jnp.eye(d, dtype=M.dtype)
                KM, token_quad = _token_terms(tokens, M)
                jittered = True
            centroids = _reinit_empty(tokens, labels, centroids, distances, empty)

    centroids, counts = _mean_centroids(tokens, labels, n_clusters)
    sizes = counts.astype(jnp.int32)
    return labels, centroids, sizes


@dataclass
class ClusterResult:
    query_labels: jnp.ndarray
    query_centroids: jnp.ndarray
    query_sizes: jnp.ndarray
    key_labels: jnp.ndarray
    key_centroids: jnp.ndarray
    key_sizes: jnp.ndarray
    M_Q: jnp.ndarray
    M_K: jnp.ndarray


def cluster_qk(
    queries,
    keys,
    n_query_clusters=None,
    n_key_clusters=None,
    M_K_prev=None,
    n_iters=None,
    seed=0,
    config=None,
    jitter=None,
):
    """One interaction-aware clustering step for queries and keys.

    M_K_prev is None on the first sparse timestep (queries use Euclidean
    k-means). Later timesteps pass the previous M_K and cluster queries with
    that quadratic form. Keys are always clustered with the M_Q just built
    from the query centroids. Key initialization uses seed + 1 so it does not
    repeat the query initialization stream.
    """
    config = config if config is not None else ICATSConfig()
    if n_query_clusters is None:
        n_query_clusters = config.n_query_clusters
    if n_key_clusters is None:
        n_key_clusters = config.n_key_clusters
    if n_iters is None:
        n_iters = config.first_sparse_iters if M_K_prev is None else config.later_iters
    if jitter is None:
        jitter = config.jitter

    query_metric = None if M_K_prev is None else M_K_prev
    q_labels, q_centroids, q_sizes = lloyd_cluster(
        queries,
        n_query_clusters,
        metric=query_metric,
        n_iters=n_iters,
        seed=seed,
        jitter=jitter,
    )
    M_Q = metric_from_centroids(q_centroids, q_sizes)
    k_labels, k_centroids, k_sizes = lloyd_cluster(
        keys,
        n_key_clusters,
        metric=M_Q,
        n_iters=n_iters,
        seed=int(seed) + 1,
        jitter=jitter,
    )
    M_K = metric_from_centroids(k_centroids, k_sizes)
    return ClusterResult(
        query_labels=q_labels,
        query_centroids=q_centroids,
        query_sizes=q_sizes,
        key_labels=k_labels,
        key_centroids=k_centroids,
        key_sizes=k_sizes,
        M_Q=M_Q,
        M_K=M_K,
    )
