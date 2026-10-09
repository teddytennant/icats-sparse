"""Equation (1), interaction-aware clustering, and the singleton metric."""

import numpy as np
import pytest

from icats.clustering import (
    METRIC_JITTER,
    cluster_qk,
    explicit_interaction_residual,
    interaction_objective,
    lloyd_cluster,
    metric_from_centroids,
    prepare_metric,
    quadratic_distances,
)
from icats.config import ICATSConfig


def test_equation1_identity():
    rng = np.random.default_rng(0)
    queries = rng.normal(size=(20, 6))
    keys = rng.normal(size=(30, 6))
    labels = rng.integers(0, 4, size=30)
    centroids = np.stack([keys[labels == t].mean(axis=0) for t in range(4)])
    metric = queries.T @ queries

    quadratic = np.asarray(
        interaction_objective(keys, labels, centroids, metric), dtype=np.float64
    )
    explicit = np.asarray(
        explicit_interaction_residual(queries, keys, labels, centroids),
        dtype=np.float64,
    )

    reference = 0.0
    for cluster in range(4):
        residual = keys[labels == cluster] - centroids[cluster]
        reference += np.sum((queries @ residual.T) ** 2)
        reference_quad = np.sum(np.einsum("ij,jk,ik->i", residual, metric, residual))
        assert reference_quad == pytest.approx(
            np.sum((queries @ residual.T) ** 2), rel=1e-5, abs=1e-5
        )

    np.testing.assert_allclose(quadratic, explicit, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(quadratic, reference, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(explicit, reference, rtol=1e-4, atol=1e-4)

    distances = np.asarray(quadratic_distances(keys, centroids, metric), dtype=np.float64)
    gathered = distances[np.arange(keys.shape[0]), labels].sum()
    np.testing.assert_allclose(gathered, reference, rtol=1e-4, atol=1e-3)


def _anisotropic_keys(seed):
    """Queries with two large covariance eigenvalues, keys with a distractor partition."""
    rng = np.random.default_rng(seed)
    n_keys = 64
    n_queries = 48
    width = 6
    n_clusters = 4
    per_group = n_keys // n_clusters
    queries = np.zeros((n_queries, width))
    queries[:, 0] = rng.normal(size=n_queries) * 4.0
    queries[:, 1] = rng.normal(size=n_queries) * 4.0
    queries[:, 2:] = rng.normal(size=(n_queries, width - 2)) * 0.02
    signal = np.array([[0.0, 0.0], [3.0, 0.0], [0.0, 3.0], [3.0, 3.0]])
    distractor = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [8.0, 0.0, 0.0, 0.0],
            [0.0, 8.0, 0.0, 0.0],
            [0.0, 0.0, 8.0, 0.0],
        ]
    )
    signal_id = np.repeat(np.arange(n_clusters), per_group)
    distractor_id = np.tile(np.arange(n_clusters), per_group)
    keys = np.zeros((n_keys, width))
    keys[:, 0:2] = signal[signal_id] + rng.normal(size=(n_keys, 2)) * 0.15
    keys[:, 2:] = distractor[distractor_id] + rng.normal(size=(n_keys, 4)) * 0.15
    return queries, keys, n_clusters


def _kmeans_pp(tokens, n_clusters, seed):
    rng = np.random.default_rng(seed)
    n = tokens.shape[0]
    first = int(rng.integers(0, n))
    indices = [first]
    for _ in range(1, n_clusters):
        cents = tokens[np.asarray(indices)]
        nearest = ((tokens[:, None, :] - cents[None, :, :]) ** 2).sum(axis=-1).min(axis=1)
        total = nearest.sum()
        if total <= 0:
            pick = next(i for i in range(n) if i not in indices)
        else:
            pick = int(rng.choice(n, p=nearest / total))
        indices.append(pick)
    return tokens[np.asarray(indices)]


def _partition_objective(queries, keys, labels):
    metric = queries.T @ queries
    total = 0.0
    for cluster in np.unique(labels):
        member = keys[labels == cluster]
        residual = member - member.mean(axis=0)
        total += np.sum(np.einsum("ij,jk,ik->i", residual, metric, residual))
    return total


def test_interaction_aware_beats_euclidean():
    queries, keys, n_clusters = _anisotropic_keys(0)
    covariance = np.cov(queries.T)
    eigenvalues = np.linalg.eigvalsh(covariance)
    assert eigenvalues[-1] > 10.0 * eigenvalues[-3]

    init = _kmeans_pp(keys, n_clusters, seed=100)
    metric = queries.T @ queries
    euclidean_labels, _, _ = lloyd_cluster(
        keys,
        n_clusters,
        metric=None,
        n_iters=40,
        init_centroids=init,
        seed=0,
    )
    interaction_labels, _, _ = lloyd_cluster(
        keys,
        n_clusters,
        metric=metric,
        n_iters=40,
        init_centroids=init,
        seed=0,
    )
    euclidean_labels = np.asarray(euclidean_labels)
    interaction_labels = np.asarray(interaction_labels)
    assert len(np.unique(euclidean_labels)) == n_clusters
    assert len(np.unique(interaction_labels)) == n_clusters
    objective_euclidean = _partition_objective(queries, keys, euclidean_labels)
    objective_interaction = _partition_objective(queries, keys, interaction_labels)
    assert objective_interaction < objective_euclidean


def test_singleton_metric_equals_qtq():
    rng = np.random.default_rng(1)
    queries = rng.normal(size=(12, 5)).astype(np.float32)
    sizes = np.ones(queries.shape[0], dtype=np.float32)
    built = metric_from_centroids(queries, sizes)
    expected = np.asarray(queries).T @ np.asarray(queries)
    np.testing.assert_allclose(built, expected, rtol=1e-6, atol=1e-6)


def test_singular_metric_is_jittered():
    singular = np.zeros((4, 4))
    prepared, did_jitter = prepare_metric(singular, 4, dtype=np.float64, jitter=METRIC_JITTER)
    assert did_jitter
    np.testing.assert_allclose(prepared, np.eye(4) * METRIC_JITTER, atol=1e-12)
    assert METRIC_JITTER == 1e-6

    asymmetric = np.array([[2.0, 4.0], [0.0, 2.0]])
    prepared, did_jitter = prepare_metric(asymmetric, 2, dtype=np.float64, jitter=METRIC_JITTER)
    assert did_jitter
    expected = np.array([[2.0, 2.0], [2.0, 2.0]]) + METRIC_JITTER * np.eye(2)
    np.testing.assert_allclose(prepared, expected, atol=1e-12)

    identity, did_jitter = prepare_metric(None, 3, dtype=np.float64)
    assert not did_jitter
    np.testing.assert_allclose(identity, np.eye(3))


def test_config_defaults_match_paper_settings():
    config = ICATSConfig()
    assert config.n_query_clusters == 400
    assert config.n_key_clusters == 1000
    assert config.first_sparse_iters == 50
    assert config.later_iters == 2
    assert config.block_size == 128
    assert config.p_min == 0.7
    assert config.tau == 0.999
    assert config.jitter == 1e-6
    hunyuan = ICATSConfig.hunyuan_base()
    assert hunyuan.p_min == 0.7
    assert hunyuan.tau == 0.98
    wan = ICATSConfig.wan_base()
    assert wan.tau == 0.999


def test_later_timestep_accepts_previous_metric():
    rng = np.random.default_rng(2)
    queries = rng.normal(size=(32, 8))
    keys = rng.normal(size=(32, 8))
    first = cluster_qk(
        queries,
        keys,
        n_query_clusters=4,
        n_key_clusters=4,
        n_iters=5,
        seed=0,
    )
    second = cluster_qk(
        queries,
        keys,
        n_query_clusters=4,
        n_key_clusters=4,
        M_K_prev=first.M_K,
        n_iters=3,
        seed=1,
    )
    assert first.M_Q.shape == (8, 8)
    assert second.M_K.shape == (8, 8)
    assert np.isfinite(np.asarray(second.M_Q)).all()
    assert np.asarray(second.query_labels).shape == (32,)
