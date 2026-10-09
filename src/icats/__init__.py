"""iCATS sparse-attention components: clustering, SNR top-p, and tail merging."""

from __future__ import annotations

import os

# CPU reference implementation. Respect an explicit JAX_PLATFORMS if one is set.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

from icats.attention import dense_attention, sparse_attention
from icats.clustering import (
    METRIC_JITTER,
    cluster_qk,
    explicit_interaction_residual,
    interaction_objective,
    kmeans_pp_init,
    lloyd_cluster,
    metric_from_centroids,
    prepare_metric,
    quadratic_distances,
)
from icats.config import ICATSConfig
from icats.mask import build_cluster_mask, cluster_softmax_scores, select_key_clusters
from icats.snr import (
    SNRSchedule,
    log_snr,
    pivot_index,
    snr_top_p_schedule,
    synthetic_noise_schedule,
    synthetic_snr_schedule,
)
from icats.tail_merge import padding_block_count, tail_merge

__all__ = [
    "ICATSConfig",
    "METRIC_JITTER",
    "SNRSchedule",
    "build_cluster_mask",
    "cluster_qk",
    "cluster_softmax_scores",
    "dense_attention",
    "explicit_interaction_residual",
    "interaction_objective",
    "kmeans_pp_init",
    "lloyd_cluster",
    "log_snr",
    "metric_from_centroids",
    "padding_block_count",
    "pivot_index",
    "prepare_metric",
    "quadratic_distances",
    "select_key_clusters",
    "snr_top_p_schedule",
    "sparse_attention",
    "synthetic_noise_schedule",
    "synthetic_snr_schedule",
    "tail_merge",
]
