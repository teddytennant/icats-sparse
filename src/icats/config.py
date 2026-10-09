"""Paper defaults for iCATS. Unit tests pass smaller cluster counts and iteration budgets."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ICATSConfig:
    """Defaults follow the paper's video settings, not the unit-test sizes.

    Wan2.1-T2V Base uses p_min = 0.7 and tau = 0.999. Hunyuan Base uses the
    same p_min with tau = 0.98 (see hunyuan_base). Query block size 128 matches
    the FlashInfer query block mentioned with the tail-merge evaluation. The
    key block of 96 is not used here: tail merging operates on query cluster
    sizes. Clustering iterations are 50 on the first sparse timestep and 2
    afterwards.
    """

    n_query_clusters: int = 400
    n_key_clusters: int = 1000
    first_sparse_iters: int = 50
    later_iters: int = 2
    p_min: float = 0.7
    tau: float = 0.999
    block_size: int = 128
    jitter: float = 1e-6

    @classmethod
    def wan_base(cls) -> "ICATSConfig":
        return cls(p_min=0.7, tau=0.999)

    @classmethod
    def hunyuan_base(cls) -> "ICATSConfig":
        return cls(p_min=0.7, tau=0.98)
