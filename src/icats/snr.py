"""SNR-guided top-p schedule (section 4.2, equation 3).

SNR(t) = (a_t / b_t)^2 and ell_t = log(SNR(t)). The pivot is the timestep
immediately after the minimum absolute gap between consecutive ell values.
If that minimum gap is between index i and i+1, the pivot index is i+1.
Ties take the earliest gap (the first minimum).

    x_t = ell_t - ell_pivot
    k = log(tau / (1 - tau)) / |ell_{t_sparse} - ell_pivot|
    top-p_t = p_min + (1 - p_min) * (1 - 1 / (1 + exp(-k x_t)))  for t <= t_sparse

Before the sparse regime (t > t_sparse) attention is dense and top-p is 1.
That matches the paper's decreasing-timestep denoising order: warmup steps
have larger t than the first sparse timestep, and the formula applies for
t <= t_sparse. The factor equals tau at t_sparse when ell_{t_sparse} <=
ell_pivot, which is the paper's setting (sparsity starts on the noisier side
of the pivot). On the other side of the pivot the same k yields 1 - tau.

min-p_t = 0.1 * top-p_t / top-p_pivot. This is a minimum fraction of key
clusters to retain, not a probability mass.

The paper uses Wan's noise schedule, which is not available here. Tests and
the demo use the synthetic schedule a_t = 1 - t/T, b_t = t/T for integer
t = 1 .. T-1 so that b_t > 0. Larger t is noisier. Defaults are the Wan2.1-T2V
Base pair p_min = 0.7, tau = 0.999. The Hunyuan Base pair p_min = 0.7,
tau = 0.98 is accepted through the same arguments.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp


@dataclass
class SNRSchedule:
    timesteps: jnp.ndarray
    a: jnp.ndarray
    b: jnp.ndarray
    snr: jnp.ndarray
    ell: jnp.ndarray
    pivot_index: int
    pivot_t: jnp.ndarray
    t_sparse: int
    k: jnp.ndarray
    x: jnp.ndarray
    factor: jnp.ndarray
    top_p: jnp.ndarray
    min_p: jnp.ndarray
    dense: jnp.ndarray
    p_min: float
    tau: float


def synthetic_noise_schedule(num_steps):
    """Return timesteps, a_t, b_t for t = 1 .. T-1.

    a_t = 1 - t/T and b_t = t/T. This is a synthetic stand-in for Wan's
    schedule, documented because the formula tests must not depend on Wan.
    """
    num_steps = int(num_steps)
    if num_steps < 3:
        raise ValueError("num_steps must be at least 3 so the schedule has a gap")
    timesteps = jnp.arange(1, num_steps)
    width = jnp.asarray(num_steps, dtype=jnp.float32)
    t = timesteps.astype(jnp.float32)
    a = 1.0 - t / width
    b = t / width
    return timesteps, a, b


def log_snr(a, b):
    """ell_t = log(SNR(t)) with SNR(t) = (a_t / b_t)^2. Natural log."""
    a = jnp.asarray(a)
    b = jnp.asarray(b)
    snr = (a / b) ** 2
    return jnp.log(snr), snr


def pivot_index(ell):
    """Index of the timestep immediately after the minimum consecutive |delta ell|.

    If the minimum gap is between index i and i+1, the pivot index is i+1.
    Equal gaps resolve to the smallest i.
    """
    ell = jnp.asarray(ell)
    if ell.shape[0] < 2:
        raise ValueError("ell must contain at least two timesteps")
    gaps = jnp.abs(jnp.diff(ell))
    return int(jnp.argmin(gaps)) + 1


def snr_top_p_schedule(a, b, timesteps, t_sparse, p_min=0.7, tau=0.999):
    """Build top-p and min-p for every timestep in the schedule."""
    if not 0.0 < float(tau) < 1.0:
        raise ValueError("tau must lie in (0, 1)")
    if not 0.0 < float(p_min) <= 1.0:
        raise ValueError("p_min must lie in (0, 1]")
    timesteps = jnp.asarray(timesteps)
    ell, snr = log_snr(a, b)
    if ell.shape != timesteps.shape:
        raise ValueError("a, b, and timesteps must have the same length")
    pidx = pivot_index(ell)
    matches = jnp.where(timesteps == int(t_sparse))[0]
    if int(matches.size) == 0:
        raise ValueError("t_sparse is not in timesteps")
    sidx = int(matches[0])
    gap = jnp.abs(ell[sidx] - ell[pidx])
    if float(gap) == 0.0:
        raise ValueError("t_sparse log-SNR equals the pivot, so k is undefined")
    k = jnp.log(jnp.asarray(tau, dtype=ell.dtype) / (1.0 - tau)) / gap
    x = ell - ell[pidx]
    factor = 1.0 - 1.0 / (1.0 + jnp.exp(-k * x))
    top_p = float(p_min) + (1.0 - float(p_min)) * factor
    dense = timesteps > int(t_sparse)
    top_p = jnp.where(dense, jnp.ones_like(top_p), top_p)
    min_p = 0.1 * top_p / top_p[pidx]
    return SNRSchedule(
        timesteps=timesteps,
        a=jnp.asarray(a),
        b=jnp.asarray(b),
        snr=snr,
        ell=ell,
        pivot_index=pidx,
        pivot_t=timesteps[pidx],
        t_sparse=int(t_sparse),
        k=k,
        x=x,
        factor=factor,
        top_p=top_p,
        min_p=min_p,
        dense=dense,
        p_min=float(p_min),
        tau=float(tau),
    )


def synthetic_snr_schedule(num_steps, t_sparse, p_min=0.7, tau=0.999):
    """SNR top-p schedule on the synthetic a_t, b_t noise schedule."""
    timesteps, a, b = synthetic_noise_schedule(num_steps)
    return snr_top_p_schedule(a, b, timesteps, t_sparse, p_min=p_min, tau=tau)
