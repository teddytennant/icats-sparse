"""SNR pivot, top-p calibration, and the dense region before t_sparse."""

import numpy as np
import pytest

from icats.snr import pivot_index, synthetic_noise_schedule, synthetic_snr_schedule


def test_snr_top_p_calibration_and_pivot():
    num_steps = 50
    timesteps, a, b = synthetic_noise_schedule(num_steps)
    ell = np.log((np.asarray(a) / np.asarray(b)) ** 2)
    gaps = np.abs(np.diff(ell))
    expected_pivot = int(np.argmin(gaps)) + 1
    assert pivot_index(ell) == expected_pivot

    # t_sparse is on the noisier side of the pivot, as in the paper, so the
    # absolute-value calibration makes the sigmoid factor equal tau.
    t_sparse = 40
    assert int(np.asarray(timesteps)[expected_pivot]) < t_sparse
    for p_min, tau in ((0.7, 0.999), (0.7, 0.98)):
        schedule = synthetic_snr_schedule(
            num_steps, t_sparse=t_sparse, p_min=p_min, tau=tau
        )
        assert schedule.pivot_index == expected_pivot
        sparse_index = int(np.where(np.asarray(schedule.timesteps) == t_sparse)[0][0])
        x = float(schedule.x[sparse_index])
        k = float(schedule.k)
        factor = 1.0 - 1.0 / (1.0 + np.exp(-k * x))
        assert factor == pytest.approx(tau, rel=1e-6, abs=1e-6)
        expected_top_p = p_min + (1.0 - p_min) * tau
        assert float(schedule.top_p[sparse_index]) == pytest.approx(expected_top_p)
        assert float(schedule.factor[sparse_index]) == pytest.approx(tau)

        before = np.asarray(schedule.timesteps) > t_sparse
        assert np.any(before)
        np.testing.assert_allclose(np.asarray(schedule.top_p)[before], 1.0)
        assert np.all(np.asarray(schedule.dense)[before])
        assert not bool(np.asarray(schedule.dense)[sparse_index])

        top_p = np.asarray(schedule.top_p, dtype=np.float64)
        min_p = np.asarray(schedule.min_p, dtype=np.float64)
        np.testing.assert_allclose(min_p, 0.1 * top_p / top_p[schedule.pivot_index])
