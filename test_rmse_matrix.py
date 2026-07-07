"""Unit tests for rmse_matrix (RMSE(bins)) in utils.py.

Reviews are grouped by the three log-quantised bins (delta_t, i,
rmse_bins_lapse), then the sample-weighted RMSE of per-bin mean(y) vs mean(p)
is taken. Expected values are computed by hand.
"""

import pandas as pd
import pytest

from utils import rmse_matrix


def _df(**cols):
    return pd.DataFrame(cols)


def test_calibrated():
    df = _df(elapsed_days=[5, 5], i=[2, 2], rmse_bins_lapse=[0, 0],
             y=[1, 0], p=[0.5, 0.5])
    assert rmse_matrix(df) == pytest.approx(0.0)


def test_uniform_gap():
    df = _df(elapsed_days=[5, 5], i=[2, 2], rmse_bins_lapse=[0, 0],
             y=[1, 1], p=[0.9, 0.9])
    assert rmse_matrix(df) == pytest.approx(0.1)


def test_unweighted_bins():
    df = _df(elapsed_days=[2, 100], i=[2, 5], rmse_bins_lapse=[0, 1],
             y=[1, 0], p=[0.9, 0.2])
    assert rmse_matrix(df) == pytest.approx((0.05 / 2) ** 0.5)


def test_weighted_bins():
    df = _df(elapsed_days=[2, 100], i=[2, 5], rmse_bins_lapse=[0, 1],
             y=[1, 0], p=[0.9, 0.2], weights=[3, 1])
    assert rmse_matrix(df) == pytest.approx(((3 * 0.01 + 0.04) / 4) ** 0.5)


def test_lapse_zero_bin():
    df = _df(elapsed_days=[5, 5], i=[2, 2], rmse_bins_lapse=[0, 3],
             y=[1, 0], p=[1.0, 0.0])
    assert rmse_matrix(df) == pytest.approx(0.0)
