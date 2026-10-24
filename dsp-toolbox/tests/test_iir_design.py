"""Numerical checks for the Butterworth lowpass design learning task."""

import numpy as np
import pytest
from scipy.signal import butter

from dsp_toolbox.filters import design_iir


@pytest.mark.parametrize("order", [1, 2, 4, 6])
@pytest.mark.parametrize("fs, fc", [(8000, 100), (16000, 1000), (16000, 4000), (16000, 7000)])
def test_lowpass_coefficients_match_scipy(order, fs, fc):
    b, a = design_iir(fc, fs, order=order)
    expected_b, expected_a = butter(order, fc, btype="low", fs=fs)

    assert b.shape == a.shape == (order + 1,)
    assert b.dtype == a.dtype == np.dtype("float64")
    assert np.all(np.isfinite(b))
    assert np.all(np.isfinite(a))
    np.testing.assert_allclose(b, expected_b, rtol=0, atol=1e-12)
    np.testing.assert_allclose(a, expected_a, rtol=0, atol=1e-12)


def test_default_design_is_fourth_order():
    b, a = design_iir(1000, 16000)
    expected_b, expected_a = butter(4, 1000, fs=16000)
    np.testing.assert_allclose(b, expected_b, rtol=0, atol=1e-12)
    np.testing.assert_allclose(a, expected_a, rtol=0, atol=1e-12)


def test_numpy_integer_order():
    b, a = design_iir(np.float64(1000), np.int64(16000), order=np.int64(4))
    expected_b, expected_a = butter(4, 1000, fs=16000)
    np.testing.assert_allclose(b, expected_b, rtol=0, atol=1e-12)
    np.testing.assert_allclose(a, expected_a, rtol=0, atol=1e-12)


@pytest.mark.parametrize("order", [0, -1, 1.5, True, np.bool_(True), np.nan, np.inf, "4", None])
def test_invalid_order(order):
    with pytest.raises(ValueError):
        design_iir(1000, 16000, order=order)


@pytest.mark.parametrize("btype", ["high", "bandpass", None, 3])
def test_unsupported_btype(btype):
    with pytest.raises(ValueError):
        design_iir(1000, 16000, btype=btype)


@pytest.mark.parametrize("ftype", ["cheby1", None, 3])
def test_unsupported_ftype(ftype):
    with pytest.raises(ValueError):
        design_iir(1000, 16000, ftype=ftype)
