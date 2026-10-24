"""Tests for the current first-order lowpass learning implementation."""

import numpy as np
import pytest
from scipy.signal import butter, lfilter

from dsp_toolbox.filters import design_iir, filter_first_order


DESIGNS = [(8000, 100), (16000, 1000), (16000, 4000), (16000, 7000)]


@pytest.mark.parametrize(
    "parameter, value",
    [
        ("fs", 0),
        ("fs", -16000),
        ("fs", np.nan),
        ("fs", np.inf),
        ("fs", -np.inf),
        ("fs", True),
        ("fs", np.bool_(True)),
        ("fs", "16000"),
        ("fs", None),
        ("fs", 16000 + 0j),
        ("fc", 0),
        ("fc", -1),
        ("fc", 8000),
        ("fc", 9000),
        ("fc", np.nan),
        ("fc", np.inf),
        ("fc", -np.inf),
        ("fc", True),
        ("fc", np.bool_(True)),
        ("fc", "1000"),
        ("fc", None),
        ("fc", 1000 + 0j),
    ],
)
def test_first_order_rejects_invalid_parameters(parameter, value):
    params = {"fc": 1000, "fs": 16000, "order": 1}
    params[parameter] = value
    with pytest.raises(ValueError):
        design_iir(**params)


@pytest.mark.parametrize("fs, fc", DESIGNS)
def test_first_order_coefficients_match_scipy(fs, fc):
    b, a = design_iir(fc=fc, fs=fs, order=1)
    expected_b, expected_a = butter(1, fc, btype="low", fs=fs)

    assert np.shape(b) == (2,)
    assert np.shape(a) == (2,)
    np.testing.assert_allclose(b, expected_b, rtol=0, atol=1e-12)
    np.testing.assert_allclose(a, expected_a, rtol=0, atol=1e-12)


@pytest.mark.parametrize("fs, fc", DESIGNS)
@pytest.mark.parametrize("kind", ["single", "impulse", "constant", "random"])
def test_first_order_output_matches_scipy(fs, fc, kind):
    if kind == "single":
        x = np.array([0.75])
    elif kind == "impulse":
        x = np.zeros(256)
        x[0] = 1.0
    elif kind == "constant":
        x = np.ones(512)
    else:
        x = np.random.default_rng(2026).standard_normal(1024)

    b, a = design_iir(fc=fc, fs=fs, order=1)
    actual = filter_first_order(x, b, a)
    expected_b, expected_a = butter(1, fc, btype="low", fs=fs)
    expected = lfilter(expected_b, expected_a, x)

    assert np.shape(actual) == x.shape
    assert np.all(np.isfinite(actual))
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-10)
