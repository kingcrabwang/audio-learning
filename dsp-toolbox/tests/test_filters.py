"""Tests for the current odd-length, Hamming/Hann FIR low-pass API."""

import numpy as np
import pytest
from scipy.signal import firwin

from dsp_toolbox.filters import design_fir_lowpass


@pytest.mark.parametrize("window", ["hamming", "hann"])
@pytest.mark.parametrize("numtaps", [3, 7, 31, 101])
@pytest.mark.parametrize("nyquist_fraction", [0.005, 0.1, 0.25, 0.75, 0.99])
def test_lowpass_coefficients_match_scipy(window, numtaps, nyquist_fraction):
    fs = 16000
    fc = nyquist_fraction * fs / 2

    actual = design_fir_lowpass(fc, fs, numtaps, window=window)
    expected = firwin(
        numtaps, fc, fs=fs, window=window, pass_zero=True, scale=True
    )

    assert actual.shape == (numtaps,)
    assert actual.dtype == np.dtype(np.float64)
    assert np.all(np.isfinite(actual))
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)
    np.testing.assert_allclose(actual, actual[::-1], rtol=0, atol=1e-14)
    np.testing.assert_allclose(np.sum(actual), 1.0, rtol=0, atol=1e-14)


def test_default_window_matches_hamming_reference():
    actual = design_fir_lowpass(2300, 16000, 51)
    expected = firwin(51, 2300, fs=16000, window="hamming", scale=True)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)


@pytest.mark.parametrize("window", ["hamming", "hann"])
def test_scaling_frequency_units_preserves_coefficients(window):
    first = design_fir_lowpass(1000, 8000, 31, window=window)
    second = design_fir_lowpass(6000, 48000, 31, window=window)
    np.testing.assert_allclose(first, second, rtol=0, atol=1e-14)


def test_numpy_scalar_parameters_are_accepted():
    actual = design_fir_lowpass(
        np.float64(2300), np.int64(16000), np.int64(51), window="hann"
    )
    expected = firwin(51, 2300, fs=16000, window="hann", scale=True)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)


# Even numtaps are deliberately rejected by this learning implementation.
# This is a project restriction, not a restriction on all FIR filters.
BAD_PARAMETERS = (
    [("fs", value) for value in (
        0, -8000, np.nan, np.inf, -np.inf, True, np.bool_(False),
        "16000", None, 16000 + 0j, [16000],
    )]
    + [("fc", value) for value in (
        0, -1, 8000, 9000, np.nan, np.inf, -np.inf,
        True, np.bool_(True), "2300", None, 2300 + 0j, [2300],
    )]
    + [("numtaps", value) for value in (
        -3, 0, 1, 2, 4, 31.0, np.float64(31), np.nan, np.inf,
        True, np.bool_(True), "31", None, [31],
    )]
    + [("window", value) for value in (
        "", "blackman", "HAMMING", None, 1, ["hann"],
    )]
)


@pytest.mark.parametrize("name,value", BAD_PARAMETERS)
def test_invalid_parameters_raise_value_error(name, value):
    params = {"fc": 2300, "fs": 16000, "numtaps": 31, "window": "hamming"}
    params[name] = value

    with pytest.raises(ValueError):
        design_fir_lowpass(**params)
