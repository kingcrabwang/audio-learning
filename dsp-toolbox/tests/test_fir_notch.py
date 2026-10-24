"""Checks for the window-method FIR notch and its nominal bandwidth."""

import numpy as np
import pytest
from scipy.signal import firwin

from dsp_toolbox.filters import design_fir_notch


@pytest.mark.parametrize(
    "f0, fs, q",
    [(50, 1000, 5), (50, 16000, 5), (1000, 8000, 5),
     (2500, 16000, 4), (1800, 8000, 3)],
)
def test_notch_matches_scipy_bandstop(f0, fs, q):
    h = design_fir_notch(f0, fs, q)
    bandwidth = f0 / q
    expected = firwin(
        len(h), [f0 - bandwidth / 2, f0 + bandwidth / 2],
        pass_zero="bandstop", window="hamming", scale=True, fs=fs,
    )

    assert h.ndim == 1 and len(h) >= 3 and len(h) % 2 == 1
    assert h.dtype == np.dtype("float64")
    assert np.all(np.isfinite(h))
    np.testing.assert_allclose(h, expected, rtol=0, atol=1e-12)
    np.testing.assert_allclose(h, h[::-1], rtol=0, atol=1e-14)
    np.testing.assert_allclose(np.sum(h), 1.0, rtol=0, atol=1e-12)

    response_at_center = np.sum(
        h * np.exp(-2j * np.pi * f0 * np.arange(len(h)) / fs)
    )
    assert abs(response_at_center) < 0.02


@pytest.mark.parametrize(
    "changes",
    [
        {"fs": 0}, {"fs": -8000}, {"fs": np.nan},
        {"fs": np.inf}, {"fs": True}, {"fs": "8000"},
        {"f0": 0}, {"f0": -50}, {"f0": 4000},
        {"f0": 5000}, {"f0": np.nan}, {"f0": np.inf},
        {"f0": True}, {"f0": None}, {"f0": 1000 + 0j},
        {"q": 0}, {"q": -5}, {"q": 0.5}, {"q": 0.25},
        {"q": np.nan}, {"q": np.inf}, {"q": True},
        {"q": np.bool_(True)}, {"q": "5"}, {"q": None},
        {"f0": 3900, "q": 5},
    ],
)
def test_notch_rejects_invalid_parameters(changes):
    params = {"f0": 1000, "fs": 8000, "q": 5}
    params.update(changes)
    with pytest.raises(ValueError):
        design_fir_notch(**params)


def test_notch_accepts_numpy_scalars():
    h = design_fir_notch(np.float64(1000), np.int64(8000), np.float64(5))
    expected = firwin(
        len(h), [900, 1100], pass_zero="bandstop", window="hamming", fs=8000,
    )
    np.testing.assert_allclose(h, expected, rtol=0, atol=1e-12)
