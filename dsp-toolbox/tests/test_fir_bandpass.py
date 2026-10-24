"""Numerical and input checks for the hand-written FIR bandpass."""

import numpy as np
import pytest
from scipy.signal import firwin

from dsp_toolbox.filters import design_fir_bandpass


@pytest.mark.parametrize("numtaps", [3, 7, 31, 101])
@pytest.mark.parametrize("edges", [(0.001, 0.01), (0.01, 0.1), (0.2, 0.6), (0.49, 0.51), (0.9, 0.99)])
def test_bandpass_matches_scipy(numtaps, edges):
    fs = 16000
    f1, f2 = np.array(edges) * (fs / 2)
    h = design_fir_bandpass(f1, f2, fs, numtaps)
    expected = firwin(
        numtaps, [f1, f2], pass_zero=False,
        window="hamming", scale=True, fs=fs,
    )

    assert h.shape == (numtaps,)
    assert h.dtype == np.dtype("float64")
    assert np.all(np.isfinite(h))
    np.testing.assert_allclose(h, expected, rtol=0, atol=1e-12)


@pytest.mark.parametrize(
    "changes",
    [
        {"fs": 0}, {"fs": -8000}, {"fs": np.nan},
        {"fs": np.inf}, {"fs": True}, {"fs": "8000"},
        {"f1": 0}, {"f1": -100}, {"f1": 2000},
        {"f1": 2500}, {"f1": np.nan}, {"f1": True},
        {"f1": "500"}, {"f1": 500 + 0j},
        {"f2": 0}, {"f2": 4000}, {"f2": 5000},
        {"f2": np.inf}, {"f2": np.bool_(True)}, {"f2": None},
        {"numtaps": 1}, {"numtaps": 32}, {"numtaps": 31.0},
        {"numtaps": True}, {"numtaps": np.nan},
    ],
)
def test_bandpass_rejects_invalid_parameters(changes):
    params = {"f1": 500, "f2": 2000, "fs": 8000, "numtaps": 31}
    params.update(changes)
    with pytest.raises(ValueError):
        design_fir_bandpass(**params)


def test_bandpass_accepts_numpy_scalars():
    h = design_fir_bandpass(
        np.float64(500), np.float64(2000),
        np.int64(8000), np.int64(31),
    )
    expected = firwin(31, [500, 2000], pass_zero=False, window="hamming", fs=8000)
    np.testing.assert_allclose(h, expected, rtol=0, atol=1e-12)
