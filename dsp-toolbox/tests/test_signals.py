import numpy as np
import pytest

from dsp_toolbox import chirp_wave, sine_wave, time_axis


def test_time_axis_uses_sampling_rate():
    t = time_axis(duration=0.0015, sr=1000)

    np.testing.assert_allclose(
        t, [0.0, 0.001], rtol=0, atol=1e-15
    )


def test_time_axis_accepts_numpy_integer_sampling_rate():
    actual = time_axis(duration=0.002, sr=np.int64(1000))
    np.testing.assert_allclose(actual, [0.0, 0.001], rtol=0, atol=1e-15)


def test_sine_matches_sample_times():
    sr = 1000
    duration = 0.0105
    freq = 100
    t = np.arange(round(sr * duration)) / sr

    expected = 0.5 * np.sin(2 * np.pi * freq * t)
    actual = sine_wave(freq, sr, duration)

    np.testing.assert_allclose(
        actual, expected, rtol=0, atol=1e-12
    )


@pytest.mark.parametrize(
    "duration,sr",
    [
        (0, 1000),
        (-1, 1000),
        (1, 0),
        (1, -1000),
        (1, 1000.5),
        (1, 1000.0),
        (1, True),
        (1, np.bool_(True)),
        (float("nan"), 1000),
        (0.0001, 1000),
    ],
)
def test_invalid_time_axis(duration, sr):
    with pytest.raises(ValueError):
        time_axis(duration, sr)


def test_chirp_rejects_zero_duration():
    with pytest.raises(ValueError):
        chirp_wave(100, 200, sr=1000, duration=0)


@pytest.mark.parametrize(
    "freq,amplitude",
    [(float("nan"), 0.5), (100, float("inf"))],
)
def test_sine_rejects_nonfinite_parameters(freq, amplitude):
    with pytest.raises(ValueError):
        sine_wave(freq, sr=1000, duration=0.01, amplitude=amplitude)


def test_chirp_rejects_nonfinite_frequency():
    with pytest.raises(ValueError):
        chirp_wave(100, float("inf"), sr=1000, duration=0.01)
