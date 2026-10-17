import numpy as np
import pytest

from dsp_toolbox import chirp_wave, sine_wave, time_axis


def test_time_axis_uses_sampling_rate():
    t = time_axis(duration=0.0015, sr=1000)

    np.testing.assert_allclose(
        t, [0.0, 0.001], rtol=0, atol=1e-15
    )


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