import numpy as np
import pytest

from dsp_toolbox.fft import myfft , myifft


@pytest.mark.parametrize("n", [1, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096])
def test_myfft_matches_numpy(n):
    rng = np.random.default_rng(n)
    x = rng.standard_normal(n) + 1j * rng.standard_normal(n)

    actual = myfft(x)
    expected = np.fft.fft(x)

    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("n", [1, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096])
def test_myifft_matches_numpy(n):
    rng = np.random.default_rng(1000 + n)
    X = rng.standard_normal(n) + 1j * rng.standard_normal(n)

    actual = myifft(X)
    expected = np.fft.ifft(X)

    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)


def test_myfft_impulse():
    x = np.zeros(8, dtype=complex)
    x[0] = 1

    np.testing.assert_allclose(myfft(x), np.ones(8), rtol=0, atol=1e-12)


def test_myfft_constant_signal():
    x = np.full(8, 3.0)
    expected = np.zeros(8, dtype=complex)
    expected[0] = 24

    np.testing.assert_allclose(myfft(x), expected, rtol=0, atol=1e-12)


@pytest.mark.parametrize("transform", [myfft, myifft])
def test_non_power_of_two_raises(transform):
    with pytest.raises(ValueError, match="2 的幂"):
        transform(np.arange(6))