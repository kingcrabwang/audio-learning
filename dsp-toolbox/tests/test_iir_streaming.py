"""Reference accuracy and state checks for the streaming filter classes."""

import numpy as np
import pytest
from scipy.signal import butter, lfilter

from dsp_toolbox.filters import FirFilter, IirFilter


def process_in_blocks(b, a, x, block_size):
    filt = IirFilter(b, a)
    return np.concatenate([
        filt.process(x[start:start + block_size])
        for start in range(0, len(x), block_size)
    ])


@pytest.mark.parametrize("order", [1, 2, 4, 6])
@pytest.mark.parametrize("block_size", [1, 7, 64, 512])
def test_iir_matches_scipy(order, block_size):
    b, a = butter(order, 100, fs=1000)
    x = np.random.default_rng(2026).standard_normal(1103)
    actual = process_in_blocks(b, a, x, block_size)
    np.testing.assert_allclose(actual, lfilter(b, a, x), rtol=0, atol=1e-10)


@pytest.mark.parametrize("order", [1, 2, 4, 6])
@pytest.mark.parametrize("block_size", [1, 7, 64, 512])
def test_iir_block_partition_is_exact(order, block_size):
    b, a = butter(order, 100, fs=1000)
    x = np.random.default_rng(73).standard_normal(1103)
    expected = IirFilter(b, a).process(x)
    actual = process_in_blocks(b, a, x, block_size)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    "b, a",
    [
        ([0.25, 0.5], [1, -0.6, 0.1]),
        ([0.1, 0.2, 0.1, 0.05], [1, -0.5]),
        ([1], [1, -0.6, 0.08]),
        ([0.2, 0.6, 0.2], [1]),
        ([0.5], [1]),
        ([2, 0.5], [2, -0.8]),
    ],
)
def test_unequal_lengths_and_general_a0(b, a):
    x = np.random.default_rng(8).standard_normal(41)
    actual = process_in_blocks(b, a, x, 7)
    np.testing.assert_allclose(actual, lfilter(b, a, x), rtol=0, atol=1e-12)


def test_iir_empty_block_keeps_state():
    b, a = [0.2, 0.2], [1, -0.6]
    filt = IirFilter(b, a)
    first = filt.process([1, 2])
    assert filt.process([]).shape == (0,)
    second = filt.process([3, 4])
    np.testing.assert_allclose(
        np.concatenate([first, second]), lfilter(b, a, [1, 2, 3, 4]),
        rtol=0, atol=1e-12,
    )


def test_iir_instances_have_independent_state():
    b, a = [0.2, 0.2], [1, -0.6]
    first, second = IirFilter(b, a), IirFilter(b, a)
    first.process([1, 2, 3])
    np.testing.assert_allclose(
        second.process([4, 5]), lfilter(b, a, [4, 5]), rtol=0, atol=1e-12,
    )


@pytest.mark.parametrize("invalid", [[], [[1, 2]], [np.nan], [np.inf], [True], [1j], ["1"]])
def test_reject_invalid_coefficients(invalid):
    with pytest.raises(ValueError):
        FirFilter(invalid)
    with pytest.raises(ValueError):
        IirFilter(invalid, [1])
    with pytest.raises(ValueError):
        IirFilter([1], invalid)


def test_zero_a0_is_rejected():
    with pytest.raises(ValueError):
        IirFilter([1], [0, 0.5])


@pytest.mark.parametrize("kind", ["fir", "iir"])
@pytest.mark.parametrize("invalid", [[[1, 2]], [1, np.nan], [np.inf], [True], [1j], ["1"]])
def test_invalid_block_does_not_change_state(kind, invalid):
    b, a = [0.2, 0.2], [1, -0.6]
    if kind == "fir":
        filt, control = FirFilter(b), FirFilter(b)
    else:
        filt, control = IirFilter(b, a), IirFilter(b, a)
    filt.process([1, 2])
    control.process([1, 2])
    with pytest.raises(ValueError):
        filt.process(invalid)
    np.testing.assert_array_equal(filt.process([3, 4]), control.process([3, 4]))


def test_filters_keep_their_own_coefficient_copies():
    b = np.array([0.2, 0.2])
    a = np.array([1.0, -0.6])
    h = np.array([0.2, 0.6, 0.2])
    iir, fir = IirFilter(b, a), FirFilter(h)
    b[:] = a[:] = h[:] = 0
    x = [1, 2, 3]
    np.testing.assert_allclose(iir.process(x), lfilter([0.2, 0.2], [1, -0.6], x))
    np.testing.assert_allclose(fir.process(x), lfilter([0.2, 0.6, 0.2], [1], x))
