"""Streaming FIR: reference accuracy, block invariance and independent state."""

import numpy as np
import pytest
from scipy.signal import lfilter

from dsp_toolbox.filters import FirFilter


def process_in_blocks(h, x, block_size):
    fir = FirFilter(h)
    return np.concatenate([
        fir.process(x[start:start + block_size])
        for start in range(0, len(x), block_size)
    ])


@pytest.mark.parametrize("numtaps", [1, 3, 17, 101])
@pytest.mark.parametrize("block_size", [1, 7, 64, 512])
def test_streaming_output_matches_scipy(numtaps, block_size):
    rng = np.random.default_rng(2026 + numtaps)
    # Non-symmetric coefficients help expose reversed coefficient order.
    h = rng.standard_normal(numtaps) / numtaps
    x = rng.standard_normal(1103)
    actual = process_in_blocks(h, x, block_size)
    expected = lfilter(h, [1.0], x)

    assert actual.shape == x.shape
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-10)


@pytest.mark.parametrize("numtaps", [1, 3, 17, 101])
@pytest.mark.parametrize("block_size", [1, 7, 64, 512])
def test_block_partition_does_not_change_any_output(numtaps, block_size):
    rng = np.random.default_rng(73 + numtaps)
    h = rng.standard_normal(numtaps) / numtaps
    x = rng.standard_normal(1103)
    expected = FirFilter(h).process(x)
    actual = process_in_blocks(h, x, block_size)

    np.testing.assert_array_equal(actual, expected)


def test_short_blocks_retain_history_from_multiple_calls():
    h = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
    blocks = [[2, 5, 1], [7, 3, 8], [4, 6]]
    fir = FirFilter(h)
    actual = np.concatenate([fir.process(block) for block in blocks])
    expected = lfilter(h, [1.0], np.concatenate(blocks))
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)


def test_filter_instances_have_independent_state():
    h = [0.25, 0.5, 0.25]
    first = FirFilter(h)
    second = FirFilter(h)
    first.process([1, 2, 3, 4])
    expected = lfilter(h, [1.0], [5, 6])
    np.testing.assert_allclose(second.process([5, 6]), expected, rtol=0, atol=1e-12)


def test_single_coefficient_filter():
    fir = FirFilter([0.5])
    np.testing.assert_array_equal(fir.process([2, -4, 6]), [1, -2, 3])


def test_empty_block_keeps_state():
    h = [0.25, 0.5, 0.25]
    fir = FirFilter(h)
    first = fir.process([1, 2])
    empty = fir.process([])
    second = fir.process([3, 4])

    assert empty.shape == (0,)
    np.testing.assert_allclose(
        np.concatenate([first, second]), lfilter(h, [1.0], [1, 2, 3, 4]),
        rtol=0, atol=1e-12,
    )
