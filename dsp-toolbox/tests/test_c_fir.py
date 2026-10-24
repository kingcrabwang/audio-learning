"""通过 ctypes 调用 C FIR；Python 读取同一份 golden 数据后传给 C。"""

import ctypes
import os
from pathlib import Path

import numpy as np
import pytest


FLOAT_POINTER = ctypes.POINTER(ctypes.c_float)
GOLDEN_DIR = Path(__file__).resolve().parents[1] / "golden" / "fir"


def pointer(array):
    return array.ctypes.data_as(FLOAT_POINTER)


@pytest.fixture(scope="session")
def library():
    path = os.environ.get("DSP_FIR_LIB")
    if not path:
        pytest.skip("C FIR 通过 ctest 单独验收；CMake 会传入 DSP_FIR_LIB。")
    lib = ctypes.CDLL(path)
    lib.fir_create.argtypes = [FLOAT_POINTER, ctypes.c_int]
    lib.fir_create.restype = ctypes.c_void_p
    lib.fir_process.argtypes = [
        ctypes.c_void_p, FLOAT_POINTER, FLOAT_POINTER, ctypes.c_int,
    ]
    lib.fir_process.restype = None
    lib.fir_destroy.argtypes = [ctypes.c_void_p]
    lib.fir_destroy.restype = None
    return lib


def load_case(numtaps):
    arrays = []
    for prefix in ("h", "x", "y"):
        path = GOLDEN_DIR / f"{prefix}_{numtaps}.npy"
        if not path.exists():
            pytest.fail("缺少 golden 数据，请先运行 python tests/make_fir_golden.py。")
        arrays.append(np.load(path, allow_pickle=False))
    return arrays


def run_filter(lib, h, x, block_size):
    h = np.ascontiguousarray(h, dtype=np.float32)
    x = np.ascontiguousarray(x, dtype=np.float32)
    output = np.empty_like(x)
    state = lib.fir_create(pointer(h), len(h))
    assert state, "fir_create 创建失败"
    try:
        for start in range(0, len(x), block_size):
            end = min(start + block_size, len(x))
            lib.fir_process(
                state, pointer(x[start:end]), pointer(output[start:end]), end - start,
            )
    finally:
        lib.fir_destroy(state)
    return output


@pytest.mark.parametrize("numtaps", [1, 3, 17, 101])
@pytest.mark.parametrize("block_size", [1, 7, 64, 512])
def test_c_fir_reference_and_block_partition(library, numtaps, block_size):
    h, x, expected = load_case(numtaps)
    actual = run_filter(library, h, x, block_size)
    whole = run_filter(library, h, x, len(x))
    error = np.max(np.abs(actual.astype(np.float64) - expected))
    assert np.isfinite(error) and error < 1e-6, f"最大逐点误差：{error}"
    np.testing.assert_array_equal(actual, whole)


def test_c_fir_copies_coefficients(library):
    h, x, expected = load_case(17)
    output = np.empty_like(x)
    state = library.fir_create(pointer(h), len(h))
    assert state
    try:
        h[:] = 0
        library.fir_process(state, pointer(x), pointer(output), len(x))
    finally:
        library.fir_destroy(state)
    np.testing.assert_allclose(output, expected, rtol=0, atol=1e-6)


def test_c_fir_allows_in_place_processing(library):
    h, x, expected = load_case(3)
    state = library.fir_create(pointer(h), len(h))
    assert state
    try:
        library.fir_process(state, pointer(x), pointer(x), len(x))
    finally:
        library.fir_destroy(state)
    np.testing.assert_allclose(x, expected, rtol=0, atol=1e-6)


def test_c_fir_instances_have_independent_state(library):
    h, x, expected = load_case(3)
    first = library.fir_create(pointer(h), len(h))
    second = library.fir_create(pointer(h), len(h))
    try:
        assert first and second
        discarded = np.empty_like(x)
        actual = np.empty_like(x)
        library.fir_process(first, pointer(x), pointer(discarded), len(x))
        library.fir_process(second, pointer(x), pointer(actual), len(x))
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-6)
    finally:
        library.fir_destroy(first)
        library.fir_destroy(second)


def test_c_fir_rejects_invalid_creation(library):
    h = np.array([1.0], dtype=np.float32)
    assert not library.fir_create(FLOAT_POINTER(), 1)
    assert not library.fir_create(pointer(h), 0)
    assert not library.fir_create(pointer(h), -1)


def test_c_fir_empty_and_null_calls_keep_state(library):
    h, x, expected = load_case(3)
    output = np.empty_like(x)
    state = library.fir_create(pointer(h), len(h))
    assert state
    try:
        library.fir_process(state, pointer(x), pointer(output), 0)
        library.fir_process(state, pointer(x), pointer(output), -1)
        library.fir_process(state, FLOAT_POINTER(), pointer(output), 1)
        library.fir_process(state, pointer(x), FLOAT_POINTER(), 1)
        library.fir_process(None, pointer(x), pointer(output), 1)
        library.fir_process(state, pointer(x), pointer(output), len(x))
    finally:
        library.fir_destroy(state)
    library.fir_destroy(None)
    np.testing.assert_allclose(output, expected, rtol=0, atol=1e-6)
