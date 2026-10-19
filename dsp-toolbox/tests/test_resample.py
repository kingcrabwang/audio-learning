import numpy as np
import pytest
from scipy.signal import firwin, resample_poly

from dsp_toolbox.resample import (
    decimate,
    interpolate,
    design_resample_filter,
)


# 1. 单独检查滤波器设计
@pytest.mark.parametrize(
    "up, down",
    [(1, 2), (3, 1), (2, 3), (3, 4)],
)
def test_filter_coefficients_match_scipy(up, down):
    max_rate = max(up, down)
    numtaps = 20 * max_rate + 1

    actual = design_resample_filter(up, down, numtaps)

    expected = firwin(
        numtaps=numtaps,
        cutoff=1.0 / max_rate,
        window="hamming",
    )

    # 你的设计函数包含插值增益，因此参考系数也乘 up
    expected = expected * up

    np.testing.assert_allclose(
        actual, expected, rtol=0, atol=1e-12
    )


# 2. 多种长度、多种倍数的抽取对拍
@pytest.mark.parametrize("M", [2, 3, 5])
@pytest.mark.parametrize("length", [1, 2, 7, 64, 101, 512])
def test_decimate_matches_scipy(M, length):
    rng = np.random.default_rng(42)
    x = rng.normal(size=length)

    reference_h = firwin(
        numtaps=20 * M + 1,
        cutoff=1.0 / M,
        window="hamming",
    )

    expected = resample_poly(
        x, up=1, down=M, window=reference_h
    )
    actual = decimate(x, M)

    assert actual.shape == expected.shape

    np.testing.assert_allclose(
        actual, expected, rtol=0, atol=1e-10
    )


# 3. 多种长度、多种倍数的内插对拍
@pytest.mark.parametrize("L", [2, 3, 5])
@pytest.mark.parametrize("length", [1, 2, 7, 64, 101, 512])
def test_interpolate_matches_scipy(L, length):
    rng = np.random.default_rng(123)
    x = rng.normal(size=length)

    reference_h = firwin(
        numtaps=20 * L + 1,
        cutoff=1.0 / L,
        window="hamming",
    )

    # resample_poly 内部会乘 up，这里传入未放大的系数
    expected = resample_poly(
        x, up=L, down=1, window=reference_h
    )
    actual = interpolate(x, L)

    assert actual.shape == expected.shape

    np.testing.assert_allclose(
        actual, expected, rtol=0, atol=1e-10
    )


# 4. 倍数为 1，应保持原信号
@pytest.mark.parametrize("convert", [decimate, interpolate])
def test_factor_one_preserves_signal(convert):
    x = np.array([0.3, -1.2, 2.5, 0.0, 0.7])

    actual = convert(x, 1)

    assert actual.shape == x.shape

    np.testing.assert_allclose(
        actual, x, rtol=0, atol=1e-12
    )


# 5. 非法信号必须抛出 ValueError
@pytest.mark.parametrize("convert", [decimate, interpolate])
@pytest.mark.parametrize(
    "bad_x",
    [
        [],                         # 空数组
        5,                          # 标量
        [[1, 2], [3, 4]],            # 二维数组
        [1, np.nan],                # NaN
        [1, np.inf],                # 无穷大
        ["a", "b"],                 # 字符串
        [1 + 2j, 3],                # 当前接口不支持复数
    ],
)
def test_invalid_signal(convert, bad_x):
    with pytest.raises(ValueError):
        convert(bad_x, 2)


# 6. 非法倍数必须抛出 ValueError
@pytest.mark.parametrize("convert", [decimate, interpolate])
@pytest.mark.parametrize(
    "bad_factor",
    [
        0,
        -1,
        2.0,
        1.5,
        True,
        np.bool_(False),
        "2",
        None,
        np.nan,
        np.inf,
    ],
)
def test_invalid_factor(convert, bad_factor):
    with pytest.raises(ValueError):
        convert([1, 2, 3], bad_factor)


# 7. 处理函数不能修改调用者的原始数据
@pytest.mark.parametrize("convert", [decimate, interpolate])
def test_input_is_not_modified(convert):
    x = np.array([2, -1, 4, 0, 3], dtype=np.int16)
    before = x.copy()

    actual = convert(x, 3)

    np.testing.assert_array_equal(x, before)
    assert actual.dtype == np.float64