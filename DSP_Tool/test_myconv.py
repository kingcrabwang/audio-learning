# test_myconv.py
"""
myconv 的对照测试：以 numpy.convolve 作为标准答案。

两种运行方式都可以：
    python test_myconv.py     # 直接运行，有失败时退出码为 1
    pytest test_myconv.py     # 交给 pytest 收集运行

为什么写成这个样子，见文件末尾的说明。
"""
import os
import sys

import numpy as np

# 把本文件所在目录加入模块搜索路径。
# 这样无论你在哪个目录下敲 python，都能找到同目录的 myconv.py。
# （等以后习惯了包导入，这几行可以删掉，改用 from DSP_Tool.myconv import myconv）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from myconv import (  # noqa: E402  必须在调整 sys.path 之后再导入
    conv_full_reference,
    conv_full_vectorized,
    myconv,
)

MODES = ("full", "same", "valid")
RNG_SEED = 0          # 固定随机种子，保证每次跑出来的用例完全一样
N_RANDOM_CASES = 100


def test_basic_cases():
    """基础样例。

    这里刻意用【非对称】核 [1, 2]。
    上一版用 [1, 1] 这种对称核，而对称核满足 h[::-1] == h，
    所以就算卷积核翻转写反了也照样通过——测试等于没测。
    """
    sig = [1, 2, 3]
    ker = [1, 2]
    for mode in MODES:
        got = myconv(sig, ker, mode=mode)
        expect = np.convolve(sig, ker, mode=mode)
        assert got.shape == expect.shape, (
            f"mode={mode} 输出长度不对：myconv 得到 {got.shape}，numpy 是 {expect.shape}"
        )
        assert np.allclose(got, expect), (
            f"mode={mode} 数值不对：\nmyconv={got}\nnumpy ={expect}"
        )


def test_kernel_longer_than_signal():
    """卷积核比信号还长的情况。

    valid / same 在这种情况下的长度语义最容易写错：
    如果照搬 full_res[M-1:N] 这种写法，核比信号长时会直接返回空数组。
    """
    sig = [1, 2]
    ker = [1, 2, 3]
    for mode in MODES:
        got = myconv(sig, ker, mode=mode)
        expect = np.convolve(sig, ker, mode=mode)
        assert got.shape == expect.shape, (
            f"mode={mode} 输出长度不对：myconv 得到 {got.shape}，numpy 是 {expect.shape}"
        )
        assert np.allclose(got, expect), (
            f"mode={mode} 数值不对：\nmyconv={got}\nnumpy ={expect}"
        )
def test_length_one_inputs():
    """长度为 1 的信号或核——最小合法输入。

    h = [k] 时卷积退化为标量乘法 y = k*x，
    x = [k] 时卷积退化为 k*h。
    这种情况下 valid 应返回与 full 相同的结果（min(N,M)-1 == 0）。
    """
    cases = [
        ([3, 5, 7], [2]),       # 核长度 1
        ([2], [3, 5, 7]),       # 信号长度 1
        ([5], [3]),             # 双方都是 1
    ]
    for sig, ker in cases:
        for mode in MODES:
            got = myconv(sig, ker, mode=mode)
            expect = np.convolve(sig, ker, mode=mode)
            assert got.shape == expect.shape, (
                f"sig={sig}, ker={ker}, mode={mode}: "
                f"shape {got.shape} != {expect.shape}"
            )
            assert np.allclose(got, expect), (
                f"sig={sig}, ker={ker}, mode={mode}: "
                f"myconv={got}, numpy={expect}"
            )


def test_equal_length():
    """信号和核等长时，valid 模式只输出 1 个元素。

    这是 valid 切片最容易出 off-by-one 错误的场景：
    full_res[N-1 : N] 只取 1 个元素，写成 full_res[N-1:N-1] 就是空的。
    """
    for length in [1, 2, 3, 5, 8]:
        sig = list(range(1, length + 1))
        ker = list(range(1, length + 1))
        for mode in MODES:
            got = myconv(sig, ker, mode=mode)
            expect = np.convolve(sig, ker, mode=mode)
            assert got.shape == expect.shape, (
                f"equal length={length}, mode={mode}: "
                f"shape {got.shape} != {expect.shape}"
            )
            assert np.allclose(got, expect), (
                f"equal length={length}, mode={mode}: "
                f"myconv={got}, numpy={expect}"
            )


def test_impulse_properties():
    """利用卷积的数学性质做边界验证。

    1. x * δ[n] = x             （与单位脉冲卷积 = 恒等）
    2. x * δ[n-k] = x 右移 k 位  （延迟脉冲 = 移位）
    3. x * 0 = 0                 （与零卷积 = 零）
    """
    sig = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    # 性质 1：与 δ 卷积 = 恒等
    delta = np.array([1.0])
    result = myconv(sig, delta, mode="full")
    assert np.array_equal(result, sig), f"δ 卷积失败: {result}"

    # 性质 2：与延迟脉冲卷积 = 右移
    delay1 = np.array([0.0, 1.0])
    result = myconv(sig, delay1, mode="full")
    expect = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    assert np.array_equal(result, expect), f"延迟卷积失败: {result}"

    # 性质 3：与零卷积 = 零
    zeros = np.array([0.0, 0.0, 0.0])
    result = myconv(sig, zeros, mode="full")
    assert np.all(result == 0.0), f"零卷积非零: {result}"


def test_extreme_float_values():
    """极端浮点数值的行为。

    不要求特定结果，但要求行为与 numpy.convolve 一致。
    """
    cases = [
        ("大数", [1e200, 2e200], [3.0]),
        ("小数", [1e-200, 2e-200], [3e-200]),
        ("正负混合", [1e100, -1e100], [1.0, 1.0]),
    ]
    for name, sig, ker in cases:
        for mode in MODES:
            got = myconv(sig, ker, mode=mode)
            expect = np.convolve(sig, ker, mode=mode)
            # 用 array_equal 而不是 allclose，因为 allclose 对 inf 等行为不同
            assert np.array_equal(got, expect) or np.allclose(
                got, expect, equal_nan=True
            ), f"{name}, mode={mode}: myconv={got}, numpy={expect}"


def test_commutative_property():
    """卷积交换律：x * h == h * x。

    这不是 numpy 的定义保证的（numpy.convolve 也保证了），
    但如果你的实现打破了交换律，说明有 bug。
    """
    rng = np.random.default_rng(42)
    for _ in range(50):
        n = int(rng.integers(1, 20))
        m = int(rng.integers(1, 20))
        sig = rng.uniform(-10, 10, size=n)
        ker = rng.uniform(-10, 10, size=m)
        for mode in MODES:
            forward = myconv(sig, ker, mode=mode)
            reverse = myconv(ker, sig, mode=mode)
            assert np.allclose(forward, reverse), (
                f"交换律失败: mode={mode}, len(sig)={n}, len(ker)={m}\n"
                f"x*h={forward}\nh*x={reverse}"
            )



def test_random_cases():
    """随机模糊测试：随机长度、随机取值，覆盖各种组合。"""
    rng = np.random.default_rng(RNG_SEED)
    for _ in range(N_RANDOM_CASES):
        n = int(rng.integers(1, 16))
        m = int(rng.integers(1, 16))
        sig = rng.uniform(-10, 10, size=n)
        ker = rng.uniform(-10, 10, size=m)
        for mode in MODES:
            got = myconv(sig, ker, mode=mode)
            expect = np.convolve(sig, ker, mode=mode)
            assert got.shape == expect.shape, (
                f"mode={mode} 输出长度不对：len(sig)={n}, len(ker)={m}, "
                f"myconv={got.shape}, numpy={expect.shape}"
            )
            assert np.allclose(got, expect), (
                f"mode={mode} 数值不对：len(sig)={n}, len(ker)={m}\n"
                f"myconv={got}\nnumpy ={expect}"
            )


def test_vectorized_matches_reference():
    """向量化实现必须和双重循环参考实现【逐位】相同。

    为什么这条测试值得单独占一个位置：
    浮点加法不满足结合律——(a+b)+c 和 a+(b+c) 可能差最后几位。
    移位累加之所以还能逐位相同，是因为对某个输出点而言，
    各项仍然是按 m 递增的顺序累加进去的，和参考实现一模一样，
    而且两者都从 0.0 起算。求和顺序没变，结果才没变。

    这条测试就是在钉死这个前提：以后谁改了累加顺序
    （比如把外层循环改成倒序），它会立刻报错。

    这里刻意用 np.array_equal 而不是 np.allclose：
    allclose 会容忍末位误差，而"顺序被改动"恰恰只体现为末位误差，
    用 allclose 就正好把要测的东西漏掉了。
    """
    rng = np.random.default_rng(RNG_SEED)
    for _ in range(N_RANDOM_CASES):
        n = int(rng.integers(1, 16))
        m = int(rng.integers(1, 16))
        sig = rng.uniform(-10, 10, size=n)
        ker = rng.uniform(-10, 10, size=m)
        got = conv_full_vectorized(sig, ker)
        expect = conv_full_reference(sig, ker)
        assert np.array_equal(got, expect), (
            f"两个 full 实现结果不同：len(sig)={n}, len(ker)={m}\n"
            f"vectorized={got}\nreference ={expect}"
        )


def test_invalid_input():
    """非法输入必须报 ValueError。

    这里只关心"报的错对不对"，不关心数值，
    所以用 try/except 而不是把结果和 numpy 比。
    """
    bad_calls = [
        ("空序列",         lambda: myconv([], [1, 2])),
        ("x 是二维数组",   lambda: myconv([[1, 2], [3, 4]], [1, 2])),
        ("h 是标量",       lambda: myconv([1, 2], 5)),
        ("mode 写成大写",  lambda: myconv([1, 2, 3], [1, 2], mode="SAME")),
    ]
    for name, call in bad_calls:
        try:
            call()
        except ValueError:
            continue
        raise AssertionError(f"{name}：应当抛出 ValueError，但没有报错")


# 所有测试函数按顺序列在这里，直接运行时依次执行
ALL_TESTS = [
    test_basic_cases,
    test_kernel_longer_than_signal,
    test_random_cases,
    test_vectorized_matches_reference,
    test_invalid_input,
    test_length_one_inputs,
    test_equal_length,
    test_impulse_properties,
    test_extreme_float_values,
    test_commutative_property,
]


if __name__ == "__main__":
    failed = 0
    for fn in ALL_TESTS:
        try:
            fn()
            print(f"[PASS] {fn.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"[FAIL] {fn.__name__}: {exc}")

    print(f"\n共 {len(ALL_TESTS)} 组测试，失败 {failed} 组")

    # 关键：用退出码告诉外面的世界"成功还是失败"。
    # 没有这一行，脚本即使测出问题也会以 0 退出，
    # 以后想挂到 GitHub Actions 或 pre-commit 钩子上就完全失效。
    sys.exit(1 if failed else 0)
