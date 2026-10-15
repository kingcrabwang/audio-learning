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
