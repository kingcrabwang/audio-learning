# myconv.py
"""
DSP_Tool: 手写一维卷积函数 myconv(x, h, mode)

实现一维离散卷积，支持 full / same / valid 三种模式。
不调用 np.convolve / scipy.signal 等现成卷积函数。

语义与 numpy.convolve / MATLAB conv 对齐（N = len(x)，M = len(h)）：
    full  : 输出长度 N + M - 1
    same  : 输出长度 max(N, M)，取 full 结果居中段
    valid : 输出长度 |N - M| + 1

模块里有【两个】full 卷积的实现，结果逐位相同：
    conv_full_vectorized : 移位累加，用 numpy 向量化，快——myconv 默认走这条
    conv_full_reference  : 双重循环，照抄定义，慢——留作对照基准
为什么两个都留、各自的取舍，见这两个函数的 docstring。
"""
import numpy as np

__all__ = ["myconv", "conv_full_reference", "conv_full_vectorized"]

_MODES = ("full", "same", "valid")


def _is_1d(seq):
    """判断输入是否为非空一维数值序列"""
    if isinstance(seq, (str, bytes)) or np.isscalar(seq):
        return False
    try:
        arr = np.asarray(seq)
    except Exception:
        return False
    return arr.ndim == 1 and arr.size > 0 and arr.dtype.kind in "biuf"


def _validate_1d(x, h):
    """校验 x 和 h 都是非空一维数值序列"""
    if not _is_1d(x):
        raise ValueError(
            f"输入序列 x 必须是非空一维数值序列，当前 dtype={np.asarray(x).dtype}, "
            f"shape={np.asarray(x).shape}"
        )
    if not _is_1d(h):
        raise ValueError(
            f"卷积核 h 必须是非空一维数值序列，当前 dtype={np.asarray(h).dtype}, "
            f"shape={np.asarray(h).shape}"
        )


def conv_full_vectorized(x, h):
    """
    full 卷积 —— 移位累加实现（快，myconv 默认用它）

    思路：不要问"第 n 个输出点 y[n] 是多少"，改问
    "第 m 个抽头 h[m] 给整个输出贡献了什么"。

    固定 m 只看一项 h[m] * x[n-m]：当 n 从 m 走到 m+N-1 时，
    x[n-m] 正好走遍 x[0]..x[N-1]。所以这一项的贡献就是
    "整条 x 乘上标量 h[m]，平移到输出位置 [m, m+N-1]"。

    于是循环次数从 N（信号长度）降到 M（核长度）：
        x 长 132300、核长 101 时，Python 层循环从 1300 万次降到 101 次，
        剩下的乘加交给 numpy 的 C 代码批量做。

    注意：乘加总次数和 conv_full_reference 完全一样（都是 N*M 次），
    快的原因不是算得少，而是"让 C 去循环，Python 只发号施令"。

    与 conv_full_reference 逐位相同。原因：浮点加法不满足结合律，
    换求和顺序本来会改变末位结果；但这里对某个输出点而言，
    各项仍然是按 m 递增的顺序累加进去的，两者都从 0.0 起算，
    求和顺序没变，所以结果一模一样。测试里用 np.array_equal 钉死了这一点。

    调用方需保证 x、h 已经是非空一维 float64 数组。
    """
    N, M = x.size, h.size
    full_res = np.zeros(N + M - 1, dtype=np.float64)
    for m in range(M):
        # 第 m 个抽头的贡献：h[m] * x，平移到输出位置 [m, m + N - 1]
        full_res[m : m + N] += h[m] * x
    return full_res


def conv_full_reference(x, h):
    """
    full 卷积 —— 参考实现（慢，但把定义直接摆在眼前）

    严格照抄 y[n] = sum_m x[n-m] * h[m] 的双重循环。
    保留它有两个用处：

    1. 读代码就能看到卷积的定义本身，不用在脑子里反推
       "向量化那几行到底在算什么"。
    2. 以后优化算法时拿它当标准答案做交叉验证
       ——见 test_myconv.py 里的 test_vectorized_matches_reference。

    注意 h 不翻转："x[n-m]" 里已经含了反转，再翻一次会变成相关运算。

    调用方需保证 x、h 已经是非空一维 float64 数组。
    """
    N, M = x.size, h.size
    full_res = np.zeros(N + M - 1, dtype=np.float64)

    for n in range(N + M - 1):
        m_lo = max(0, n - N + 1)  # 保证 n - m <= N - 1
        m_hi = min(M - 1, n)      # 保证 n - m >= 0
        acc = 0.0
        for m in range(m_lo, m_hi + 1):
            acc += x[n - m] * h[m]
        full_res[n] = acc

    return full_res


def myconv(x, h, mode="full"):
    """
    手写一维离散卷积

    Parameters
    ----------
    x : array_like, 输入信号（一维）list/tuple/np.ndarray
    h : array_like, 卷积核/单位脉冲响应（一维）list/tuple/np.ndarray
    mode : str, {'full', 'same', 'valid'}, default='full'
        full: 完整卷积，输出长度 N+M-1
        same: 输出长度 max(N, M)，取 full 结果的居中段
        valid: 仅核与信号完全重叠的部分，输出长度 |N-M|+1

    Returns
    -------
    np.ndarray : 卷积结果，float64 一维数组

    Notes
    -----
    三种模式都先算出 full 卷积，再切片得到 same / valid，
    所以模式切换只是切片位置的差别，不涉及另一套数值算法。

    full 卷积默认调用 conv_full_vectorized（移位累加，快）。
    需要和定义逐行对照时，改用 conv_full_reference。
    全程不调用现成卷积函数。
    """
    if mode not in _MODES:
        raise ValueError(f"mode 必须是 {_MODES} 之一，收到 {mode!r}")

    _validate_1d(x, h)
    x = np.asarray(x, dtype=np.float64)
    h = np.asarray(h, dtype=np.float64)
    N = x.size  # 信号长度
    M = h.size  # 卷积核长度

    full_res = conv_full_vectorized(x, h)

    if mode == "full":
        return full_res
    if mode == "valid":
        return full_res[min(N, M) - 1 : max(N, M)]

    # mode == "same"：长度取较长者，居中截取
    length, shorter = max(N, M), min(N, M)
    start = (shorter - 1) // 2
    return full_res[start : start + length]


# 模块自测：单独运行 myconv.py 时执行基础测试
# 注意：这里刻意用非对称核，对称核会掩盖"翻转"类错误。
if __name__ == "__main__":
    print("=== myconv 模块内置自测 ===")
    cases = [
        ("非对称核", [1, 2, 3], [1, 2],
         {"full": [1, 4, 7, 6], "same": [1, 4, 7], "valid": [4, 7]}),
        ("核长于信号", [1, 2], [1, 2, 3],
         {"full": [1, 4, 7, 6], "same": [1, 4, 7], "valid": [4, 7]}),
    ]
    failed = 0
    for name, sig, ker, expect in cases:
        print(f"-- {name}: x={sig}, h={ker}")
        for m in _MODES:
            res = myconv(sig, ker, mode=m)
            ok = np.allclose(res, expect[m])
            failed += (not ok)
            print(f"   mode={m:5} result={res} 期望={expect[m]} [{'OK' if ok else 'FAIL'}]")

    # 交叉验证：两个 full 实现必须逐位相同（用 array_equal，不容忍误差）
    rng = np.random.default_rng(0)
    mismatch = 0
    for _ in range(200):
        n = int(rng.integers(1, 40))
        m = int(rng.integers(1, 40))
        sig = rng.uniform(-10, 10, size=n)
        ker = rng.uniform(-10, 10, size=m)
        if not np.array_equal(conv_full_vectorized(sig, ker), conv_full_reference(sig, ker)):
            mismatch += 1
    ok = mismatch == 0
    failed += (not ok)
    print(f"-- 交叉验证 200 组随机用例：逐位不同的有 {mismatch} 组 [{'OK' if ok else 'FAIL'}]")

    print("自测" + ("全部通过" if failed == 0 else f"失败 {failed} 项"))
