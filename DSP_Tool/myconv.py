# myconv.py
"""
DSP_Tool: 手写一维卷积函数 myconv(x, h, mode)

实现一维离散卷积，支持 full / same / valid 三种模式。
不调用 np.convolve / scipy.signal 等现成卷积函数。

语义与 numpy.convolve / MATLAB conv 对齐（N = len(x)，M = len(h)）：
    full  : 输出长度 N + M - 1
    same  : 输出长度 max(N, M)，取 full 结果居中段
    valid : 输出长度 |N - M| + 1
"""
import numpy as np

__all__ = ["myconv"]

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
    按定义 y[n] = sum_m x[n-m] * h[m] 用双重循环实现 full 卷积，
    再切片得到 same / valid，全程不调用现成卷积函数。
    """
    if mode not in _MODES:
        raise ValueError(f"mode 必须是 {_MODES} 之一，收到 {mode!r}")

    _validate_1d(x, h)
    x = np.asarray(x, dtype=np.float64)
    h = np.asarray(h, dtype=np.float64)
    N = x.size  # 信号长度
    M = h.size  # 卷积核长度

    full_len = N + M - 1
    full_res = np.zeros(full_len, dtype=np.float64)

    # 按定义直接计算 full 卷积：y[n] = sum_m x[n-m] * h[m]
    # 这里 h 不翻转——"x[n-m]" 已经含有反转，再翻一次会变成相关运算。
    for n in range(full_len):
        m_lo = max(0, n - N + 1)  # 保证 n - m <= N - 1
        m_hi = min(M - 1, n)      # 保证 n - m >= 0
        acc = 0.0
        for m in range(m_lo, m_hi + 1):
            acc += x[n - m] * h[m]
        full_res[n] = acc

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
    print("自测" + ("全部通过" if failed == 0 else f"失败 {failed} 项"))
