import cmath
import math

import numpy as np


def reverse_bits(index, bits):
    result = 0
    for _ in range(bits):
        bit = index % 2
        result = 2 * result + bit
        index //= 2
    return result

def conjugate(x):
    arr = np.asarray(x)

    if arr.ndim != 1:
        raise ValueError("x 必须是一维序列")

    if arr.size == 0:
        raise ValueError("x 不能为空")

    if arr.dtype.kind not in "biufc":
        raise ValueError("x 必须是数值序列")

    arr = arr.astype(np.complex128)

    return np.conjugate(arr)


def myfft(x):
    samples = np.asarray(x)

    if samples.ndim != 1 or samples.size == 0:
        raise ValueError("x 必须是非空一维序列")
    if samples.dtype.kind not in "biufc":
        raise ValueError("x 必须是数值序列")
    if not np.all(np.isfinite(samples)):
        raise ValueError("x 不能包含 NaN 或无穷大")

    n = samples.size

    # 一边检查长度是否为 2 的幂，一边算二进制需要多少位。
    bits = 0
    remaining = n
    while remaining > 1:
        if remaining % 2 != 0:
            raise ValueError("x 的长度必须是 2 的幂")
        remaining //= 2
        bits += 1

    samples = samples.astype(np.complex128)
    ordered = np.empty(n, dtype=np.complex128)

    # 按二进制位倒序排列输入。
    for i in range(n):
        ordered[reverse_bits(i, bits)] = samples[i]

    # 每组长度依次为 2、4、8……直到 n。
    size = 2
    while size <= n:
        half = size // 2

        for start in range(0, n, size):
            for k in range(half):
                left = start + k
                right = left + half

                u = ordered[left]
                rotation = cmath.exp(-2j * math.pi * k / size)
                t = ordered[right] * rotation

                ordered[left] = u + t
                ordered[right] = u - t

        size *= 2

    return ordered

def myifft(X):
    X_conj = conjugate(X)
    N = len(X)
    return conjugate(myfft(X_conj)) / N