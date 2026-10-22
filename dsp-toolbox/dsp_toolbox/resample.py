import numpy as np
import math
from .filters import hamming_window, ideal_lowpass

def validate_signal(x):
    arr = np.asarray(x)

    if arr.ndim != 1:
        raise ValueError("x 必须是一维序列")

    if arr.size == 0:
        raise ValueError("x 不能为空")

    if arr.dtype.kind not in "biuf":
        raise ValueError("x 必须由实数构成")

    arr = arr.astype(np.float64)

    if not np.all(np.isfinite(arr)):
        raise ValueError("x 不能包含 NaN 或无穷大")

    return arr


def filtered_sample(x,h,up,q):
    total=0.0
    k_min = max(0, math.ceil((q - (len(h) - 1)) / up))
    k_max = min(len(x) - 1, q // up)

    for k in range(k_min, k_max + 1):
        tap_index=q-k*up

        if 0<=tap_index<len(h):
            total+=x[k]*h[tap_index]
    return total

def resample_reference(x,h,up,down):
    center=(len(h)-1)//2
    output_length=math.ceil(len(x)*up/down)

    out_put=[]

    for m in range(output_length):
        q=center+down*m
        value=filtered_sample(x,h,up,q)
        out_put.append(value)

    return np.asarray(out_put)

def design_resample_filter(up,down,numtaps):
    cutoff = 1.0 / max(up, down)

    h_ideal = ideal_lowpass(cutoff, numtaps)
    window = hamming_window(numtaps)

    h_windowed = h_ideal * window
    h_base = h_windowed / np.sum(h_windowed)

    return h_base * up


def decimate(x, M):
    x = validate_signal(x)
    M = validate_factor(M, "M")
    up = 1
    down = M
    numtaps = 20 * M + 1

    h = design_resample_filter(up, down, numtaps)
    return resample_reference(x, h, up, down)


def interpolate(x, L):
    x = validate_signal(x)
    L = validate_factor(L, "L")
    up = L
    down = 1
    numtaps = 20 * L + 1

    h = design_resample_filter(up, down, numtaps)
    return resample_reference(x, h, up, down)

def validate_factor(value, name):
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value < 1
    ):
        raise ValueError(f"{name} 必须是正整数")

    return int(value)
