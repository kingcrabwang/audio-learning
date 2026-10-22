import numpy as np
import math

def _validate_fir_lowpass_params(fc, fs, numtaps, window):
    """检查低通滤波器的设计参数；非法输入抛出 ValueError。"""
    real_types = (int, float, np.integer, np.floating)
    bool_types = (bool, np.bool_)

    # 先检查采样率，后面才能安全地计算 fs / 2。
    if isinstance(fs, bool_types) or not isinstance(fs, real_types):
        raise ValueError("fs 必须是正的有限实数")

    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("fs 必须是正的有限实数")

    # 截止频率必须位于零和奈奎斯特频率之间。
    if isinstance(fc, bool_types) or not isinstance(fc, real_types):
        raise ValueError("fc 必须是实数，不能是布尔值")

    if not np.isfinite(fc) or fc <= 0 or fc >= fs / 2:
        raise ValueError("fc 必须是有限实数，且满足 0 < fc < fs / 2")

    # 当前实现采用奇数长度，且至少有三个系数。
    if isinstance(numtaps, bool_types) or not isinstance(
        numtaps, (int, np.integer)
    ):
        raise ValueError("numtaps 必须是整数，不能是布尔值")

    if numtaps < 3 or numtaps % 2 == 0:
        raise ValueError("numtaps 必须是大于等于 3 的奇数")

    # 检查窗名称的类型及支持范围。
    if not isinstance(window, str):
        raise ValueError("window 必须是字符串")

    if window not in ("hamming", "hann"):
        raise ValueError("window 仅支持 'hamming' 或 'hann'")


def ideal_lowpass(cutoff,numtaps):
    center=(numtaps-1)//2
    h=np.zeros(numtaps)

    for n in range(numtaps):
        d=n-center

        if d==0:
            h[n]=cutoff
        else:
            h[n]=math.sin(math.pi*cutoff*d)/(math.pi*d)
    return h

def hamming_window(numtaps):
    window=np.zeros(numtaps)

    for n in range(numtaps):
        window[n]=0.54-0.46*math.cos(2*math.pi*n/(numtaps-1))
    return window

def hann_window(numtaps):
    window = np.zeros(numtaps)

    for n in range(numtaps):
        window[n] = 0.5 - 0.5 * math.cos(
            2 * math.pi * n / (numtaps - 1)
        )

    return window
    
def design_fir_lowpass(fc, fs, numtaps, window="hamming"):
    """使用对称窗设计奇数长度的 FIR 低通滤波器。"""
    #1、检验参数是否合法
    _validate_fir_lowpass_params(fc, fs, numtaps, window)
    #2、生成理想低通
    cutoff = 2 * fc / fs
    h_ideal = ideal_lowpass(cutoff, numtaps)
    #3、选窗
    if window == "hamming":
        window_value = hamming_window(numtaps)
    else:
        # 检查已经通过，所以另一种合法选择就是 Hann。
        window_value = hann_window(numtaps)
    #4、加窗
    h_windowed = h_ideal * window_value
    #5、归一化处理
    h_base = h_windowed / np.sum(h_windowed)

    return h_base
