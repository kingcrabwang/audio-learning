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

def _validate_iir_params(fc,fs):
    real_types = (int, float, np.integer, np.floating)
    bool_types = (bool, np.bool_)
    if isinstance(fc,bool_types) or not isinstance(fc,real_types):
        raise ValueError("截至频率必须为实数，不能为布尔值")
    if isinstance(fs,bool_types) or not isinstance(fs,real_types):
        raise ValueError("采样率必须为实数，不能为布尔值")
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("采样率必须有限且大于零")
    if not np.isfinite(fc) or fc<= 0 or fc>=fs/2:
        raise ValueError("fc 必须是有限实数，且满足 0 < fc < fs / 2")

def _validate_fir_bandpass_params(f1, f2, fs, numtaps):
    # 分别检查两个频率，以及采样率、系数数量。
    _validate_fir_lowpass_params(f1, fs, numtaps, "hamming")
    _validate_fir_lowpass_params(f2, fs, numtaps, "hamming")

    if f1 >= f2:
        raise ValueError("必须满足 0 < f1 < f2 < fs / 2")

def _validate_fir_notch_params(f0, fs, q):
    real_types = (int, float, np.integer, np.floating)

    for name, value in (("fs", fs), ("f0", f0), ("q", q)):
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, real_types)
        ):
            raise ValueError(f"{name} 必须是实数，不能是布尔值")

        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} 必须有限且大于零")

    bandwidth = f0 / q
    f1 = f0 - bandwidth / 2
    f2 = f0 + bandwidth / 2

    if not 0 < f1 < f2 < fs / 2:
        raise ValueError("陷波频段必须满足 0 < f1 < f2 < fs / 2")

def _filter_real_vector(values, name, allow_empty=False):
    """Validate first, so a rejected block cannot partly change filter state."""
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError(f"{name} 必须是一维实数序列")
    if not allow_empty and arr.size == 0:
        raise ValueError(f"{name} 不能为空")
    if arr.dtype.kind not in "iuf":
        raise ValueError(f"{name} 必须是实数序列，不能包含布尔值或复数")
    result = np.array(arr, dtype=np.float64, copy=True)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} 不能包含 NaN 或无穷大")
    return result


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

def design_iir(fc, fs, order=4, btype="low", ftype="butter"):
    """设计 Butterworth 数字低通，返回分子 b、分母 a 系数。"""
    _validate_iir_params(fc, fs)

    if (
        isinstance(order, (bool, np.bool_))
        or not isinstance(order, (int, np.integer))
        or order < 1
    ):
        raise ValueError("order 必须是正整数")

    if not isinstance(btype, str) or btype != "low":
        raise ValueError("当前仅支持 btype='low'")

    if not isinstance(ftype, str) or ftype != "butter":
        raise ValueError("当前仅支持 ftype='butter'")

    order = int(order)

    # 1. 预畸变：从目标数字截止频率得到模拟角截止频率。
    omega_c = 2 * fs * np.tan(np.pi * fc / fs)

    # 2. 生成模拟极点，再映射到数字系统。
    analog_poles = butterworth_analog_poles(order, omega_c)
    digital_poles = bilinear_poles(analog_poles, fs)

    # 3. 展开数字极点，得到分母系数。
    a = polynomial_from_roots(digital_poles)

    # 4. 数字零点均在 -1，展开后得到尚未调整增益的分子。
    digital_zeros = [-1.0] * order
    b_base = polynomial_from_roots(digital_zeros)

    # 只清理接近零的虚部。
    a = np.real_if_close(a)
    b_base = np.real_if_close(b_base)

    if np.iscomplexobj(a) or np.iscomplexobj(b_base):
        raise ValueError("系数存在明显虚部，请检查极点和展开过程")

    # 5. 调整分子，使直流增益 H(1) = 1。
    gain = np.sum(a) / np.sum(b_base)
    b = gain * b_base

    return b.astype(np.float64), a.astype(np.float64)


def filter_first_order(x, b, a):
    previous_input=0.0
    previous_output=0.0
    outputs = []
    for sample in x:
        current_output=b[0]*sample+b[1]*previous_input-a[1]*previous_output
        outputs.append(current_output)
        previous_input=sample
        previous_output=current_output
    return np.array(outputs,dtype=np.float64)

def butterworth_analog_poles(order, omega_c):
    poles=[]
    for k in range(order):
        angle=(2*k+order+1)*np.pi/(2*order)
        poles.append(omega_c*np.exp(1j*angle))

    return np.array(poles,dtype=np.complex128)

def bilinear_poles(analog_poles, fs):
    poles_digital=[]
    for p in analog_poles:
        p_digital=(2*fs+p)/(2*fs-p)
        poles_digital.append(p_digital)
    return np.array(poles_digital,dtype=np.complex128)

def polynomial_from_roots(roots):
    coefficients = np.array([1.0], dtype=np.complex128)
    for p in roots:
        new_coefficients=np.zeros(len(coefficients)+1,dtype=np.complex128)
        for i in range(len(coefficients)):
            new_coefficients[i] += coefficients[i]
            new_coefficients[i + 1] -= p * coefficients[i]
        coefficients = new_coefficients
    return coefficients

def design_fir_bandpass(f1, f2, fs, numtaps):
    _validate_fir_bandpass_params(f1, f2, fs, numtaps)
    c1=2*f1/fs
    c2=2*f2/fs
    ideal1=ideal_lowpass(c1,numtaps)
    ideal2=ideal_lowpass(c2,numtaps)
    h_ideal=ideal2-ideal1
    window=hamming_window(numtaps)
    h_windowed=h_ideal*window
    mid_freq = (f1 + f2) / 2

    # 各个系数相对于对称中心的位置。
    positions = np.arange(numtaps) - (numtaps - 1) / 2

    # 去掉统一延迟后，通带中心的实数响应。
    weights = np.cos(2 * np.pi * mid_freq * positions / fs)
    gain = np.sum(h_windowed * weights)

    # 让通带中心的幅度增益变为 1。
    return h_windowed / gain

def design_fir_notch(f0, fs, q):
    _validate_fir_notch_params(f0, fs, q)
    bandwidth = f0 / q
    f1 = f0 - bandwidth / 2
    f2 = f0 + bandwidth / 2

    numtaps = math.ceil(4 * fs / bandwidth)
    if numtaps % 2 == 0:
        numtaps += 1
    ideal_band = (
    ideal_lowpass(2 * f2 / fs, numtaps)
    - ideal_lowpass(2 * f1 / fs, numtaps)
    )
    h_band = ideal_band * hamming_window(numtaps)
    h_delay=np.zeros(numtaps)
    h_delay[(numtaps-1)//2]=1.0
    h_notch=h_delay-h_band
    # 把直流增益调整为 1。
    return h_notch / np.sum(h_notch)

class FirFilter:
    """Causal real FIR with a persistent circular input buffer."""

    def __init__(self, h):
        self.h = _filter_real_vector(h, "h")
        self.buffer = np.zeros(len(self.h), dtype=np.float64)
        self.write_index = 0

    def process(self, block):
        samples = _filter_real_vector(block, "block", allow_empty=True)
        outputs = []
        for sample in samples:
            self.buffer[self.write_index] = sample
            total = 0.0
            for k in range(len(self.h)):
                index = (self.write_index - k) % len(self.buffer)
                total += self.h[k] * self.buffer[index]
            outputs.append(total)
            self.write_index = (self.write_index + 1) % len(self.buffer)
        return np.asarray(outputs, dtype=np.float64)


class IirFilter:
    """Causal real IIR using separate input and output circular buffers."""

    def __init__(self, b, a):
        b = _filter_real_vector(b, "b")
        a = _filter_real_vector(a, "a")
        if a[0] == 0:
            raise ValueError("a[0] 不能为零")

        # Scale both arrays by the ORIGINAL a[0], then a[0] becomes 1.
        scale = a[0]
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            self.b = b / scale
            self.a = a / scale
        if not (
            np.all(np.isfinite(self.b))
            and np.all(np.isfinite(self.a))
        ):
            raise ValueError("系数归一化后必须为有限实数")

        self.input_buffer = np.zeros(len(self.b), dtype=np.float64)
        self.output_buffer = np.zeros(len(self.a), dtype=np.float64)
        self.input_index = 0
        self.output_index = 0

    def process(self, block):
        samples = _filter_real_vector(block, "block", allow_empty=True)
        outputs = []
        for sample in samples:
            self.input_buffer[self.input_index] = sample
            total = 0.0

            for k in range(len(self.b)):
                index = (self.input_index - k) % len(self.input_buffer)
                total += self.b[k] * self.input_buffer[index]

            for k in range(1, len(self.a)):
                index = (self.output_index - k) % len(self.output_buffer)
                total -= self.a[k] * self.output_buffer[index]

            self.output_buffer[self.output_index] = total
            outputs.append(total)
            self.input_index = (self.input_index + 1) % len(self.input_buffer)
            self.output_index = (self.output_index + 1) % len(self.output_buffer)

        return np.asarray(outputs, dtype=np.float64)
