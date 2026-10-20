"""用 Goertzel 算法识别单个或连续的 DTMF 按键音。"""

import argparse
import wave

import numpy as np


KEYPAD = ("123A", "456B", "789C", "*0#D")
LOW_FREQS = (697, 770, 852, 941)
HIGH_FREQS = (1209, 1336, 1477, 1633)


def validate_input(x, sr):
    """检查单声道信号和采样率，并返回浮点数组。"""
    if isinstance(sr, (bool, np.bool_)) or not isinstance(sr, (int, np.integer)):
        raise ValueError("sr 必须是正整数")
    if sr <= 2 * HIGH_FREQS[-1]:
        raise ValueError("sr 太低，无法区分全部 DTMF 频率")

    samples = np.asarray(x)
    if samples.ndim != 1 or samples.size == 0:
        raise ValueError("x 必须是非空一维序列")
    if samples.dtype.kind not in "iuf":
        raise ValueError("x 必须是实数序列")
    if not np.all(np.isfinite(samples)):
        raise ValueError("x 不能包含 NaN 或无穷大")
    return samples.astype(np.float64)


def rms_level(x):
    """计算信号的均方根音量。"""
    return float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2)))


def goertzel_strength(x, freq, sr):
    """计算信号在指定频率处的强度。"""
    coefficient = 2 * np.cos(2 * np.pi * freq / sr)
    previous1 = 0.0
    previous2 = 0.0

    for sample in x:
        current = sample + coefficient * previous1 - previous2
        previous2 = previous1
        previous1 = current

    power = previous1**2 + previous2**2 - coefficient * previous1 * previous2
    return float(np.sqrt(max(power, 0.0)))


def peak_ratio(scores):
    """返回候选频率中最大强度与第二大强度的比值。"""
    ordered = sorted(scores.values(), reverse=True)
    if len(ordered) < 2:
        raise ValueError("至少需要两个候选频率")
    if ordered[1] == 0:
        return float("inf") if ordered[0] > 0 else 0.0
    return ordered[0] / ordered[1]


def detect_key(x, sr, min_rms=0.02, min_ratio=1.5, min_concentration=0.15):
    """识别一段按键音；证据不足时返回 None。

    输入应为幅度约在 -1 到 1 之间的单声道样本。
    """
    samples = validate_input(x, sr)
    level = rms_level(samples)
    if level < min_rms:
        return None

    low_scores = {}
    high_scores = {}

    for freq in LOW_FREQS:
        low_scores[freq] = goertzel_strength(samples, freq, sr)
    for freq in HIGH_FREQS:
        high_scores[freq] = goertzel_strength(samples, freq, sr)

    if peak_ratio(low_scores) < min_ratio or peak_ratio(high_scores) < min_ratio:
        return None

    best_low = max(low_scores, key=low_scores.get)
    best_high = max(high_scores, key=high_scores.get)

    weaker = min(low_scores[best_low], high_scores[best_high])
    if weaker / (len(samples) * level) < min_concentration:
        return None

    row = LOW_FREQS.index(best_low)
    col = HIGH_FREQS.index(best_high)
    return KEYPAD[row][col]


def read_wav(path):
    """读取未压缩的 16-bit PCM WAV，转换为浮点单声道。"""
    with wave.open(str(path), "rb") as wav:
        if wav.getcomptype() != "NONE" or wav.getsampwidth() != 2:
            raise ValueError("目前只支持未压缩的 16-bit PCM WAV")
        sr = wav.getframerate()
        channels = wav.getnchannels()
        data = wav.readframes(wav.getnframes())

    samples = np.frombuffer(data, dtype="<i2").astype(np.float64) / 32768.0
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    return samples, sr


def detect_dtmf(path):
    """从 WAV 文件中识别连续按键，返回如 '1380' 的字符串。"""
    samples, sr = read_wav(path)
    samples = validate_input(samples, sr)

    frame_size = round(0.04 * sr)  # 每次查看 40 ms 音频
    hop_size = round(0.01 * sr)    # 每隔 10 ms 查看一次

    if len(samples) < frame_size:
        return detect_key(samples, sr) or ""

    starts = list(range(0, len(samples) - frame_size + 1, hop_size))
    if starts[-1] != len(samples) - frame_size:
        starts.append(len(samples) - frame_size)

    labels = [
        detect_key(samples[start : start + frame_size], sr)
        for start in starts
    ]

    # 合并相邻窗口；短暂的一帧空白不拆开同一个按键。
    result = []
    active = None
    active_count = 0
    blank_count = 0

    for label in labels + [None, None]:
        if label == active and label is not None:
            active_count += 1
            blank_count = 0
        elif label is None and active is not None:
            blank_count += 1
            if blank_count >= 2:
                if active_count >= 2:
                    result.append(active)
                active = None
                active_count = 0
                blank_count = 0
        elif label is not None:
            if active is not None and active_count >= 2:
                result.append(active)
            active = label
            active_count = 1
            blank_count = 0

    return "".join(result)


def main():
    parser = argparse.ArgumentParser(description="识别 16-bit PCM WAV 中的 DTMF 按键")
    parser.add_argument("path", help="WAV 文件路径")
    args = parser.parse_args()
    print(detect_dtmf(args.path))


if __name__ == "__main__":
    main()