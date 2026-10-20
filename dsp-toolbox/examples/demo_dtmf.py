"""生成一段 DTMF WAV，并识别其中的连续按键。"""

from pathlib import Path
import wave

import numpy as np

from dsp_toolbox.dtmf import HIGH_FREQS, KEYPAD, LOW_FREQS, detect_dtmf
from dsp_toolbox.signals import sine_wave


SR = 8000
KEYS = "13800138000"
TONE_SECONDS = 0.12
GAP_SECONDS = 0.06


def make_sequence(keys):
    """把各按键的双音和按键之间的静音拼接起来。"""
    key_to_freqs = {
        key: (LOW_FREQS[row], HIGH_FREQS[col])
        for row, line in enumerate(KEYPAD)
        for col, key in enumerate(line)
    }

    pieces = [np.zeros(round(0.05 * SR))]
    gap = np.zeros(round(GAP_SECONDS * SR))

    for key in keys:
        low_freq, high_freq = key_to_freqs[key]
        low = sine_wave(low_freq, SR, TONE_SECONDS, amplitude=0.25)
        high = sine_wave(high_freq, SR, TONE_SECONDS, amplitude=0.25)
        pieces.extend((low + high, gap))

    return np.concatenate(pieces)


def save_pcm16(path, samples):
    """保存为检测器支持的 16-bit PCM WAV。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.round(np.clip(samples, -1, 1) * 32767).astype("<i2")

    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SR)
        wav.writeframes(pcm.tobytes())


def main():
    project_root = Path(__file__).resolve().parents[1]
    wav_path = project_root / "samples" / "dtmf_demo.wav"

    samples = make_sequence(KEYS)
    save_pcm16(wav_path, samples)

    print(f"WAV 文件：{wav_path}")
    print(f"预期按键：{KEYS}")
    print(f"识别结果：{detect_dtmf(wav_path)}")


if __name__ == "__main__":
    main()