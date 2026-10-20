"""DTMF 单键、带噪音频及 WAV 序列检测测试。"""

import subprocess
import sys
import wave

import numpy as np
import pytest

from dsp_toolbox.dtmf import (
    HIGH_FREQS,
    KEYPAD,
    LOW_FREQS,
    detect_dtmf,
    detect_key,
    goertzel_strength,
    peak_ratio,
)


def direct_strength(x, freq, sr):
    """用正弦、余弦直接求和作为 Goertzel 的独立参照。"""
    t = np.arange(len(x)) / sr
    sine_score = np.sum(x * np.sin(2 * np.pi * freq * t))
    cosine_score = np.sum(x * np.cos(2 * np.pi * freq * t))
    return np.hypot(sine_score, cosine_score)


@pytest.mark.parametrize("freq", [697, 770, 852, 941, 1209, 1336, 1477, 1633, 913.25])
def test_goertzel_matches_direct_calculation(freq):
    rng = np.random.default_rng(42)
    x = rng.standard_normal(1024)
    actual = goertzel_strength(x, freq, 8000)
    expected = direct_strength(x, freq, 8000)
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize(
    ("row", "col"), [(row, col) for row in range(4) for col in range(4)]
)
def test_detects_each_key(row, col):
    sr = 8000
    t = np.arange(1024) / sr
    low = np.sin(2 * np.pi * LOW_FREQS[row] * t + 0.37)
    high = np.sin(2 * np.pi * HIGH_FREQS[col] * t - 0.22)
    tone = 0.25 * (low + high)
    assert detect_key(tone, sr) == KEYPAD[row][col]


def test_accepts_list_input():
    sr = 8000
    t = np.arange(1024) / sr
    tone = np.sin(2 * np.pi * 770 * t) + np.sin(2 * np.pi * 1336 * t)
    assert detect_key(tone.tolist(), sr) == "5"


def test_silence_returns_none():
    assert detect_key(np.zeros(1024), 8000) is None


def test_quiet_tone_needs_lower_rms_threshold():
    sr = 8000
    t = np.arange(1024) / sr
    tone = 0.001 * (
        np.sin(2 * np.pi * 770 * t) + np.sin(2 * np.pi * 1336 * t)
    )
    assert detect_key(tone, sr) is None
    assert detect_key(tone, sr, min_rms=0.0001) == "5"


@pytest.mark.parametrize(
    ("x", "sr"),
    [
        ([1, 2], 0),
        ([1, 2], 8000.5),
        ([1, 2], True),
        ([], 8000),
        ([[1, 2], [3, 4]], 8000),
        (["a", "b"], 8000),
        ([1 + 1j], 8000),
        ([np.nan, 1], 8000),
        ([np.inf, 1], 8000),
    ],
)
def test_invalid_input_raises_value_error(x, sr):
    with pytest.raises(ValueError):
        detect_key(x, sr)


def write_pcm16(path, samples, sr):
    """为端到端测试生成与真实 WAV 格式相同的文件。"""
    pcm = np.round(np.clip(samples, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sr)
        wav.writeframes(pcm.tobytes())


def make_tone(row, col, sr=8000, duration=0.12):
    t = np.arange(round(sr * duration)) / sr
    low = np.sin(2 * np.pi * LOW_FREQS[row] * t + 0.37)
    high = np.sin(2 * np.pi * HIGH_FREQS[col] * t - 0.22)
    return 0.25 * (low + high)


def test_peak_ratio():
    assert peak_ratio({697: 8.0, 770: 256.0, 852: 7.0, 941: 2.0}) == 32.0
    assert peak_ratio({697: 0.0, 770: 0.0}) == 0.0
    assert peak_ratio({697: 1.0, 770: 0.0}) == float("inf")


@pytest.mark.parametrize("snr_db", [20, 10, 0])
@pytest.mark.parametrize(("row", "col"), [(r, c) for r in range(4) for c in range(4)])
def test_detects_16_keys_at_three_noise_levels(tmp_path, row, col, snr_db):
    sr = 8000
    tone = make_tone(row, col, sr)
    clean = np.concatenate((np.zeros(400), tone, np.zeros(400)))
    rng = np.random.default_rng(1000 + row * 40 + col * 10 + snr_db)
    noise_std = rms_level_for_test(tone) / 10 ** (snr_db / 20)
    noisy = clean + rng.normal(0, noise_std, len(clean))
    path = tmp_path / "key.wav"
    write_pcm16(path, noisy, sr)
    assert detect_dtmf(path) == KEYPAD[row][col]


def rms_level_for_test(samples):
    return float(np.sqrt(np.mean(samples**2)))


def test_detects_key_sequence_with_repeated_key(tmp_path):
    sr = 8000
    keys = "13800138000"
    lookup = {
        key: (row, col)
        for row, line in enumerate(KEYPAD)
        for col, key in enumerate(line)
    }
    pieces = [np.zeros(400)]
    for key in keys:
        pieces.extend((make_tone(*lookup[key]), np.zeros(480)))
    path = tmp_path / "sequence.wav"
    write_pcm16(path, np.concatenate(pieces), sr)
    assert detect_dtmf(path) == keys


def test_noise_without_tones_returns_empty_string(tmp_path):
    sr = 8000
    rng = np.random.default_rng(123)
    noise = rng.normal(0, 0.2, 3200)
    path = tmp_path / "noise.wav"
    write_pcm16(path, noise, sr)
    assert detect_dtmf(path) == ""


def test_command_line_prints_detected_key(tmp_path):
    path = tmp_path / "key.wav"
    write_pcm16(path, make_tone(1, 1), 8000)
    result = subprocess.run(
        [sys.executable, "-m", "dsp_toolbox.dtmf", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == "5"
