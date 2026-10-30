"""比较两种 FFT 长度和三种窗函数对频谱泄漏的影响。"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from dsp_toolbox.fft import myfft
from dsp_toolbox.signals import sine_wave


def main(argv=None):
    default_output = (
        Path(__file__).resolve().parents[1] / "outputs" / "window_matrix.png"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=default_output)
    args = parser.parse_args(argv)

    sr = 8192
    single_freq = 1005
    lengths = (1024, 2048)
    names = ("矩形窗", "Hann 窗", "Hamming 窗")
    plot_data = {}

    for n in lengths:
        signal = sine_wave(single_freq, sr, n / sr, amplitude=0.5)
        windows = {
            "矩形窗": np.ones(n),
            "Hann 窗": np.hanning(n),
            "Hamming 窗": np.hamming(n),
        }
        for name, window in windows.items():
            spectrum = myfft(signal * window)
            positive = spectrum[: n // 2 + 1]
            freqs = np.arange(n // 2 + 1) * sr / n
            amplitude = 2 * np.abs(positive) / np.sum(window)
            db = 20 * np.log10(np.maximum(amplitude, 1e-12))
            plot_data[(n, name)] = (freqs, db)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for row, n in enumerate(lengths):
        for col, name in enumerate(names):
            freqs, db = plot_data[(n, name)]
            ax = axes[row, col]
            ax.plot(freqs, db)
            ax.set_xlim(800, 1200)
            ax.set_ylim(-100, 0)
            ax.set_title(f"{n} 点 · {name}")
            ax.set_xlabel("频率 (Hz)")
            ax.set_ylabel("幅度 (dB)")
            ax.grid(True)
    fig.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150)
    plt.close(fig)
    print(f"图片已保存：{args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
