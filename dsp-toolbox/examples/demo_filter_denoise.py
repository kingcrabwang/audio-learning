"""比较三种 FIR 滤波器对合成信号的作用。"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from dsp_toolbox.conv import myconv
from dsp_toolbox.fft import myfft
from dsp_toolbox.filters import (
    design_fir_bandpass,
    design_fir_lowpass,
    design_fir_notch,
)
from dsp_toolbox.signals import sine_wave


def main(argv=None):
    default_output = (
        Path(__file__).resolve().parents[1]
        / "outputs"
        / "filter_denoise_comparison.png"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=default_output)
    args = parser.parse_args(argv)

    fs = 1000
    duration = 4096 / fs
    target = sine_wave(200, fs, duration)
    hum = sine_wave(50, fs, duration, amplitude=0.2)
    high_noise = sine_wave(350, fs, duration, amplitude=0.15)
    x = target + hum + high_noise

    h_low = design_fir_lowpass(250, fs, 101)
    h_band = design_fir_bandpass(150, 250, fs, 101)
    h_notch = design_fir_notch(50, fs, q=5)

    signals = {
        "Original": x,
        "Lowpass": myconv(x, h_low, mode="same"),
        "Bandpass": myconv(x, h_band, mode="same"),
        "Notch": myconv(x, h_notch, mode="same"),
    }

    n = len(x)
    freqs = np.arange(n // 2 + 1) * fs / n
    window = np.hanning(n)
    plot_data = {}
    for name, samples in signals.items():
        spectrum = myfft(samples * window)
        amplitude = np.abs(spectrum[: n // 2 + 1]) / np.sum(window)
        # 首尾是直流和奈奎斯特频点，其余正频率合并负频率的幅度。
        amplitude[1:-1] *= 2
        plot_data[name] = 20 * np.log10(np.maximum(amplitude, 1e-12))

    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True, sharey=True)
    for ax, (name, amplitude_db) in zip(axes, plot_data.items()):
        ax.plot(freqs, amplitude_db)
        ax.set_title(name)
        ax.set_ylabel("Amplitude (dB)")
        ax.grid(True)
        for frequency in (50, 200, 350):
            ax.axvline(frequency, color="gray", linestyle="--", alpha=0.4)

    axes[-1].set_xlabel("Frequency (Hz)")
    axes[-1].set_xlim(0, fs / 2)
    axes[-1].set_ylim(-100, 0)
    fig.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150)
    plt.close(fig)
    print(f"图片已保存：{args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
