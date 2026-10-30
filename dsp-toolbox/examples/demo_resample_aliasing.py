"""画出直接抽取与抗混叠抽取的频谱差异。"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from dsp_toolbox.resample import decimate


def main(argv=None):
    default_output = (
        Path(__file__).resolve().parents[1] / "outputs" / "resample_aliasing.png"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=default_output)
    args = parser.parse_args(argv)

    fs_in = 12_000
    factor = 3
    fs_out = fs_in // factor
    duration = 0.1
    t = np.arange(round(fs_in * duration)) / fs_in
    x = np.sin(2 * np.pi * 600 * t) + np.sin(2 * np.pi * 2800 * t)

    direct = x[::factor]
    filtered = decimate(x, factor)
    frequency = np.fft.rfftfreq(len(direct), d=1 / fs_out)
    direct_spectrum = np.abs(np.fft.rfft(direct)) / len(direct)
    filtered_spectrum = np.abs(np.fft.rfft(filtered)) / len(filtered)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(frequency, direct_spectrum)
    axes[0].set_title("Direct decimation")
    axes[0].axvline(1200, color="red", linestyle="--", label="Alias: 1200 Hz")
    axes[0].legend()

    axes[1].plot(frequency, filtered_spectrum)
    axes[1].set_title("Low-pass filtering + decimation")
    axes[1].set_xlabel("Frequency (Hz)")
    for ax in axes:
        ax.set_ylabel("Normalized magnitude")
        ax.set_xlim(0, fs_out / 2)
        ax.grid(alpha=0.3)
    fig.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150)
    plt.close(fig)
    print(f"图片已保存：{args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
