"""比较 FIR 与 Butterworth IIR 低通的幅频、相位和群延迟。"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from dsp_toolbox.fft import myfft
from dsp_toolbox.filters import design_fir_lowpass, design_iir


def main(argv=None):
    default_output = (
        Path(__file__).resolve().parents[1] / "outputs" / "fir_iir_comparison.png"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=default_output)
    args = parser.parse_args(argv)

    fs = 1000
    fc = 200
    numtaps = 101
    order = 4
    n_fft = 1024

    # 一、设计滤波器。
    h_fir = design_fir_lowpass(
        fc, fs, numtaps, window="hamming"
    )
    b_iir, a_iir = design_iir(fc, fs, order=order)

    # 二、把系数补零到相同长度，方便使用同一组 FFT 频率点。
    h_padded = np.zeros(n_fft)
    b_padded = np.zeros(n_fft)
    a_padded = np.zeros(n_fft)

    h_padded[:len(h_fir)] = h_fir
    b_padded[:len(b_iir)] = b_iir
    a_padded[:len(a_iir)] = a_iir

    # FIR：频率响应就是冲激响应的 FFT。
    # IIR：分别计算分子、分母的频响，然后相除。
    response_fir = myfft(h_padded)
    response_iir = myfft(b_padded) / myfft(a_padded)

    # 三、取非负频率部分。
    positive_count = n_fft // 2 + 1
    freqs = np.arange(positive_count) * fs / n_fft

    response_fir = response_fir[:positive_count]
    response_iir = response_iir[:positive_count]

    responses = {
        "FIR": response_fir,
        "IIR": response_iir,
    }
    colors = {
        "FIR": "tab:blue",
        "IIR": "tab:orange",
    }

    # 相位和群延迟只观察通带内部，
    # 避开阻带零点附近幅度过小、相位难以解释的位置。
    passband_limit = 0.8 * fc
    passband_mask = freqs <= passband_limit
    passband_freqs = freqs[passband_mask]

    # 四、创建四幅图。
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    ax_gain = axes[0, 0]
    ax_phase = axes[0, 1]
    ax_fir_delay = axes[1, 0]
    ax_iir_delay = axes[1, 1]

    delay_axes = {
        "FIR": ax_fir_delay,
        "IIR": ax_iir_delay,
    }

    for name, response in responses.items():
        color = colors[name]

        # 频率响应的幅度增益：不乘 2，也不除以 FFT 长度。
        gain = np.abs(response)
        gain_db = 20 * np.log10(np.maximum(gain, 1e-12))
        ax_gain.plot(freqs, gain_db, color=color, label=name)

        # 取出通带相位，并连接跨越 ±π 产生的跳变。
        phase = np.unwrap(
            np.angle(response[passband_mask])
        )
        ax_phase.plot(
            passband_freqs, phase, color=color, label=name
        )

        # 相邻两点计算相位斜率，换算成群延迟。
        slope = np.diff(phase) / np.diff(passband_freqs)
        delay_ms = -slope / (2 * np.pi) * 1000

        # 每个斜率对应两个频率点之间的中点。
        delay_freqs = (
            passband_freqs[:-1] + passband_freqs[1:]
        ) / 2

        ax_delay = delay_axes[name]
        ax_delay.plot(delay_freqs, delay_ms, color=color)
        ax_delay.set_title(f"{name} group delay")
        ax_delay.set_xlabel("Frequency (Hz)")
        ax_delay.set_ylabel("Group delay (ms)")
        ax_delay.set_xlim(0, passband_limit)

        # 两种延迟分别画图，方便看清 IIR 延迟随频率的变化。
        # 注意：两幅延迟图的纵轴范围不同，请比较数值。
        ax_delay.set_ylim(0, np.max(delay_ms) * 1.2)

        print(
            f"{name} 通带群延迟范围："
            f"{np.min(delay_ms):.4f} ～ "
            f"{np.max(delay_ms):.4f} ms"
        )

    # 五、设置幅频图。
    ax_gain.axvline(
        fc,
        color="gray",
        linestyle="--",
        label=f"fc = {fc} Hz",
    )
    ax_gain.set_title("Magnitude response")
    ax_gain.set_xlabel("Frequency (Hz)")
    ax_gain.set_ylabel("Gain (dB)")
    ax_gain.set_xlim(0, fs / 2)
    ax_gain.set_ylim(-140, 5)
    ax_gain.legend()

    # 设置相位图。
    ax_phase.set_title("Phase response in the passband")
    ax_phase.set_xlabel("Frequency (Hz)")
    ax_phase.set_ylabel("Phase (rad)")
    ax_phase.set_xlim(0, passband_limit)
    ax_phase.legend()

    for ax in axes.flat:
        ax.grid(True)

    fig.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150)
    plt.close(fig)

    expected_delay_ms = (numtaps - 1) / (2 * fs) * 1000
    print(f"FIR 理论群延迟：{expected_delay_ms:.4f} ms")
    print(f"图片已保存：{args.output.resolve()}")


if __name__ == "__main__":
    main()
