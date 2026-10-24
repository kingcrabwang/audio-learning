"""比较不同长度的 Hamming 窗 FIR 低通滤波器幅频响应。"""

import matplotlib.pyplot as plt
import numpy as np

from dsp_toolbox.filters import design_fir_lowpass
from dsp_toolbox.fft import myfft


def main():
    fs = 16000
    fc = 2000
    n_fft = 1024
    filter_lengths = (31, 63, 127)

    # 三个滤波器都补零到相同长度，因此共用一个频率轴。
    # 保留 0 Hz 到奈奎斯特频率，共 513 个频率点。
    freqs = np.arange(n_fft // 2 + 1) * fs / n_fft

    # 先创建画布，上下两副。
    fig, (ax, ax_phase, ax_delay) = plt.subplots(
    3, 1, figsize=(9, 11)
)

    for numtaps in filter_lengths:
        # 设计滤波器，得到 numtaps 个系数。
        h = design_fir_lowpass(
            fc, fs, numtaps, window="hamming"
        )

        # 前面放滤波器系数，剩余位置补零。
        # 补零使频响曲线采样更密，不改变滤波器本身。
        h_padded = np.zeros(n_fft)
        h_padded[:len(h)] = h

        # 对系数做 FFT，得到频率响应。
        response = myfft(h_padded)

        # 实数系数的频响幅度左右对称，只取非负频率部分。
        positive_response = response[:n_fft // 2 + 1]

        # 复数的模表示幅度增益，再转换成 dB。
        # 设置下限，避免对零取对数。
        gain = np.abs(positive_response)
        gain_db = 20 * np.log10(np.maximum(gain, 1e-12))

        #使用unwrap把跨越边界造成的 2π 跳变接起来，方便观察相位变化的趋势
        phase = np.unwrap(np.angle(positive_response))
        # np.diff 计算相邻元素的差：
        # [a, b, c] → [b-a, c-b]
        phase_change = np.diff(phase)
        frequency_change = np.diff(freqs)

        # 相位变化量 ÷ 频率变化量，得到斜率。
        slope = phase_change / frequency_change

        # 换算为延迟：先得到秒，再乘 1000 变成毫秒。
        delay_ms = -slope / (2 * np.pi) * 1000
        delay_freqs = (freqs[:-1] + freqs[1:]) / 2
        ax.plot(freqs, gain_db, label=f"{numtaps} taps")
    
        ax_phase.plot(freqs, phase, label=f"{numtaps} taps")
        ax_delay.plot(delay_freqs, delay_ms, label=f"{numtaps} taps")

        print(
            f"{numtaps} taps: "
            f"FFT 长度={n_fft}, "
            f"频率点数={len(freqs)}, "
            f"直流增益={gain[0]:.6f}"
        )

    # 标出截止频率，方便比较三条曲线的过渡区域。
    ax.axvline(
        fc,
        color="red",
        linestyle="--",
        label=f"Cutoff: {fc} Hz",
    )

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Gain (dB)")
    ax.set_title("FIR lowpass: different filter lengths")

    # 放大截止频率附近，观察过渡带宽度。
    # 若要查看完整非负频率范围，改成 ax.set_xlim(0, fs / 2)。
    ax.set_xlim(1000, 3000)
    ax.set_ylim(-100, 5)

    ax.grid(True)
    ax.legend()

    ax_phase.set_xlabel("Frequency (Hz)")
    ax_phase.set_ylabel("Phase (rad)")
    ax_phase.set_title("Phase response in the passband")

    # 先看通带，避开阻带零点附近难以解释的相位。
    ax_phase.set_xlim(0, 1500)

    ax_phase.grid(True)
    ax_phase.legend()
    ax_delay.set_xlabel("Frequency (Hz)")
    ax_delay.set_ylabel("Group delay (ms)")
    ax_delay.set_title("Group delay in the passband")
    ax_delay.set_xlim(0, 1500)
    ax_delay.grid(True)
    ax_delay.legend()
    fig.tight_layout()

    # 保存到运行程序时的当前工作目录。
    output_path = "fir_lowpass_response.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"图片已保存：{output_path}")


# 直接运行本文件时执行；被其他文件导入时不自动画图。
if __name__ == "__main__":
    main()