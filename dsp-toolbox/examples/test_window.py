import numpy as np
import matplotlib.pyplot as plt

from dsp_toolbox.signals import sine_wave
from dsp_toolbox.fft import myfft


sr = 8192                 # 采样率：每秒采集 8192 个点
single_freq = 1005        # 要观察的正弦波频率

signals = {}              # 保存不同长度的原始信号
plot_data = {}            # 保存每一格图需要的 (频率轴, dB 幅度)

# 生成两段信号。点数不同，实际观察的时长也不同。
for n in (1024, 2048):
    duration = n / sr
    signals[n] = sine_wave(single_freq, sr, duration, amplitude=0.5)

# 分别分析两种信号长度。
for n in (1024, 2048):
    signal = signals[n]

    # 窗数组必须与当前信号一样长，才能逐点相乘。
    windows = {
        "矩形窗": np.ones(n),
        "Hann 窗": np.hanning(n),
        "Hamming 窗": np.hamming(n),
    }

    for name, window in windows.items():
        # 先加窗，再用自己写的 FFT 计算频谱。
        spectrum = myfft(signal * window)

        # 实数信号只画非负频率：从 0 Hz 到奈奎斯特频率。
        positive = spectrum[: n // 2 + 1]
        freqs = np.arange(n // 2 + 1) * sr / n

        # 用窗系数之和校正幅度，便于公平比较不同的窗。
        amplitude = 2 * np.abs(positive) / np.sum(window)

        # 转为 dB；给幅度设置下限，避免出现 log10(0)。
        db = 20 * np.log10(np.maximum(amplitude, 1e-12))

        # 用“点数 + 窗名称”作为键，保存这一格图的数据。
        plot_data[(n, name)] = (freqs, db)

# 两行对应两种点数，三列对应三种窗。
lengths = (1024, 2048)
names = ("矩形窗", "Hann 窗", "Hamming 窗")
fig, axes = plt.subplots(2, 3, figsize=(15, 8))

for row, n in enumerate(lengths):
    for col, name in enumerate(names):
        freqs, db = plot_data[(n, name)]
        ax = axes[row, col]  # 选中当前要画的格子

        ax.plot(freqs, db)
        ax.set_xlim(800, 1200)
        ax.set_ylim(-100, 0)
        ax.set_title(f"{n} 点 · {name}")
        ax.set_xlabel("频率 (Hz)")
        ax.set_ylabel("幅度 (dB)")
        ax.grid(True)

fig.tight_layout()                         # 调整间距，避免文字重叠
fig.savefig("window_matrix.png", dpi=150)  # 保存图片
plt.close(fig)                              # 释放图形资源