from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from dsp_toolbox.resample import decimate


# 1. 生成包含两个频率成分的信号
fs_in = 12_000
M = 3
fs_out = fs_in // M

duration = 0.1
t = np.arange(round(fs_in * duration)) / fs_in

x = (
    np.sin(2 * np.pi * 600 * t)
    + np.sin(2 * np.pi * 2800 * t)
)


# 2. 分别进行直接抽取、抗混叠抽取
direct = x[::M]
filtered = decimate(x, M)


# 3. 计算两种结果的频谱
frequency = np.fft.rfftfreq(len(direct), d=1 / fs_out)

direct_spectrum = np.abs(np.fft.rfft(direct)) / len(direct)
filtered_spectrum = np.abs(np.fft.rfft(filtered)) / len(filtered)


# 4. 绘图
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


# 5. 保存图片
output = (
    Path(__file__).resolve().parents[1]
    / "docs"
    / "resample_aliasing.png"
)
output.parent.mkdir(parents=True, exist_ok=True)

fig.savefig(output, dpi=150)
plt.close(fig)

print("图片已保存到：", output)