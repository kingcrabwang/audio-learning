from dsp_toolbox.signals import sine_wave
from dsp_toolbox.filters import (
    design_fir_lowpass,
    design_fir_bandpass,
    design_fir_notch
    )
from dsp_toolbox.conv import myconv
import numpy as np
from dsp_toolbox.fft import myfft
import matplotlib.pyplot as plt

fs=1000
duration=4096/fs

target=sine_wave(200,fs,duration)
hum=sine_wave(50,fs,duration,amplitude=0.2)
high_noise=sine_wave(350,fs,duration,amplitude=0.15)

x=target+hum+high_noise

h_low=design_fir_lowpass(250,fs,101)
h_band=design_fir_bandpass(150,250,fs,101)
h_notch = design_fir_notch(50, fs, q=5)

y_low=myconv(x,h_low,mode="same")
y_band=myconv(x,h_band,mode="same")
y_notch=myconv(x,h_notch,mode="same")

signals = {
    "Original": x,
    "Lowpass": y_low,
    "Bandpass": y_band,
    "Notch": y_notch,
}

n = len(x)
freqs = np.arange(n // 2 + 1) * fs / n
window = np.hanning(n)

plot_data = {}
for name, samples in signals.items():
    spectrum = myfft(samples * window)

    # 取非负频率部分，并计算幅度。
    amplitude = np.abs(spectrum[:n // 2 + 1]) / np.sum(window)

    # 合并正负频率的幅度贡献。
    # 当前 n=4096，是偶数，首尾分别为直流和奈奎斯特频点。
    amplitude[1:-1] *= 2

    # 转成 dB；设置下限以避免 log10(0)。
    amplitude_db = 20 * np.log10(np.maximum(amplitude, 1e-12))

    # 按信号名称保存，稍后画图。
    plot_data[name] = amplitude_db
fig, axes = plt.subplots(
    4, 1,
    figsize=(10, 10),
    sharex=True,
    sharey=True,
)

# 每次取一个子图，以及一组“名称、频谱”。
for ax, (name, amplitude_db) in zip(axes, plot_data.items()):
    ax.plot(freqs, amplitude_db)
    ax.set_title(name)
    ax.set_ylabel("Amplitude (dB)")
    ax.grid(True)

    # 标出已知的三个信号频率。
    for frequency in (50, 200, 350):
        ax.axvline(
            frequency,
            color="gray",
            linestyle="--",
            alpha=0.4,
        )

axes[-1].set_xlabel("Frequency (Hz)")
axes[-1].set_xlim(0, fs / 2)
axes[-1].set_ylim(-100, 0)

fig.tight_layout()
fig.savefig("filter_denoise_comparison.png", dpi=150)
plt.close(fig)