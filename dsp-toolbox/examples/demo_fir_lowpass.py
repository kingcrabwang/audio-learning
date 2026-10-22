import numpy as np
from dsp_toolbox.filters import design_fir_lowpass
from dsp_toolbox.fft import myfft
import matplotlib.pyplot as plt

fs=16000
fc=2000
fig, ax = plt.subplots(figsize=(9, 4))

n_fft = 1024
freqs = np.arange(n_fft // 2 + 1) * fs / n_fft

for numtaps in (31,63,127):
    h=design_fir_lowpass(fc,fs,numtaps,window="hamming")
    h_padded = np.zeros(n_fft)
    h_padded[:len(h)]=h

    response=myfft(h_padded)
    positive_response=response[:n_fft//2+1]

    gain = np.abs(positive_response)
    # 避免对 0 取对数；这里只限制绘图数值，不改变滤波器。
    gain_db = 20 * np.log10(np.maximum(gain, 1e-12))

    print("频率点数：", len(freqs))
    print("增益点数：", len(gain))
    print("0 Hz 增益：", gain[0])
    ax.plot(freqs, gain_db, label=f"{numtaps} taps")





# 标出设计时指定的截止频率。
ax.axvline(fc, color="red", linestyle="--", label=f"Cutoff: {fc} Hz")

ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Gain (dB)")
ax.set_title("FIR lowpass frequency response")
ax.set_xlim(1000, 3000)
ax.set_ylim(-100, 5)
ax.grid(True)
ax.legend()

fig.tight_layout()
fig.savefig("fir_lowpass_response.png", dpi=150)
plt.close(fig)