import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

#1、定义一个二阶系统，分子为1，分母为1-2rcos(w0)z^-1+r^2z^-2
r=0.9
w0=np.pi/4

b=1#分子为1，表示0阶
a=[1,-2*r*np.cos(w0),r**2]#分母，数组下标就是阶数，1对应0阶，-2r*cos(w0)对应1阶，r**2对应2阶
#2、计算零点、极点和增益
z,p,k=signal.tf2zpk(b,a)
print("零点：",z)
print("极点：",p)
print("增益：",k)
print("极点的模：",np.abs(p))
print("极点的幅角：",np.angle(p))
#3、计算频率响应
fs=8000
w,h=signal.freqz(b,a,worN=2048)#返回数字角频率w，h是复数数组，幅值和相位都在。2048是采样点数
f_hz = w / (2 * np.pi) * fs                # rad/sample → Hz
f0 = w0 / (2 * np.pi) * fs                 # 极点角度也换回 Hz
mag_db = 20 * np.log10(np.abs(h))          # 幅度取 dB


#4、画图
fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))

# 单位圆（放在最底层）
theta = np.linspace(0, 2 * np.pi, 400)
axes[0].plot(np.cos(theta), np.sin(theta), color="0.75", lw=1, zorder=1)

# 极点 / 零点
axes[0].plot(p.real, p.imag, "x", ms=12, mew=2.5, color="crimson", zorder=3)
if z.size:
    axes[0].plot(z.real, z.imag, "o", mfc="none", ms=10, color="navy", zorder=3)

# 用箭头标注，而不是图例
axes[0].annotate("poles", xy=(p[0].real, p[0].imag), xytext=(0.05, 0.95),
            color="crimson", fontsize=10,
            arrowprops=dict(arrowstyle="-", color="crimson", lw=0.8))

# 辅助线与外观
axes[0].axhline(0, color="0.9", lw=0.8, zorder=0)
axes[0].axvline(0, color="0.9", lw=0.8, zorder=0)
axes[0].set_aspect("equal")                 # 必须保留，否则圆会变椭圆
axes[0].set_xlim(-1.25, 1.25)               # 固定范围，图不会偏
axes[0].set_ylim(-1.25, 1.25)
axes[0].set_xlabel("Real"); axes[0].set_ylabel("Imaginary")
axes[0].grid(alpha=0.3)

axes[1].plot(f_hz, mag_db, color="steelblue", lw=1.8)
axes[1].axvline(f0, color="crimson", ls="--", lw=1, label=f"w0 = {f0:.0f} Hz")
axes[1].set_xlabel("Frequency (Hz)")
axes[1].set_ylabel("Magnitude (dB)")
axes[1].set_title(f"Frequency Response (r={r})")
axes[1].grid(alpha=0.3)
axes[1].legend(loc="upper right")
fig.tight_layout()
plt.show()
print(f"峰值 = {np.abs(h).max():.2f}  ({mag_db.max():.1f} dB)")