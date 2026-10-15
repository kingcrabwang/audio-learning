# -*- coding: utf-8 -*-
"""
demo_moving_average.py —— 滑动平均滤波实战演示 (Moving Average Filter Demo)

本脚本演示如何使用手写的一维离散卷积 myconv 实现滑动平均滤波（去噪）：
1. 用 audio_base 生成 50Hz 纯净正弦波 (Clean Sine Wave)；
2. 人为叠加高斯白噪声 (Gaussian White Noise)，模拟受干扰信号；
3. 构造长度为 M 的归一化矩形平滑核：h = [1/M, ..., 1/M] (Boxcar Kernel)；
4. 调用 DSP_Tool.myconv(noisy, h, mode="same") 进行低通滤波；
5. 用 matplotlib 画出对比图。

注意：为了避免不同操作系统缺少中文字体而导致图表出现方块乱码（Tofu blocks），
图表内部（标题、坐标轴、图例）统一使用英文标注，代码注释与终端输出保留中文。

用法：
    python examples/demo_moving_average.py
    python examples/demo_moving_average.py --window 25 --noise 0.5
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# 将项目根目录加入模块搜索路径，保证在任何目录下运行本脚本都能正确导入
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from audio_base import sine_wave, time_axis  # noqa: E402
from DSP_Tool import myconv  # noqa: E402


def build_parser():
    parser = argparse.ArgumentParser(
        description="使用 myconv 实现正弦信号滑动平均去噪演示 (Moving Average Denoising)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--freq", type=float, default=50.0, help="正弦波频率 Frequency (Hz)")
    parser.add_argument("--sr", type=int, default=2000, help="采样率 Sampling Rate (Hz)")
    parser.add_argument("--duration", type=float, default=0.2, help="信号时长 Duration (秒)")
    parser.add_argument("--noise", type=float, default=0.35, help="白噪声标准差 Noise Std Dev (强度)")
    parser.add_argument("--window", type=int, default=15, help="滑动平均窗口大小 Window Size (点数)")
    parser.add_argument("--no-show", action="store_true", help="不弹窗显示（仅用于测试）")
    return parser


def moving_average_filter(x, window_size):
    """
    使用一维卷积实现的滑动平均滤波 (Moving Average Filter via 1D Convolution)。

    核 h 的每个抽头均为 1 / window_size，卷积结果即为局部滑动窗口的算术平均值。
    选用 mode='same'，保持滤波后信号长度与输入完全一致。
    """
    if window_size < 1:
        raise ValueError("window_size 必须大于等于 1")
    # 构造归一化的矩形窗滤波核 (Boxcar kernel)
    h = np.ones(window_size, dtype=np.float64) / float(window_size)
    return myconv(x, h, mode="same")


def main(argv=None):
    args = build_parser().parse_args(argv)

    # 1. 生成纯净正弦信号 (Clean Sine Signal)
    t = time_axis(args.duration, args.sr)
    clean = sine_wave(args.freq, args.sr, args.duration, amplitude=1.0)

    # 2. 叠加强高斯白噪声 (Additive White Gaussian Noise)
    rng = np.random.default_rng(seed=42)  # 固定随机数种子，保证每次运行结果一致
    noise = rng.normal(0.0, args.noise, size=len(clean))
    noisy = clean + noise

    # 3. 运行手写的 myconv 卷积做低通平滑滤波 (Smoothing / Low-pass Filter)
    smoothed = moving_average_filter(noisy, window_size=args.window)

    print("=== 滑动平均滤波演示 (Moving Average Filter Demo) ===")
    print(f"信号参数: 频率={args.freq}Hz, 采样率={args.sr}Hz, 总点数={len(clean)}")
    print(f"滤波参数: 窗口点数={args.window}, 对应时间窗={args.window / args.sr * 1000:.2f}ms")

    # 4. 绘图对比 (Plot Comparison)
    # 【注意】图表内部统一使用英文字符，避免操作系统缺失中文字体导致方块乱码
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=False)

    # 子图 1：整体波形全貌 (Global Overview)
    axes[0].plot(t * 1000, noisy, color="lightgray", label="Noisy Signal", alpha=0.9)
    axes[0].plot(t * 1000, clean, color="black", linestyle="--", linewidth=1.5, label="Clean Sine (Reference)")
    axes[0].plot(t * 1000, smoothed, color="crimson", linewidth=1.8, label=f"Smoothed (myconv, M={args.window})")
    axes[0].set_title(f"Moving Average Filter - Global Overview (freq={args.freq} Hz, sr={args.sr} Hz)")
    axes[0].set_xlabel("Time (ms)")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend(loc="upper right")

    # 子图 2：局部放大视图 (Zoomed View: 前 ~2.5 个完整周期，观察毛刺消除与微小相位延迟)
    zoom_pts = int(round(2.5 * (args.sr / args.freq)))
    t_zoom = t[:zoom_pts] * 1000

    axes[1].plot(t_zoom, noisy[:zoom_pts], color="darkgray", marker=".", markersize=4, label="Noisy (samples)")
    axes[1].plot(t_zoom, clean[:zoom_pts], color="black", linestyle="--", linewidth=1.5, label="Clean (Reference)")
    axes[1].plot(t_zoom, smoothed[:zoom_pts], color="crimson", linewidth=2.0, label="Smoothed Output")
    axes[1].set_title("Zoomed View (First 2.5 cycles: smoothing effect and slight phase shift)")
    axes[1].set_xlabel("Time (ms)")
    axes[1].set_ylabel("Amplitude")
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend(loc="upper right")

    plt.tight_layout()

    if not args.no_show:
        print("正在弹出对比图窗口，关闭窗口后脚本退出...")
        plt.show()

    return 0


if __name__ == "__main__":
    sys.exit(main())
