# -*- coding: utf-8 -*-
"""
读取 wav 文件，打印基本信息，并画出波形。

用法：
    python read_wave.py                 # 读 dsp-toolbox/samples/chirp.wav
    python read_wave.py 某个文件.wav     # 读指定文件

对应原来的 1readwavefile.py。改了三处：
  1. 装进函数 + 加 __main__ 保护，import 它不再会直接弹图窗；
  2. 默认路径基于【本脚本所在目录】而不是【当前工作目录】，
     所以在任何目录下敲 python 都能找到文件；
  3. 顺手处理了立体声文件（原来遇到双声道会直接报错）。
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

# 本脚本所在目录。默认文件就找这里，不受「当前工作目录」影响。
SAMPLES = Path(__file__).resolve().parent.parent / "samples"

__all__ = ["read_wave", "describe", "plot_waveform", "main"]


def read_wave(path):
    """读 wav，返回 (数据, 采样率)。"""
    data, sr = sf.read(str(path))
    return np.asarray(data), int(sr)


def describe(data, sr):
    """返回描述信息的文本。单独抽出来是为了能被测试直接检查。"""
    duration = len(data) / sr
    return (
        f"采样率：{sr}Hz\n"
        f"持续时间：{duration}s\n"
        f"数据形状：{data.shape}\n"
        f"数据类型：{data.dtype}"
    )


def plot_waveform(data, sr, title="Waveform"):
    """画波形图，横轴单位秒。"""
    t = np.arange(len(data)) / sr
    plt.figure(figsize=(12, 4))          # 画布大小
    plt.plot(t, data)                    # 绘制二维线图
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else SAMPLES / "chirp.wav"

    if not path.exists():
        print(f"找不到文件：{path}")
        print("提示：先运行 python dsp-toolbox/examples/make_wave.py chirp 生成它。")
        return 1

    data, sr = read_wave(path)
    print(f"文件：{path}")
    print(describe(data, sr))

    # 立体声文件没法直接画成一条线，只取第 1 个声道并说明一下
    if data.ndim > 1:
        print(f"注意：这是 {data.shape[1]} 声道文件，波形图只画第 1 个声道。")
        plot_waveform(data[:, 0], sr, title=f"Waveform - {path.name} (ch 1)")
    else:
        plot_waveform(data, sr, title=f"Waveform - {path.name}")

    plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
