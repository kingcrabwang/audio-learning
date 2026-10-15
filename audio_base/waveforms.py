# -*- coding: utf-8 -*-
"""
信号生成函数 —— 只负责算，不碰文件和窗口。

设计原则：把「计算信号」和「写文件 / 画图」分开。
本模块里全是纯函数：给同样的输入永远得到同样的输出，
不读文件、不写文件、不弹窗口。好处有两个：

  1. 别处可以直接 import 复用，不用把整个脚本从头跑一遍；
  2. 没有副作用，所以可以放心地写测试。

原来的 creatWAV.py 和 2creatwavefile.py 把「算信号」和「写文件」混在一起，
想单独要一个正弦波数组都没办法，只能运行整个脚本。
"""
import numpy as np

__all__ = ["time_axis", "sine_wave", "chirp_wave"]


def time_axis(duration, sr):
    """生成时间轴，单位秒。

    Parameters
    ----------
    duration : float
        时长，秒。
    sr : int
        采样率，Hz。

    Returns
    -------
    np.ndarray
        长度为 int(round(sr * duration))，从 0 开始、不含终点。
    """
    n = int(round(sr * duration))
    return np.linspace(0.0, duration, n, endpoint=False)


def sine_wave(freq, sr, duration, amplitude=0.5):
    """生成单频正弦波：amplitude * sin(2πft)。

    Parameters
    ----------
    freq : float      频率，Hz
    sr : int          采样率，Hz
    duration : float  时长，秒
    amplitude : float 振幅，默认 0.5
    """
    t = time_axis(duration, sr)
    return amplitude * np.sin(2 * np.pi * freq * t)


def chirp_wave(f0, f1, sr, duration, amplitude=0.5):
    """生成线性扫频信号（chirp），瞬时频率从 f0 线性升到 f1。

    相位取 φ(t) = 2π·(f0·t + (f1-f0)/(2T)·t²)。
    对相位求导再除以 2π，就得到瞬时频率 f0 + (f1-f0)·t/T ——
    正好是从 f0 线性变到 f1，可以自己代入 t=0 和 t=T 验证。

    Parameters
    ----------
    f0, f1 : float    起始 / 终止频率，Hz
    sr : int          采样率，Hz
    duration : float  时长，秒
    amplitude : float 振幅，默认 0.5
    """
    t = time_axis(duration, sr)
    phase = 2 * np.pi * (f0 * t + (f1 - f0) / (2 * duration) * t ** 2)
    return amplitude * np.sin(phase)
