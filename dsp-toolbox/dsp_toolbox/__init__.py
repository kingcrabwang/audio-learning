# -*- coding: utf-8 -*-
"""
dsp_toolbox —— 手写 DSP 算法工具箱（audio-projects 的子项目，对应 W1–W7）。

有了这个 __init__.py，dsp_toolbox 才是一个真正的 Python 包，
别处才能 `from dsp_toolbox import myconv` 这样导入它。

注意这里【只】导入纯计算模块（它们只依赖 numpy）。
故意不在这里导入 examples/ 里的脚本，因为它们依赖 matplotlib / soundfile ——
否则你只想用一下 sine_wave，却被迫先装齐画图和写文件的库。
"""
from .conv import myconv
from .signals import chirp_wave, sine_wave, time_axis

__all__ = ["myconv", "time_axis", "sine_wave", "chirp_wave"]
