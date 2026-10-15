# -*- coding: utf-8 -*-
"""
audio_base —— 音频基础练习包。

有了这个 __init__.py，audio_base 才是一个真正的 Python 包，
别处才能 `from audio_base import sine_wave` 这样导入。

注意这里【只】导入 waveforms（它只依赖 numpy）。
故意不在这里导入 read_wave / make_wave，因为它们依赖 matplotlib / soundfile。
否则你只想用一下 sine_wave，却被迫先装齐画图和写文件的库。
"""
from .waveforms import chirp_wave, sine_wave, time_axis

__all__ = ["time_axis", "sine_wave", "chirp_wave"]
