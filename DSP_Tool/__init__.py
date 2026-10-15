# -*- coding: utf-8 -*-
"""
DSP_Tool —— 学习信号处理与音频时手写的 DSP 工具集。

有了这个 __init__.py，DSP_Tool 才是一个真正的 Python 包，
别的地方才能用 `from DSP_Tool import myconv` 这种写法导入它。
没有它的话，Python 只会把 DSP_Tool 当成一个普通文件夹。
"""
from .myconv import myconv

__all__ = ["myconv"]
