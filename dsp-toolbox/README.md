# dsp-toolbox

手写 DSP 算法实现，并用 NumPy / SciPy 或独立计算方法检查结果。这个仓库用于学习与验证，不以实时性能为目标。

> 这是 [audio-projects](../README.md) 的子项目；计划中的完整规格见 [SPEC.md](SPEC.md)。

## 环境与安装

在 `dsp-toolbox/` 目录下运行：

```bash
pip install -e ".[dev]"
python -m pytest -q
```

如果从仓库根目录开始，也可以先按根目录的 `environment.yml` 创建并激活 `audio-learning` 环境。

## 30 秒体验

```python
from dsp_toolbox import myconv

print(myconv([1, 2, 3], [1, 2], mode="full"))  # [1. 4. 7. 6.]
```

## 模块进度

| 模块 | 当前实现 | 核对方式 / 达标条件 | 状态 |
|---|---|---|---|
| `conv.py` | 一维卷积及参考实现 | 与 `np.convolve` 对拍、检查边界 | 已实现 |
| `signals.py` | 时间轴、正弦波、线性扫频 | 样本时刻与公式测试 | 已实现 |
| `fft.py` | radix-2 迭代 FFT / IFFT | 与 `np.fft.fft` / `ifft` 对拍 | 已实现 |
| `resample.py` | 抗混叠抽取、插零内插 | 与 `scipy.signal.resample_poly` 对拍，误差 < `1e-10` | 已实现 |
| `dtmf.py` | Goertzel 双音检测、连续按键识别 | 合成 16 键 × 3 档信噪比测试及命令行测试 | 已实现 |
| `filters.py` | FIR / IIR 设计与流式实现 | 见 SPEC | 待实现 |
| `spectrum.py` | 周期图、Welch 等谱估计 | 见 SPEC | 待实现 |
| `adaptive.py` | LMS / NLMS / RLS | 见 SPEC | 待实现 |

“已实现”只表示仓库已有实现与相应测试；详细验收仍以 [SPEC.md](SPEC.md) 为准。

## 运行 DTMF 演示

在 `dsp-toolbox/` 目录下运行：

```bash
python examples/demo_dtmf.py
python -m dsp_toolbox.dtmf samples/dtmf_demo.wav
python -m pytest tests/test_dtmf.py -q
```

第一条命令会合成按键序列 `13800138000`，保存到 `samples/dtmf_demo.wav`，并打印识别结果。第二条命令读取这个文件，只打印识别出的按键。生成的 WAV 被 `.gitignore` 忽略；换台机器重新运行演示即可生成。

例如，单独识别一段内存中的按键音：

```python
from dsp_toolbox.dtmf import detect_key

key = detect_key(samples, sr=8000)  # 证据不足时返回 None
```

这里的 `samples` 需要由你提供，是一维实数音频数组。`detect_key` 从 `dsp_toolbox.dtmf` 导入，不需要写进包的 `__init__.py`。

## 测试

```bash
python -m pytest -q
```

目前测试覆盖卷积、信号生成、FFT/IFFT、整数倍抽取与内插，以及 DTMF 的 16 个键、合成噪声、重复按键和 WAV 命令行调用。DTMF 的“16 键 × 3 档信噪比”是**合成测试集**的结果，尚不能当作真实录音中的识别率。

## 已知限制

- FFT 输入长度必须是 2 的幂，不会自动补零。
- 重采样目前只提供整数倍 `decimate(x, M)` 与 `interpolate(x, L)`；尚未提供通用的任意 `up/down` 公共接口。
- DTMF 文件读取只支持未压缩的 **16-bit PCM WAV**。立体声会先取两声道平均；两声道相位相反时可能互相抵消。
- 连续按键检测按约 40 ms 窗、10 ms 步长处理，重复同一按键需要有可识别的间隔；阈值针对演示与合成测试，尚未用真实电话录音校准。
- 尚未做实时性优化；性能数据仍待补充到 `docs/bench.md`。

DTMF 的推导和本项目代码路径见 [学习笔记 03：DTMF 与 Goertzel 检测](../docs/学习笔记%2003：DTMF%20与%20Goertzel%20检测.md)。