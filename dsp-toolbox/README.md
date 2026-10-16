# dsp-toolbox

手写 DSP 算法实现 + 与 numpy/scipy **逐点对拍**的测试集。

> 子项目，隶属于 [audio-projects](../README.md)。完整需求见 [SPEC.md](SPEC.md)。
> 它要证明的：这些算法是我自己写的、能与参考实现逐点对上、并且能落到 C。

## 环境与安装

```bash
pip install -e ".[dev]"
pytest
```

## 30 秒 demo

```python
from dsp_toolbox import myconv
myconv([1, 2, 3], [1, 2], mode="full")     # array([1., 4., 7., 6.])
```

## 模块清单

| 模块 | 独立实现 | 对拍对象 | 达标值 |
|---|---|---|---|
| `conv.py` | 线性卷积、互相关 | `np.convolve` / `scipy.signal.correlate` | 待填 |
| `signals.py` | 正弦 / 复指数 / 按 SNR 加噪 | 统计性质自验 | 待填 |
| `fft.py` | radix-2 迭代 FFT/IFFT | `np.fft.fft` | 未开工 |
| `filters.py` | FIR/IIR 设计 + 流式实现 | `scipy.signal.firwin` / `butter` / `lfilter` | 未开工 |
| `resample.py` | 抽取 / 内插 / 多相分解 | `scipy.signal.resample_poly` | 未开工 |
| `spectrum.py` | 自相关 / 周期图 / Welch / AR | `scipy.signal.periodogram` / `welch` | 未开工 |
| `adaptive.py` | LMS / NLMS / RLS | 已知系统辨识（无直接对照） | 未开工 |
| `dtmf.py` | Goertzel DTMF 双音检测 | 自建 16 键测试集 | 未开工 |

## 测试

```bash
pytest            # 当前：conv 10 组（含边界、数学性质、向量化与参考实现逐位一致）
```

## 已知限制

- FFT 只支持 2 的幂（不允许静默补零）
- 未做实时性优化；性能数据见 `docs/bench.md`（待填）
- 各算法的失效场景见 `docs/failure_modes.md`（待填）
