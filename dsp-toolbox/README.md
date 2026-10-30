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
| `filters.py` | 窗函数 FIR 低通/带通/陷波；Butterworth IIR 低通；实数流式 FIR/IIR | 系数与 `firwin` / `butter` 误差 < `1e-12`；输出与 `lfilter` 误差 < `1e-10`；不同分块完全一致 | 当前上述功能已实现，等波纹等后续项待补 |
| `spectrum.py` | 周期图、Welch 等谱估计 | 见 SPEC | 待实现 |
| `adaptive.py` | LMS / NLMS / RLS | 见 SPEC | 待实现 |

“已实现”只表示仓库已有实现与相应测试；详细验收仍以 [SPEC.md](SPEC.md) 为准。

## 滤波器演示与流式处理

在 `dsp-toolbox/` 目录下运行：

```bash
MPLBACKEND=Agg python examples/demo_fir_lowpass.py
MPLBACKEND=Agg python examples/demo_filter_denoise.py
MPLBACKEND=Agg python examples/demo_fir_iir_comparison.py
```

三段演示分别比较 FIR 长度、三种 FIR 对合成干扰的作用，以及 FIR/IIR 的幅频、相位和群延迟。合成数据中的 200 Hz 是目标，50 Hz 和 350 Hz 是干扰。选型表、参数、图片与结论见[滤波器报告](docs/filter_design_report.md)。

这些绘图演示默认把图片放到 `outputs/`，与运行命令时所在的目录无关；该目录由脚本自动创建，且不提交到 Git。窗函数和重采样演示也遵循同一规则：

```bash
MPLBACKEND=Agg python examples/demo_window_leakage.py
MPLBACKEND=Agg python examples/demo_resample_aliasing.py
```

重采样的参考图保存在[文档图片](docs/images/resample_aliasing.png)。

要更新报告中跟随代码提交的图片，可显式指定 `--output`：

```bash
MPLBACKEND=Agg python examples/demo_filter_denoise.py --output docs/images/filter_denoise_comparison.png
MPLBACKEND=Agg python examples/demo_fir_iir_comparison.py --output docs/images/fir_iir_comparison.png
```

在命令行中写的相对 `--output` 路径以当前工作目录为起点；上面两条命令请在 `dsp-toolbox/` 目录执行。

流式 FIR 示例：

```python
from dsp_toolbox.filters import FirFilter, design_fir_lowpass

h = design_fir_lowpass(fc=200, fs=1000, numtaps=101)
fir = FirFilter(h)
first = fir.process([1.0, 0.0, 0.0])
second = fir.process([0.0, 0.0])  # 同一个对象继续使用历史
```

`IirFilter(b, a)` 使用相同的 `process(block)` 方式，并保存输入和输出历史。流式结果是因果输出，保留滤波器延迟；不能直接与居中截取的 `myconv(..., mode="same")` 结果比较。

## C 代码位置与浮点 FIR 对拍

`csrc/src/` 放可复用的 C 实现，`csrc/include/` 放对应头文件，`csrc/examples/` 放演示程序。独立的双指针练习放在仓库根目录 `exercises/c/`，不参与 DSP 库构建。后续计划中的 C 卷积、FFT、LMS 可依次在 `csrc/src/` 和 `csrc/include/dsp/` 增加对应文件；Python 的谱估计、自适应滤波分别在 `dsp_toolbox/spectrum.py`、`dsp_toolbox/adaptive.py` 实现，并在 `tests/` 中添加对应测试。尚未实现的模块不创建空函数。

需要 C 编译器、CMake，以及已经安装开发依赖的 Python。实现位于 `csrc/include/dsp/fir.h`、`csrc/src/fir.c`。

在已激活 `audio-learning` 的 Ubuntu 终端、`dsp-toolbox/` 目录下执行：

```bash
python tests/make_fir_golden.py
cmake -S csrc -B /tmp/dsp-toolbox-build -DPython3_EXECUTABLE="$(command -v python)"
cmake --build /tmp/dsp-toolbox-build
ctest --test-dir /tmp/dsp-toolbox-build --output-on-failure
```

生成器把固定的系数和输入存为 `golden/fir/*.npy`。测试读取这些文件，通过 ctypes 把相同数据传给 C 共享库。参考输出由相同 float32 数据转换为双精度后交给 `lfilter` 计算。

验收要求：C 浮点输出与参考逐点误差 < `1e-6`；按 1/7/64/512 点分块时，与 C 整段处理输出完全相同。`fir_create` 分配并初始化，`fir_process` 使用已有内存，`fir_destroy` 释放。

CTest 注册一个 `fir_golden`，内部运行 21 个 pytest 用例。修改 `CMakeLists.txt` 后要保存并重新配置，可用 `ctest --test-dir /tmp/dsp-toolbox-build -N` 检查注册结果。Ubuntu CTest 已由用户确认通过；尚未测量 C 性能或进行内存泄漏检测。

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

目前测试覆盖卷积、信号生成、FFT/IFFT、整数倍抽取与内插、DTMF，以及 FIR/IIR 系数、逐样本输出和分块一致性。C FIR 单独通过 CTest 验收；普通 `python -m pytest -q` 未提供 C 库路径时，会跳过 C 的 21 个用例，跳过不代表 C 已通过。

DTMF 的“16 键 × 3 档信噪比”与当前滤波器去噪演示都使用合成数据，尚不能当作真实录音中的性能结论。最新回归环境与结果记录在[滤波器报告](docs/filter_design_report.md)。

## 已知限制

- FFT 输入长度必须是 2 的幂，不会自动补零。
- 重采样目前只提供整数倍 `decimate(x, M)` 与 `interpolate(x, L)`；尚未提供通用的任意 `up/down` 公共接口。
- DTMF 文件读取只支持未压缩的 **16-bit PCM WAV**。立体声会先取两声道平均；两声道相位相反时可能互相抵消。
- 连续按键检测按约 40 ms 窗、10 ms 步长处理，重复同一按键需要有可识别的间隔；阈值针对演示与合成测试，尚未用真实电话录音校准。
- FIR 设计目前限定奇数长度；低通支持 Hamming/Hann，带通和陷波使用 Hamming。等波纹设计尚未实现。
- `design_iir` 当前只支持 `btype="low"`、`ftype="butter"`，不提供高通、带通或其他 IIR 类型。
- FIR 陷波的 `q` 定义名义宽度 `f0/q`，不保证实际 −3 dB 带宽；自动长度取约 `4*fs/带宽` 并调整成奇数，窄带低频时可能很长。
- 流式滤波接口只接受有限实数序列。当前 IIR 采用直接型递推；高阶或极端截止频率需另行验证数值表现，未实现二阶节结构。
- 尚未做实时性优化；性能数据仍待补充到 `docs/bench.md`。

公式与边界说明见[公式速查](docs/formulas.md)、[限制与易错点](docs/failure_modes.md)。个人学习笔记保留在本机 `../docs/`，不进入公开仓库。
