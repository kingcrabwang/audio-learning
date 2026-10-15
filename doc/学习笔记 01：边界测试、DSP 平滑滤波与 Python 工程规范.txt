# 学习笔记 01：边界测试、DSP 平滑滤波与 Python 工程规范

> 整理日期：2026-10  
> 关联代码：[`DSP_Tool/myconv.py`], [`examples/demo_moving_average.py`]

---

## 目录
1. [代码结构规范审查与现状点评](#一代码结构规范审查与现状点评)
2. [专题一：边界测试怎么做？（核心思维与实战）](#二专题一边界测试怎么做核心思维与实战)
3. [专题二：滑动平滑滤波与卷积的物理联系](#三专题二滑动平滑滤波与卷积的物理联系)
4. [专题三：Python 工程化与架构设计规范](#四专题三python-工程化与架构设计规范)
5. [专题四：经典踩坑与防御性编程细节](#五专题四经典踩坑与防御性编程细节)

---

## 一、代码结构规范审查与现状点评

目前仓库的架构设计已达到非常优秀的工程水准，主要规范体现在：

```text
audio-projects/
├── audio_base/               # [数据源] 音频信号生成与读写（底层支持）
│   ├── waveforms.py          # 纯数学计算，无副作用
│   ├── make_wave.py          # 命令行参数解析 + 文件写入
│   └── read_wave.py          # 命令行参数解析 + 文件读取 + 绘图
├── DSP_Tool/                 # [核心库] 数字信号处理底层算法
│   ├── myconv.py             # 一维离散卷积（纯算法，只依赖 numpy）
│   └── test_myconv.py        # 10 组自动化测试（覆盖常规、边界、数学性质）
├── examples/                 # [业务层] 综合应用演示
│   └── demo_moving_average.py# 端到端串联 audio_base + DSP_Tool + 可视化
├── README.md                 # 完整的项目说明与快速上手
└── environment.yml           # Conda 环境依赖锁定
```

### 审查亮点：
1. **纯计算与副作用严格分离**：`waveforms.py` 和 `myconv.py` 坚持纯函数设计，不弹窗、不写文件，给测试提供了极佳的确定性。
2. **测试维度极其立体**：`test_myconv.py` 从最初的 5 组扩展到 **10 组全覆盖**，不仅有随机模糊测试，还包括单点边界、等长边界、冲激响应恒等性、极端浮点数与交换律验证。
3. **分层清晰**：新增的滤波演示没有污染 `DSP_Tool`，而是规范地存放在 `examples/` 中。

---

## 二、专题一：边界测试怎么做？（核心思维与实战）

### 1. 核心思想
**Bug 最喜欢藏在合法与非法的交界线上**。普通随机测试（如 100 组长度在 1~16 之间的信号）主要覆盖正常区间，而边界测试是**专门针对交界点精心设计**的。

### 2. 边界测试三原则 (On / Off / In-point)
- **On-point（正好在边界上）**：恰好合法的极限值。例如数组长度 $N=1$。
- **Off-point（刚好跨出边界）**：紧挨着边界的非法值。例如空数组 $N=0$、负数时长 `duration = -1`。
- **In-point（边界内部附近）**：刚好进入正常区的值。例如数组长度 $N=2$。

### 3. 在一维卷积中落地的 5 类关键边界

| 边界类型 | 测试用例设计 | 防范的隐患 |
|---|---|---|
| **最小合法输入** | `len(x)=1` 或 `len(h)=1` | `valid` 切片索引计算在 $N=1$ 时退化为全选，容易发生越界或切出空数组 |
| **等长输入** | `len(x) == len(h)` | `valid` 模式此时**输出只有 1 个点**，最容易出现 off-by-one 差一位的切片错误 |
| **核比信号长** | `len(h) > len(x)` | 保证 `valid` 与 `same` 在不对称输入下的长度与数值依然与 numpy 完全对齐 |
| **数学性质边界** | $x * \delta = x$<br>$x * \delta[n-1] = \text{Shift}(x)$<br>$x * 0 = 0$ | 绕过第三方基准，利用离散卷积固有的数学恒等性与零元特性自证正确性 |
| **浮点极端值** | 大数 $10^{200}$、小数 $10^{-200}$、正负大数抵消 | 验证移位累加算法在大数下溢、溢出及精度截断时的数值稳定性 |

---

## 三、专题二：滑动平滑滤波与卷积的物理联系

### 1. 为什么滑动平均滤波本质就是卷积？
设信号序列为 $x[n]$，滑动窗口大小为 $M$。  
窗口内 $M$ 个采样点的算术平均值为：
$$y[n] = \frac{x[n] + x[n-1] + \dots + x[n-M+1]}{M} = \sum_{k=0}^{M-1} x[n-k] \cdot \frac{1}{M}$$

对比一维离散卷积定义式 $y[n] = \sum_k x[n-k] \cdot h[k]$，两者形式完全一致！  
- **卷积核**：$h = \left[ \frac{1}{M}, \frac{1}{M}, \dots, \frac{1}{M} \right]$（长为 $M$ 的矩形窗 Boxcar Kernel）。
- **物理本质**：滑动平均滤波在频域上是一个**低通滤波器（Low-pass Filter）**，高频毛刺与白噪声被平均抹平，低频正弦波主轮廓被保留。

### 2. 核心代码深度拆解：`h = np.ones(window_size) / float(window_size)`
- `np.ones(window_size, dtype=np.float64)`：生成全为 1.0 的数组（长度为 $M$）。
- `/ float(window_size)`：每个元素都除以 $M$（Numpy 广播除法），得到每个抽头的权重均为 $\frac{1}{M}$。
- **为什么必须除以 $M$（归一化）？**  
  核的所有权重之和必须满足 $\sum h_i = 1$。  
  如果全用 1 不除以 $M$，卷积算出来的就变成局部累加和，滤波后正弦波的幅值会被放大 $M$ 倍；归一化后才能保证原信号的能量和尺度不发生失真。

### 3. 三种模式（`full` / `same` / `valid`）在滤波中的物理表现
- **`mode="same"`**：**最适合音频处理**。滤波后输出点数与原始音频严格一致（输入 1 秒，输出依然 1 秒）。但在信号两端由于卷积窗部分悬空（零填充），会出现轻微的“边缘衰减/下坠”。
- **`mode="valid"`**：**适合严格科学计算**。只保留滑动窗口完全处于信号内部的点，彻底消除边缘伪影，但代价是输出点数变短（减少 $M-1$ 个点）。
- **`mode="full"`**：展示系统从无信号输入、逐步充满滤波器、再到信号结束完全衰减释放的全瞬态过程。

---

## 四、专题三：Python 工程化与架构设计规范

### 1. 命令行解析：为什么抽离 `build_parser()`？
在 [`make_wave.py`](file:///f:/antigravity/audio-projects/audio_base/make_wave.py) 和 [`demo_moving_average.py`](file:///f:/antigravity/audio-projects/examples/demo_moving_average.py) 中，均采用了 `build_parser()` 范式：

```python
def build_parser():
    parser = argparse.ArgumentParser(
        description="...",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,  # 自动在 --help 打印默认值
    )
    parser.add_argument("--sr", type=int, default=44100, help="采样率 (Hz)")
    return parser

def main(argv=None):
    args = build_parser().parse_args(argv)
    ...
```

**三大好处**：
1. **解耦声明与执行**：`build_parser()` 只声明规则，`main()` 负责调度逻辑。
2. **便于单元测试**：测试参数解析时，直接调用 `build_parser().parse_args(["--sr", "48000"])`，无需 hack 系统的 `sys.argv`。
3. **规避隐式类型错误**：命令行读入的参数全是字符串，显式指定 `type=int` / `type=float` 能彻底杜绝 `"44100" * 2 = "4410044100"` 的恶性拼接 Bug。

### 2. 模块导入与包封装哲学
- **不推荐（暴露底层细节）**：`from audio_base.waveforms import sine_wave`
- **推荐（面向包级接口）**：`from audio_base import sine_wave`

**为什么？**
在 [`audio_base/__init__.py`](file:///f:/antigravity/audio-projects/audio_base/__init__.py) 中利用 `__all__ = ["time_axis", "sine_wave", "chirp_wave"]` 定义了公共 API。  
对外隐藏具体文件名，日后重构将 `waveforms.py` 拆解为多个子文件时，外部调用方的代码一行都不用修改。

### 3. `sys`、`os`、`pathlib.Path` 的分工与替代
- **`sys`（Python 解释器自身）**：
  - `sys.path.insert(0, str(PROJECT_ROOT))`：告诉解释器去哪里搜索本地包。
  - `sys.exit(1 if failed else 0)`：返回进程退出码，告知操作系统或 CI 测试通过与否。
- **`pathlib.Path`（现代面向对象路径神器，替代 `os.path`）**：
  - `HERE = Path(__file__).resolve().parent`：不受终端敲命令路径的影响，精准定位当前文件所在目录。
  - `path = HERE / "chirp.wav"`：像搭积木一样跨平台自动处理 Windows `\` 与 Linux `/`。
- **`os`（操作系统底层通信）**：
  - 现代 Python 中路径拼接已被 `Path` 全面取代，`os` 现在主要用于读取系统环境变量（`os.environ`）或进行底层的进程信号控制。

### 4. Matplotlib 绘图乱码根因与双语方案
- **乱码成因**：Matplotlib 默认英文字体（`DejaVu Sans`）不包含 CJK（中日韩）汉字字形，遇到中文便会 fallback 绘制为方块 `□`（Tofu blocks）。
- **工程化决策**：
  - **图表内部（Title, Labels, Legends）**：**统一采用地道英文**（如 `Time (ms)`, `Amplitude`, `Noisy Signal`），彻底消除不同机器上中文字体缺失的风险；
  - **代码与学习部分（Docstring, 注释, 控制台输出）**：**保留中文及中英对照**，保证可读性与学习效率。

---

## 五、专题四：经典踩坑与防御性编程细节

### 1. 异常处理格式化时的“二次崩溃”
在校验输入序列 `_validate_1d(x, h)` 时，原始写法为：
```python
# ❌ 危险写法：如果 x 传入 object()，np.asarray(x) 抛错会掩盖原本的 ValueError
raise ValueError(f"... 当前 dtype={np.asarray(x).dtype}, shape={np.asarray(x).shape}")
```
**防御性改进**：
```python
# ✅ 安全写法：用 try-except 保护诊断信息获取
try:
    arr = np.asarray(x)
    detail = f"dtype={arr.dtype}, shape={arr.shape}"
except Exception:
    detail = f"type={type(x).__name__}"
raise ValueError(f"输入序列 x 必须是非空一维数值序列，当前 {detail}")
```

### 2. 隐式除零产生 `nan/inf` 的隐患
在扫频信号 `chirp_wave` 中存在公式项 `(f1 - f0) / (2 * duration)`。  
若不对输入进行非负校验，当调用者传入 `duration = 0` 时，Numpy 不会抛出异常，而是静默产生 `inf/nan` 并传导到后续的整个计算链路中。必须在函数入口强制拦截 `duration <= 0` 与 `sr <= 0`。
