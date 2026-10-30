# dsp-toolbox 需求规格书（W1 → W7）

> 对应计划表：`2-周计划表（36周）` 的 W1–W7（阶段一 经典DSP内核，共 84 小时）
> 用途：这份文件应当作为 `dsp-toolbox/SPEC.md` 放进仓库，边写边对着勾

---

## 0. 先回答「写到什么程度」

**它不是产品，是证据。**

dsp-toolbox 存在的唯一理由，是在面试里替你回答三个问题：

| 面试官会问 | 它替你回答什么 |
|---|---|
| 「这些算法是你自己写的，还是调库？」 | 独立实现 + 与参考库**逐点对拍**的测试 |
| 「你能把它落到端侧吗？」 | 有 **C 版本**，能脱离 Python 跑 |
| 「你懂工程吗？」 | 能一键构建、一键测试、有性能与精度数据 |

所以**完成度不由算法数量决定**。再加第 9、第 10 个算法，价值远小于把前 8 个补上 C 版和对拍测试。

### 三条硬线（全满足才算写完）

- **硬线 A**：每个算法函数内部**不许调用** numpy / scipy 的对应实现。numpy 只能在测试里当参照物，或做最基础的数组运算。
- **硬线 B**：每个算法都有 `tests/test_*.py`，断言是**逐点数值对比**（`np.allclose(..., atol=...)`），不是「跑通了没报错」。
- **硬线 C**：`README.md` 能让一个陌生人 5 分钟内完成：`pip install -e .` → `pytest` 全绿 → 跑通一个 demo。

> 计划表里每周「验收标准」那一列，就是这份需求的压缩版。例如 W1 写的「pytest 全绿」= 下面的 §3.1。

---

## 1. 范围（锁死，不再扩）

只做 W1–W7 教过的 **8 个算法族**，不多不少：

| # | 模块文件 | 独立实现的东西 | 对拍对象 |
|---|---|---|---|
| 1 | `conv.py` | 线性卷积、互相关（不调库） | `np.convolve`、`scipy.signal.correlate` |
| 2 | `signals.py` | 正弦/复指数合成、白噪声、按 SNR 加噪 | 统计性质自验（无直接对照） |
| 3 | `fft.py` | radix-2 迭代版 FFT / IFFT | `np.fft.fft` |
| 4 | `filters.py` | FIR 设计（窗函数/等波纹）+ IIR 设计（双线性）+ **流式** FIR/IIR | `scipy.signal.firwin` / `butter` / `lfilter` |
| 5 | `resample.py` | 整数倍抽取、内插、多相分解 | `scipy.signal.resample_poly` |
| 6 | `spectrum.py` | 自相关、周期图、Welch、AR（Yule-Walker） | `scipy.signal.periodogram` / `welch` / `correlate` |
| 7 | `adaptive.py` | LMS / NLMS / RLS | 已知系统辨识（无直接对照，见 §3.7） |
| 8 | `dtmf.py` | 基于 Goertzel 的 DTMF 双音检测（端到端 demo） | 自建 16 键测试集 |

### 明确不做（防止无限膨胀）

- ❌ 波束成形 / DOA / AEC / ANC / 降噪 —— 这些是 W12 之后的**独立项目**，各自建库，不塞进 dsp-toolbox
- ❌ 任何深度学习
- ❌ GUI、Web、命令行以外的交互
- ❌ pip 公开发布、多语言（只 Python + C）
- ❌ 「和 scipy 一样全」—— 不追覆盖面，只追**这 8 个都写透**

---

## 2. 版本演进（「程度」的刻度尺）

| 版本 | 时间 | 加入内容 | 代码量级 | 完成判定 |
|---|---|---|---|---|
| **v0.1** | W1 末 | `conv` + `signals` + pytest 骨架 + Git 仓库 + README | Python ~200 行 | `pytest` 全绿；`np.convolve` 100 组对拍通过 |
| **v0.5** | W4 末 | + `fft` + `filters`（设计 + 流式）+ `resample` | Python ~700 行 | 这 5 个模块的 L2（对拍）全部通过 |
| **v0.9** | W6 末 | + `spectrum` + `adaptive` + `dtmf` + `docs/` | Python ~1300 行 | 8 个模块 L2 全通过；陌生人能按 README 跑通 |
| **v1.0** | W7 末 | + `csrc/`（C 版 4 个核心算法，浮点 + Q15）+ CMake + benchmark + `golden/` | C ~800 行 | C 版与 Python 版逐点误差 < 1e-12；性能表 4 行数据齐全 |

**注意 v1.0 是终点。** 之后不再往这个仓库加东西（W12+ 的项目会**复用**它，但不合并进来）。

---

## 3. 每个模块的验收标准（可直接抄进 pytest）

### 3.1 `conv.py`
```python
myconv(x, h, mode="full")     # mode: full | same | valid
mycorr(x, y, mode="full")
```
- **对拍**：100 组随机序列（长度 1–512，三种 mode 全覆盖），逐点 `|Δ| < 1e-12`
- **边界用例**（必须写）：`h` 比 `x` 长；`x` 或 `h` 长度 1；空数组 → 抛 `ValueError`；`mode` 传错 → 抛 `ValueError`
- **L3 可讲**：白板写出卷积和式；解释「卷积为什么是 LTI 的完整描述」；解释 `same` 模式在偶数长度下的取整约定

### 3.2 `signals.py`
```python
sine(f, fs, dur, amp=1.0, phase=0.0)
complex_exp(f, fs, dur)
add_noise(x, snr_db, seed=None)
```
- **验收**：同一 `seed` 完全可复现；`add_noise` 实测 SNR 与设定值误差 **< 0.1 dB**（用 1000 次蒙特卡洛取均）
- **L3 可讲**：解释复指数为什么是 LTI 系统的特征函数

### 3.3 `fft.py`
```python
myfft(x)     # radix-2 迭代版
myifft(X)
```
- **对拍**：长度 8/16/…/4096 的复随机序列，相对误差 **< 1e-10**
- **边界用例**：长度 1；单位冲激（DFT 应为全 1）；直流信号（频谱除 DC 外全 0）；**非 2 的幂必须抛 `ValueError`**（不许静默补零）
- **L3 可讲**：默画 8 点蝶形流图；推出 O(N log N)；说清旋转因子的对称性与周期性

### 3.4 `filters.py`（最容易写错的一个）
```python
design_fir_lowpass(fc, fs, numtaps, window="hamming")
design_fir_bandpass(f1, f2, fs, numtaps)
design_fir_notch(f0, fs, q)          # W4 用 50Hz 工频陷波
design_iir(fc, fs, order=4, btype="low", ftype="butter")

FirFilter(h).process(block)           # 流式，内部环形缓冲
IirFilter(b, a).process(block)        # 流式
```
- **设计系数对拍**：与 `scipy.signal.firwin` / `butter` 误差 < 1e-12
- **输出对拍**：与 `scipy.signal.lfilter` 逐样本误差 < 1e-10
- **★ 核心验收（流式正确性）**：把同一段信号分别按 **1 / 7 / 64 / 512 样本一块**喂进 `process()`，结果必须与一次性处理**逐点完全相同**（`atol=0`）。这一条不过，说明环形缓冲边界写错了 —— 这是面试官最爱追问的点
- **L3 可讲**：解释「音频为什么偏爱线性相位 FIR」；双线性变换为什么无混叠但有频率翘曲、预畸变怎么用

### 3.5 `resample.py`
```python
decimate(x, M)        # 先抗混叠低通，再抽取
interpolate(x, L)     # 先插零，再去镜像低通
```
- **对拍**：与 `scipy.signal.resample_poly` 误差 **< 1e-10**
- **必须产出的图**：同一信号「直接抽取 vs 先低通再抽取」的频谱对比图（这张图是面试可讲的材料）
- **L3 可讲**：解释多相分解为什么能把滤波搬到低采样率侧、省下多少乘加

### 3.6 `spectrum.py`
```python
autocorr(x, maxlag)
periodogram(x, fs, window="hann")
welch(x, fs, nperseg=256, noverlap=None)
ar_yule_walker(x, order)
```
- **对拍**：周期图 / Welch / 自相关与 scipy 误差 < 1e-10
- **统计验收（这条比数值对拍更重要）**：用 100 次蒙特卡洛，断言 **Welch 的谱方差 < 周期图方差的 0.7 倍**（用真实的统计性质证明你懂它为什么存在）
- **应用验收**：AR 模型提取元音共振峰 F1/F2，落在合理区间（男声 F1 约 250–800 Hz）
- **L3 可讲**：推导维纳-辛钦定理；说清 Welch 降方差的代价是频率分辨率

### 3.7 `adaptive.py`
```python
lms(x, d, mu, order)      -> (w_hist, e, y)
nlms(x, d, mu, order, eps=1e-6)
rls(x, d, lam, order, delta=1.0)
```
没有等价的 scipy 函数，所以用**已知系统**验证：
- **验收 A**：随机生成一个 16 阶 FIR 系统，用 LMS 做辨识；2000 迭代后 `10*log10(MSE) < -60 dB`
- **验收 B（反向验证收敛条件）**：把 μ 设成 `2/λ_max` 以上，**断言它会发散** —— 证明你写的收敛条件是对的，而不是碰巧能跑
- **验收 C**：同样条件下 NLMS 在输入幅度突变时比 LMS 稳（构造一个幅度跳变 10 倍的输入，比较两者的失调）
- **L3 可讲**：白板推导 `w[n+1] = w[n] + μe[n]x[n]`；给出 `0 < μ < 2/λ_max`；解释 NLMS 归一化解决了什么

### 3.8 `dtmf.py`（端到端 demo，纯为「能演示」而存在）
```python
detect_dtmf(path) -> str          # 返回按键序列，如 "13800138000"
```
- **验收**：自建 16 键 ×  3 档 SNR 的测试集，识别率 100%
- **必须可命令行运行**：`python -m dsp_toolbox.dtmf samples/tone.wav`

---

## 4. 三级完成度（每个模块都要走完 L1 → L2）

| 级别 | 含义 | 判定动作 |
|---|---|---|
| **L1 能跑** | 给个输入能出个结果 | 函数不报错 |
| **L2 对拍** | 与参考实现逐点对比达标 + 边界用例覆盖 | `pytest` 该模块全绿，含空/单点/非 2 的幂/超长四类边界 |
| **L3 可讲** | 能白板推导；能说出「什么时候这个方法会失效」；有性能数字 | 不看资料讲 3 分钟；`docs/failure_modes.md` 有对应条目 |

**要求**：8 个模块全部过 L2；**核心 4 个（conv / fft / filters / adaptive）必须过 L3 且有 C 版**。

---

## 5. 仓库结构

```
dsp-toolbox/
├── SPEC.md                     ← 本文件
├── README.md
├── pyproject.toml
├── dsp_toolbox/
│   ├── __init__.py
│   ├── conv.py  signals.py  fft.py  filters.py  resample.py  dtmf.py
│   └── spectrum.py  adaptive.py      ← 后续实现，当前不放空文件
├── tests/
│   ├── test_conv.py  test_fft.py  test_filters.py  test_resample.py  test_dtmf.py
│   └── test_spectrum.py  test_adaptive.py  ← 随新模块一起添加
├── csrc/                       ← v1.0（W7）加入
│   ├── CMakeLists.txt
│   ├── include/dsp/fir.h
│   ├── src/fir.c
│   ├── examples/demo_gain.c
│   └── include/dsp/ 与 src/     ← 后续增加 conv、fft、lms
├── golden/                     ← v1.0 加入：C/Python 对拍用的固定输入与期望输出
├── docs/
│   ├── formulas.md             ← 公式速查（面试前直接翻这个）
│   ├── bench.md                ← 性能与精度对比表
│   ├── failure_modes.md        ← 每个算法什么时候会失效
│   └── images/                 ← 报告引用、随代码提交的图片
├── examples/                   ← 演示脚本；导入时不运行
└── outputs/                    ← 演示输出；自动创建，不提交
```

仓库根目录另设 `exercises/c/` 保存独立的 C 语法与算法练习；它们不编入 DSP 库。个人学习笔记留在根目录 `docs/`，由 `.gitignore` 排除。`spectrum.py`、`adaptive.py` 和其余 C 算法的位置已经约定，实施时再添加真实代码与测试。

---

## 6. README 必须包含的 6 项

1. **一句话定位**：「手写 DSP 算法实现 + 与 numpy/scipy 逐点对拍的测试集」，别写成「高性能 DSP 库」
2. **环境与安装**：3 条命令以内（含 `pip install -e .` 与 `pytest`）
3. **30 秒 demo**：一段可直接复制运行的代码
4. **模块清单表**：`模块 | 独立实现 | 对拍对象 | 误差达标值` —— **这张表是面试官第一眼看的地方**
5. **测试怎么跑 + 当前覆盖情况**
6. **已知限制**：明确写出「没做什么」（例如「FFT 仅支持 2 的幂」「未做实时性优化」）

> 「已知限制」这一节最容易被忽略，但它是**加分项**：主动划边界说明你知道自己写了什么。

---

## 7. C 版本（W7）的具体要求

### 接口风格（C99，头文件与实现分离，不许用全局变量）
```c
typedef struct fir_state fir_state_t;

fir_state_t *fir_create(const float *h, int nh);
void         fir_process(fir_state_t *s, const float *in, float *out, int n);
void         fir_destroy(fir_state_t *s);
```
- **允许**在 `create` 里 `malloc`；**严禁**在 `process` 里分配内存（这是音频实时处理的铁律，面试必问）
- 四个核心：`conv`、`fft`、`fir`、`lms`

### 构建
```bash
cmake -B build && cmake --build build && ctest --test-dir build
```
一条命令能跑通，README 里写清楚。

### 对拍方式（关键）
不要用随机数在 C 和 Python 里各生成一遍。**Python 侧把输入与期望输出存成 `golden/*.npy`，C 测试读同一份数据逐点断言**：
- C 浮点版 vs Python：逐点 `|Δ| < 1e-12`（float 版本放宽到 1e-6）
- C Q15 定点版 vs Python：给出**位宽、饱和策略、以及误差/SNR 损失**

### 性能表（`docs/bench.md`）
| 算法 | 输入规模 | numpy (ms) | C 浮点 (ms) | C-Q15 (ms) | 加速比 | 精度损失 |
|---|---|---|---|---|---|---|
| conv | 1×10⁶ × 64 | | | | | |
| fft | 4096 × 1000 | | | | | |
| fir | 10⁶ 样本 | | | | | |
| lms | 16 阶 × 10⁵ 迭代 | | | | | |

**「加速比」和「精度损失」必须同时给** —— 只给加速比不给精度损失，等于没做定点化。

---

## 8. 面试官会怎么看这个仓库

| 他的动作 | 你希望他看到的 |
|---|---|
| 先看 README 的模块清单表 | 一眼就知道你真写了，而不是 fork 的 |
| 翻 `tests/` | 有逐点对拍 + 边界用例 = 加分；只有 `assert x is not None` = 减分 |
| 问「为什么不直接用 scipy」 | 答：为了理解算法内部，也为了能在没有 Python 的端侧复用（顺势引出 C 版） |
| 问「最难的地方在哪」 | 能说出具体一个坑（流式 FIR 的分块边界 / FFT 位反转 / LMS 的 μ 上限） |
| 打开 `csrc/fir.c` | `process()` 里**没有** `malloc` —— 这一条是嵌入式岗的硬门槛 |

---

## 9. 什么时候可以停

**停的标准**：8 个模块全过 L2，核心 4 个过 L3 且有 C 版，README 能让陌生人跑通 → **停手**。

- 不要加：新算法、GUI、Web、更多语言、pip 发布
- 时间不够时的**砍单顺序**：`dtmf` → `resample` → `spectrum` → `signals`
- **绝对不能砍**：`conv` / `fft` / `filters` / `adaptive` + 它们的 C 版

---

## 附：为什么 W7 要单独花一整周做 C 重写

因为「端侧 C++ 落地」是你锁定的三件事之一，它**不能只当每周 2 小时的工程线**。W7 这一周要完成的是一次完整的能力切换：

1. 把 Python 里靠 numpy 向量化省掉的内存管理、索引边界、类型转换**全部显式写出来**
2. 用 CMake 把散落的 `.c` 组织成可复用的静态库
3. 用 `golden/` 数据证明 C 版与 Python 版**逐点一致**（而不是「看起来波形差不多」）
4. 拿到第一组性能与精度数字，成为后面 AEC / ANC / 端侧部署的性能基线

这一周结束后，你手里就有了「同一个算法，Python 版和 C 版我都能写，且能量化差异」的证据 —— 这正是音频算法岗最稀缺的那一格。
