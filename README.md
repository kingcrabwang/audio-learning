# audio-projects

学习 **信号处理基础** 和 **音频基础** 的练习仓库。
原则是尽量手写实现、不直接调用现成的库函数 —— 目标是**搞明白原理**，而不是把结果算出来。

## 快速开始

```bash
conda env create -f environment.yml     # 只有第一次需要
conda activate audio-learning
pip install -e "./dsp-toolbox[dev]"     # 只有第一次需要（可编辑安装子项目）

cd dsp-toolbox && python -m pytest -q    # 运行整个子项目的测试
```

## 这是什么

一个边学边写的练习仓库，目前分两部分：

| 目录 | 内容 |
|---|---|
| `dsp-toolbox/dsp_toolbox/` | 算法实现 |
| `dsp-toolbox/tests/` | 自动化测试 |
| `dsp-toolbox/examples/` | 演示脚本 |
| `dsp-toolbox/csrc/` | 可复用的 C 算法与 C 演示 |
| `dsp-toolbox/docs/` | 算法说明与报告图片 |
| `dsp-toolbox/outputs/` | 演示生成的图片（Git 忽略） |
| `exercises/c/` | 独立的 C 语法与算法练习 |
| `docs/` | 本地学习笔记与可共享的命令速查卡 |

设计上刻意把「**纯计算**」和「**读写 / 绘图**」分开：
`signals.py` 只算数，不读文件、不写文件、不弹窗口 —— 所以能直接复用，
也才好写测试。它只依赖 numpy，不 import matplotlib 或 soundfile。

想看清有哪些内容：`ls dsp-toolbox exercises docs`。

## 本地学习笔记

`docs/学习笔记*.md` 保留在本机，Git 不再跟踪它们；公开仓库中的算法说明、规格和实验报告仍在 `dsp-toolbox/docs/`。`docs/Linux命令速查卡.md` 继续跟踪。

## 环境准备

需要 conda（miniconda 即可）。

```bash
conda env create -f environment.yml     # 按 environment.yml 创建环境
conda activate audio-learning           # 之后每次开工先激活
```

`environment.yml` 里的版本号是**保守写的**（没写死小版本），这样换机器更容易装上。
想换成你当前环境的真实版本：

```bash
conda env export --from-history > environment.yml
```

`--from-history` 只导出你**手动装过**的包，不带一堆自动装上的依赖。

## 怎么跑

### 1. 跑 dsp-toolbox 的对照测试

```bash
conda activate audio-learning
cd dsp-toolbox && pytest
```

看到测试全部通过，说明当前已有模块通过了仓库里的自动检查。若只想检查卷积，可运行 `python -m pytest tests/test_conv.py -q`。
（不再需要任何 `sys.path` 技巧 —— `pip install -e .` 之后直接从包导入。）

### 2. 生成音频文件、观察波形

脚本路径基于「脚本所在目录」，所以**在任何目录**下运行都行；
生成的 wav 统一落到 `dsp-toolbox/samples/`（该目录已被 `.gitignore` 排除）：

```bash
conda activate audio-learning

python dsp-toolbox/examples/make_wave.py sine     # 生成 samples/sine_1K.wav（1kHz 正弦，1 秒）
python dsp-toolbox/examples/make_wave.py chirp    # 生成 samples/chirp.wav（200→2000Hz 扫频，3 秒）

python dsp-toolbox/examples/read_wave.py          # 读 samples/chirp.wav，打印信息并画波形
```

> `read_wave.py` 会弹出波形窗口，然后**停在那里等你关闭窗口** —— 这不是卡死。
> 没有图形界面时，用 `MPLBACKEND=Agg python dsp-toolbox/examples/read_wave.py` 跳过弹窗。

生成的 wav **不在仓库里**（`.gitignore` 排除了 `*.wav`，因为体积大且能随时重新生成）。

想改采样率、时长、频率：

```bash
python dsp-toolbox/examples/make_wave.py --help
python dsp-toolbox/examples/make_wave.py sine --freq 440 --sr 48000 --duration 2
python dsp-toolbox/examples/make_wave.py chirp --f0 50 --f1 5000 --out sweep.wav
```

### 3. 跑滑动平均滤波综合演示

```bash
conda activate audio-learning

python dsp-toolbox/examples/demo_moving_average.py                # 50Hz 正弦 + 噪声，窗口 15 点
python dsp-toolbox/examples/demo_moving_average.py --window 31    # 换窗口大小
python dsp-toolbox/examples/demo_moving_average.py --no-show      # 不弹窗（没有图形界面时用）
```

## 手写实现说明

目前已有卷积、信号生成、FFT/IFFT、整数倍重采样、DTMF 检测、窗函数 FIR 设计、Butterworth IIR 低通，以及 Python 流式 FIR/IIR。C 浮点 FIR 已在 Ubuntu 完成对拍。各模块进度、运行方法与限制见 [dsp-toolbox/README.md](dsp-toolbox/README.md)。

滤波器实验结论和选型表见 [滤波器报告](dsp-toolbox/docs/filter_design_report.md)。当前验证包含合成信号实验，真实音频实验、等波纹设计等后续项目仍需验收。

想直接体验 DTMF 按键识别：

```bash
cd dsp-toolbox
python examples/demo_dtmf.py
python -m dsp_toolbox.dtmf samples/dtmf_demo.wav
```

演示会生成 `samples/dtmf_demo.wav`；该文件被 Git 忽略，不需要提交。

### `dsp-toolbox/dsp_toolbox/signals.py`

信号生成函数，全是**纯函数** —— 给同样的输入永远得到同样的输出：

```python
from dsp_toolbox import sine_wave, chirp_wave

x = sine_wave(freq=440, sr=48000, duration=2.0, amplitude=0.5)
y = chirp_wave(f0=200, f1=2000, sr=44100, duration=3.0)
```

扫频相位取 `φ(t) = 2π·(f0·t + (f1-f0)/(2T)·t²)`，
对它求导再除以 2π，得到瞬时频率 `f0 + (f1-f0)·t/T` —— 正好从 f0 线性变到 f1。

### `dsp-toolbox/dsp_toolbox/conv.py`

`myconv(x, h, mode)` 手写一维离散卷积，**不调用** `np.convolve`
或 `scipy.signal` 的卷积函数。

三种模式的语义与 `numpy.convolve` / MATLAB `conv` 对齐（记 N = len(x)，M = len(h)）：

| mode | 含义 | 输出长度 |
|---|---|---|
| `full` | 完整卷积 | N + M − 1 |
| `same` | 取 full 结果的居中段 | max(N, M) |
| `valid` | 只保留完全重叠的部分 | \|N − M\| + 1 |

```python
from dsp_toolbox import myconv

myconv([1, 2, 3], [1, 2], mode="full")     # array([1., 4., 7., 6.])
myconv([1, 2, 3], [1, 2], mode="valid")    # array([4., 7.])
```

> 注意 `full` 模式下 `h` **不做翻转** —— `x[n-m]` 这个下标本身已经含有反转，
> 再手动翻一次就变成互相关（correlation）了。这是最容易写错的一处。

#### 同一个 full 卷积，留了两个实现

模块里并存两条路径，结果**逐位相同**：

| 函数 | 写法 | 说明 |
|---|---|---|
| `conv_full_vectorized` | 移位累加 | `myconv` 默认走这条，快 |
| `conv_full_reference` | 双重循环 | 照抄定义，慢，留作对照基准 |

向量化的思路是换个提问角度：不问「第 n 个输出点 y[n] 是多少」，
而问「第 m 个抽头 h[m] 给整个输出贡献了什么」。
固定 m 只看 `h[m] * x[n-m]` 这一项，它随 n 平移，
正好是**整条 x 乘上标量 h[m]、搬到输出位置 `[m, m+N-1]`**：

```python
full_res = np.zeros(N + M - 1)
for m in range(M):                      # 只循环 M 次，不是 N 次
    full_res[m : m + N] += h[m] * x
```

**乘加总次数两个实现完全一样**（都是 N·M 次）。快的不是算得少，
而是 Python 层的循环次数从 N 降到 M，剩下的交给 numpy 的 C 代码批量做：
x 长 132300、核长 101 时，Python 层循环从 1300 万次降到 101 次。

之所以还能**逐位相同**：浮点加法不满足结合律，换求和顺序本该改变末位结果；
但这里对某个输出点而言，各项仍按 m 递增的顺序累加，两者都从 0.0 起算，
求和顺序没变。所以测试里用的是 `np.array_equal`（不容忍误差），
而不是通常的 `np.allclose` —— 后者恰好会把「顺序被改动」这类问题放过。

那为什么慢的还要留着？读 `conv_full_reference` 能**直接看到卷积的定义本身**，
不用在脑子里反推向量化那几行在算什么；而且以后优化算法时，
它就是现成的标准答案 —— `test_vectorized_matches_reference` 做的正是这件事。

## 开发约定

一次提交只做一件事；**提交前必须先看 `git status`**：

```bash
git status                     # 我到底改了什么
git add <具体文件或目录>        # 放进暂存区
git status                     # 再确认一次，别把垃圾文件带进去
git commit -m "一句话说清这次做了什么"
git push
```

几件容易踩的事，记在这里省得重复踩：

- **从共享文件夹 `cp` 过来的文件会带「可执行」权限**（宿主机不携带 Unix 权限位）。
  提交前用 `chmod 644 <文件>` 去掉，否则 git 会把它们记成可执行文件。
- **`.gitignore` 不支持行尾注释**。`#` 只有在行首才是注释，
  写在规则后面会让这条规则**静默失效**、且不报任何错。
- 想验证某条忽略规则到底生不生效：`git check-ignore -v <文件路径>`。

## 待改进

- [ ] `full` 卷积仍是 O(N·M)。抽头很长时可以改用 FFT
      （卷积定理：时域卷积 = 频域相乘，复杂度降到 O(L log L)）；
      代价是结果不再逐位相同，只能保证到约 1e-12 的相对误差
- [x] 把测试挪到 `tests/` 目录（已随 monorepo 重构完成）
- [x] 给 `signals.py` 补上 pytest 测试
- [x] 加 `pyproject.toml`，让这个工具集能 `pip install -e .`

## 许可证

[MIT](LICENSE) © 2026 kingcrabwang
