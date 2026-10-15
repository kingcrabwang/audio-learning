# audio-learning

学习 **信号处理基础** 和 **音频基础** 的练习仓库。
原则是尽量手写实现、不直接调用现成的库函数 —— 目标是**搞明白原理**，而不是把结果算出来。

## 快速开始

```bash
conda env create -f environment.yml     # 只有第一次需要
conda activate audio-learning
python DSP_Tool/test_myconv.py          # 应看到 4 个 [PASS]、退出码 0
```

## 这是什么

一个边学边写的练习仓库，目前分两部分：

| 目录 | 内容 |
|---|---|
| `audio_base/` | 音频基础：生成正弦 / 扫频信号、读写 wav、画波形 |
| `DSP_Tool/` | 手写的 DSP 工具集：目前有一维卷积 |

设计上刻意把「**纯计算**」和「**读写 / 绘图**」分开：
`waveforms.py` 只算数，不读文件、不写文件、不弹窗口 —— 所以能直接复用，
也才好写测试。它只依赖 numpy，不 import matplotlib 或 soundfile。

想看清各目录里有什么：`ls audio_base DSP_Tool`。

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

### 1. 跑 DSP_Tool 的对照测试

```bash
conda activate audio-learning
python DSP_Tool/test_myconv.py
```

看到 4 个 `[PASS]` 且退出码为 0，就说明卷积实现是对的。
这个脚本也能交给 pytest：`pytest DSP_Tool/test_myconv.py`。

### 2. 生成音频文件、观察波形

脚本用的是**基于「脚本所在目录」**的路径，所以在**任何目录**下运行都行，
**不必**先 `cd` 进 `audio_base/`：

```bash
conda activate audio-learning

python audio_base/make_wave.py sine     # 生成 sine_1K.wav（1kHz 正弦，1 秒）
python audio_base/make_wave.py chirp    # 生成 chirp.wav（200→2000Hz 扫频，3 秒）

python audio_base/read_wave.py          # 读 chirp.wav，打印信息并画波形
```

> `read_wave.py` 会弹出波形窗口，然后**停在那里等你关闭窗口** —— 这不是卡死。
> 没有图形界面时，用 `MPLBACKEND=Agg python audio_base/read_wave.py` 跳过弹窗。

生成的 wav **不在仓库里**（`.gitignore` 排除了 `*.wav`，因为体积大且能随时重新生成）。

想改采样率、时长、频率：

```bash
python audio_base/make_wave.py --help
python audio_base/make_wave.py sine --freq 440 --sr 48000 --duration 2
python audio_base/make_wave.py chirp --f0 50 --f1 5000 --out sweep.wav
```

## 手写实现说明

### `audio_base/waveforms.py`

信号生成函数，全是**纯函数** —— 给同样的输入永远得到同样的输出：

```python
from audio_base import sine_wave, chirp_wave

x = sine_wave(freq=440, sr=48000, duration=2.0, amplitude=0.5)
y = chirp_wave(f0=200, f1=2000, sr=44100, duration=3.0)
```

扫频相位取 `φ(t) = 2π·(f0·t + (f1-f0)/(2T)·t²)`，
对它求导再除以 2π，得到瞬时频率 `f0 + (f1-f0)·t/T` —— 正好从 f0 线性变到 f1。

### `DSP_Tool/myconv.py`

`myconv(x, h, mode)` 手写一维离散卷积，按定义 `y[n] = Σ x[n-m] · h[m]`
用双重循环实现，**不调用** `np.convolve` 或 `scipy.signal` 的卷积函数。

三种模式的语义与 `numpy.convolve` / MATLAB `conv` 对齐（记 N = len(x)，M = len(h)）：

| mode | 含义 | 输出长度 |
|---|---|---|
| `full` | 完整卷积 | N + M − 1 |
| `same` | 取 full 结果的居中段 | max(N, M) |
| `valid` | 只保留完全重叠的部分 | \|N − M\| + 1 |

```python
from DSP_Tool import myconv

myconv([1, 2, 3], [1, 2], mode="full")     # array([1., 4., 7., 6.])
myconv([1, 2, 3], [1, 2], mode="valid")    # array([4., 7.])
```

> 注意 `full` 模式下 `h` **不做翻转** —— `x[n-m]` 这个下标本身已经含有反转，
> 再手动翻一次就变成互相关（correlation）了。这是最容易写错的一处。

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

## 参考

（边学边记，方便以后回来接着看）

- 教材 / 课程：
- 主要查的文档：

## 待改进

- [ ] `myconv` 的 `full` 模式是 O(N·M) 的纯 Python 双重循环，信号长时会明显变慢
      （可以改成 `full_res[k:k+N] += h[k] * x` 这种向量化写法，仍是手写、不调用现成卷积）
- [ ] 清理本地的孤儿文件 `audio_base/test_440hz.wav`（没有任何脚本生成它）
- [ ] 把测试挪到 `tests/` 目录，并给 `audio_base` 也补上 pytest 测试
- [ ] 加 `pyproject.toml`，让这个工具集能 `pip install -e .`
- [ ] 把工作区文件名从 `audio-projects.code-workspace` 改成 `audio-learning.code-workspace`，
      和仓库名保持一致

## 许可证

[MIT](LICENSE) © 2026 kingcrabwang
