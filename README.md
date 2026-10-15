# audio-learning

学习 **信号处理基础** 和 **音频基础** 的练习仓库。
原则是尽量手写实现、不直接调用现成的库函数，目的是把原理搞明白。

## 目录结构

```
audio-learning/
├── README.md                       # 本文件
├── environment.yml                 # conda 环境定义
├── .gitignore                      # 哪些文件不进版本管理
├── .gitattributes                  # 统一换行符、声明二进制文件
├── audio-projects.code-workspace   # VS Code 多根工作区配置
│
├── audio_base/                     # 音频基础练习
│   ├── __init__.py                 # 让它成为一个真正的 Python 包
│   ├── waveforms.py                # 信号生成函数：正弦 / 扫频（只算数，不碰文件）
│   ├── make_wave.py                # 命令行入口：生成波形并写成 wav
│   └── read_wave.py                # 读 wav → 打印参数 → 画波形
│
└── DSP_Tool/                       # 手写的 DSP 工具集
    ├── __init__.py                 # 让它成为一个真正的 Python 包
    ├── myconv.py                   # 手写一维卷积 myconv(x, h, mode)
    └── test_myconv.py              # 与 numpy.convolve 对照测试
```

> `audio_base/` 里的 `chirp.wav`、`sine_1K.wav` 等音频文件**不在仓库里**
> （`.gitignore` 排除了 `*.wav`，因为它们体积大且能随时重新生成）。
> 用下面的 `make_wave.py` 自己生成。

## 环境准备

需要 conda（miniconda 即可）。

```bash
# 第一次：按 environment.yml 创建环境
conda env create -f environment.yml

# 之后每次开工：激活环境
conda activate audio-learning
```

## 怎么跑

### 1. 跑 DSP_Tool 的对照测试

```bash
conda activate audio-learning
python DSP_Tool/test_myconv.py
```

看到 4 个 `[PASS]` 且退出码为 0，就说明卷积实现是对的。
这个脚本也能交给 pytest：`pytest DSP_Tool/test_myconv.py`。

### 2. 生成音频文件、观察波形

这些脚本用的是**基于「脚本所在目录」**的路径，
所以在**任何目录**下运行都行，**不必**先 `cd` 进 `audio_base/`：

```bash
conda activate audio-learning

python audio_base/make_wave.py sine     # 生成 sine_1K.wav（1kHz 正弦，1 秒）
python audio_base/make_wave.py chirp    # 生成 chirp.wav（200→2000Hz 扫频，3 秒）

python audio_base/read_wave.py          # 读 chirp.wav，打印信息并画波形
```

想改采样率、时长、频率这些参数：

```bash
python audio_base/make_wave.py --help
python audio_base/make_wave.py sine --freq 440 --sr 48000 --duration 2
python audio_base/make_wave.py chirp --f0 50 --f1 5000 --out sweep.wav
```

如果所在的机器没有图形界面，`plt.show()` 会卡住或报错，可以临时跳过弹窗：

```bash
MPLBACKEND=Agg python audio_base/read_wave.py
```

## audio_base/waveforms.py

信号生成函数。它们都是**纯函数** —— 给同样的输入永远得到同样的输出，
不读文件、不写文件、不弹窗口。所以可以直接复用，不用把整个脚本跑一遍：

```python
from audio_base import sine_wave, chirp_wave

x = sine_wave(freq=440, sr=48000, duration=2.0, amplitude=0.5)
y = chirp_wave(f0=200, f1=2000, sr=44100, duration=3.0)
```

扫频信号的相位取 `φ(t) = 2π·(f0·t + (f1-f0)/(2T)·t²)`，
对它求导再除以 2π 就是瞬时频率 `f0 + (f1-f0)·t/T` —— 正好从 f0 线性变到 f1。

## DSP_Tool/myconv.py

`myconv(x, h, mode)` 手写一维离散卷积，按定义 `y[n] = Σ x[n-m] · h[m]`
用双重循环实现，**不调用** `np.convolve` 或 `scipy.signal` 的卷积函数。

三种模式的语义与 `numpy.convolve` / MATLAB `conv` 对齐
（记 N = len(x)，M = len(h)）：

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

## 提交约定

一次提交只做一件事；提交前必须先看 `git status`：

```bash
git status                     # 我到底改了什么
git add <具体文件或目录>        # 放进暂存区
git status                     # 再确认一次，别把垃圾文件带进去
git commit -m "一句话说清这次做了什么"
git push
```

## 待改进（TODO）

- [x] `audio_base/` 脚本加 `if __name__ == "__main__":` 保护并函数化，避免 `import` 就执行
- [x] 文件名不符合 PEP 8：数字开头**永远无法被 import** → 已改成 `read_wave.py` / `make_wave.py`
- [x] 脚本的相对路径改成基于「脚本所在目录」，任何工作目录下都能跑
- [x] `2creatwavefile.py` 和 `creatWAV.py` 功能重叠 → 已合并成 `make_wave.py`
- [x] 采样率、时长、频率等改成命令行参数（`--help` 可查）
- [ ] `.code-workspace` 里写死的解释器路径 `/home/kincrab/miniconda3/...` 建议删掉，交给 VS Code 自动识别 conda 环境
- [ ] `myconv` 的 `full` 模式是 O(N·M) 的纯 Python 双重循环，信号长时会明显变慢
- [ ] 清理本地的孤儿文件 `audio_base/test_440hz.wav`（没有任何脚本生成它）
- [ ] 把测试挪到 `tests/` 目录，并给 `audio_base` 也补上 pytest 测试
