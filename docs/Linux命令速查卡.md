# Linux 常用命令速查卡

适用场景：在 Ubuntu 中学习 Python、DSP 和 Git。示例以 `audio-projects` 仓库为例。

> 复制命令时不要复制终端提示符（例如 `kingcrab@ubuntu:~$`），只复制命令本身。
> 示例中的 `.` 表示当前文件夹；`~` 表示你的用户主目录。

## 1. 看位置、浏览文件夹

| 命令 | 用途 | 示例 |
|---|---|---|
| `pwd` | 显示当前所在位置 | `pwd` |
| `ls` | 列出当前文件夹内容 | `ls` |
| `ls -la` | 连隐藏文件一起列出 | `ls -la` |
| `cd 路径` | 进入某个文件夹 | `cd ~/audio-projects` |
| `cd ..` | 回到上一级文件夹 | `cd ..` |

进入 DSP 子项目：

```bash
cd ~/audio-projects/dsp-toolbox
```

## 2. 阅读文件

| 命令 | 用途 | 示例 |
|---|---|---|
| `cat 文件` | 显示整个短文件 | `cat README.md` |
| `less 文件` | 翻页阅读，按 `q` 退出 | `less README.md` |
| `head -n 数字 文件` | 查看开头若干行 | `head -n 60 dsp_toolbox/conv.py` |
| `tail -n 数字 文件` | 查看最后若干行 | `tail -n 20 README.md` |
| `sed -n '起始,结束p' 文件` | 显示指定行范围 | `sed -n '1,20p' dsp_toolbox/conv.py` |

## 3. 找文件、找代码

### `find`：按文件名查找

```bash
find . -name "test_*.py"
```

从当前目录开始，查找名字以 `test_` 开头、以 `.py` 结尾的文件。`*` 代表任意长度的文字。

### `grep`：在文件内容中查找文字

```bash
grep -n "myconv" dsp_toolbox/conv.py
```

在指定文件中查找 `myconv`，并显示行号。`-R` 表示递归搜索文件夹：

```bash
grep -R -n "myconv" .
```

## 4. 文字处理工具

### `awk`：按列取出文字

```bash
ls -l | awk '{print $9}'
```

`awk` 会把每行按空白分列，`$9` 表示第九列，`print` 表示显示出来。这个简短例子用于理解语法；文件名含空格时，按列切分可能不准确。

### `sed`：按规则查看或替换文字

只查看第 1 到 20 行：

```bash
sed -n '1,20p' dsp_toolbox/conv.py
```

按规则替换并直接修改文件（会写入文件）：

```bash
sed -i 's/旧文字/新文字/g' 文件名
```

使用 `sed -i` 前先确认文件路径和替换内容；修改后可用 `git diff` 检查。

## 5. 管道 `|` 和 `xargs`

### 管道：把左边命令的输出交给右边

```bash
find . -name "*.py" | grep "test_"
```

先找出 Python 文件，再从结果里筛选包含 `test_` 的路径。`|` 本身不搜索，只负责传递输出。

### `xargs`：把收到的文字作为后一个命令的参数

```bash
find . -name "*.py" | xargs grep -n "myconv"
```

`find` 找到文件后，`xargs` 把文件名交给 `grep` 搜索。文件名含空格时使用带零字符分隔的写法：

```bash
find . -name "*.py" -print0 | xargs -0 grep -n "myconv"
```

在自己的代码仓库里，通常直接用下面这条更简单：

```bash
grep -R -n "myconv" .
```

## 6. 创建、复制和移动文件

| 命令 | 用途 | 示例 |
|---|---|---|
| `mkdir 名字` | 新建文件夹 | `mkdir notes` |
| `touch 文件` | 新建空文件 | `touch notes/today.md` |
| `cp 来源 目标` | 复制文件 | `cp a.txt b.txt` |
| `mv 来源 目标` | 移动文件或改名 | `mv old.txt new.txt` |

## 7. Python 项目常用命令

```bash
python --version                         # 查看 Python 版本
python -m pytest -q                      # 运行项目测试
python -m pytest tests/test_conv.py -q   # 只运行卷积测试
python examples/demo_moving_average.py --no-show  # 运行演示，不弹图窗
```

运行示例前，先进入子项目目录：

```bash
cd ~/audio-projects/dsp-toolbox
```

## 8. Git 检查命令

```bash
git status   # 查看哪些文件改过、是否有未提交修改
git diff     # 查看尚未暂存的具体改动
```

提交之前先运行 `git status` 和 `git diff`，确认改动是自己预期的。

## 最先记住这几条

```bash
pwd                         # 我现在在哪？
ls -la                      # 这里有哪些文件？
cd 路径                     # 进入哪个文件夹？
less 文件名                 # 打开文件慢慢看，按 q 退出
find . -name "名字模式"     # 按名字找文件
grep -R -n "要找的文字" .    # 在项目里找代码
git status                  # 我改了哪些文件？
python -m pytest -q         # 测试有没有通过？
```
