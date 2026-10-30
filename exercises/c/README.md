# C 算法练习

这里放独立的 C 语法和 LeetCode 练习，不参与 `dsp-toolbox/csrc/` 的算法库构建。

每个文件有自己的 `main()`，可单独编译，例如在仓库根目录运行：

```bash
gcc -std=c11 -Wall -Wextra exercises/c/demo_remove_element.c -o /tmp/demo_remove_element
/tmp/demo_remove_element
```

可复用的音频算法与对应头文件仍放在 `dsp-toolbox/csrc/src/` 和 `dsp-toolbox/csrc/include/`。未来的 C 卷积、FFT、LMS 也按这个位置添加。
