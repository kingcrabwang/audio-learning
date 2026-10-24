"""生成 C/Python FIR 对拍使用的固定输入和参考输出。"""

from pathlib import Path

import numpy as np
from scipy.signal import lfilter


def main():
    output_dir = Path(__file__).resolve().parents[1] / "golden" / "fir"
    output_dir.mkdir(parents=True, exist_ok=True)

    for numtaps in (1, 3, 17, 101):
        rng = np.random.default_rng(2026 + numtaps)
        # 使用非对称系数，避免系数顺序错误被对称性掩盖。
        h = (rng.standard_normal(numtaps) / numtaps).astype(np.float32)
        x = (0.2 * rng.standard_normal(1103)).astype(np.float32)
        # 参考值使用 C 实际接收到的 float32 数据，转为双精度计算。
        expected = lfilter(h.astype(np.float64), [1.0], x.astype(np.float64))
        np.save(output_dir / f"h_{numtaps}.npy", h, allow_pickle=False)
        np.save(output_dir / f"x_{numtaps}.npy", x, allow_pickle=False)
        np.save(output_dir / f"y_{numtaps}.npy", expected, allow_pickle=False)

    print(f"已生成 4 组固定对拍数据：{output_dir}")


if __name__ == "__main__":
    main()
