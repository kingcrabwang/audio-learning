# -*- coding: utf-8 -*-
"""
生成波形 wav 文件。

用法：
    python make_wave.py sine                 # sine_1K.wav：1kHz 正弦，1 秒
    python make_wave.py chirp                # chirp.wav：200→2000Hz 扫频，3 秒

    # 参数默认值就是原来两个脚本里的值，不改参数时输出和原来完全一样
    python make_wave.py sine --freq 440 --duration 2 --sr 48000
    python make_wave.py chirp --f0 50 --f1 5000 --out sweep.wav

对应原来的 2creatwavefile.py 和 creatWAV.py。
这两个脚本除了「生成哪种波形」之外几乎一模一样，所以合并成一个，
波形算法本身抽到 waveforms.py，这里只负责解析参数 + 写文件。
"""
import argparse
import sys
from pathlib import Path

import soundfile as sf

from dsp_toolbox import chirp_wave, sine_wave

SAMPLES = Path(__file__).resolve().parent.parent / "samples"

__all__ = ["build_parser", "main"]

# 各种模式下的默认时长，取自原来两个脚本
DEFAULT_DURATION = {"sine": 1.0, "chirp": 3.0}


def build_parser():
    parser = argparse.ArgumentParser(
        description="生成波形 wav 文件",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("kind", choices=sorted(DEFAULT_DURATION),
                        help="要生成哪种波形")
    parser.add_argument("--sr", type=int, default=44100, help="采样率 Hz")
    parser.add_argument("--duration", type=float, default=None,
                        help="时长（秒），不填则用各波形的默认值")
    parser.add_argument("--amplitude", type=float, default=0.5, help="振幅")
    parser.add_argument("--freq", type=float, default=1000,
                        help="正弦波频率 Hz（仅 sine 模式用）")
    parser.add_argument("--f0", type=float, default=200,
                        help="扫频起始频率 Hz（仅 chirp 模式用）")
    parser.add_argument("--f1", type=float, default=2000,
                        help="扫频终止频率 Hz（仅 chirp 模式用）")
    parser.add_argument("--out", default=None,
                        help="输出文件名，不填则用默认名")
    return parser


DEFAULT_OUT = {"sine": "sine_1K.wav", "chirp": "chirp.wav"}


def main(argv=None):
    args = build_parser().parse_args(argv)
    duration = DEFAULT_DURATION[args.kind] if args.duration is None else args.duration

    if args.kind == "sine":
        signal = sine_wave(args.freq, args.sr, duration, args.amplitude)
    else:
        signal = chirp_wave(args.f0, args.f1, args.sr, duration, args.amplitude)

    out = SAMPLES / (args.out or DEFAULT_OUT[args.kind])
    SAMPLES.mkdir(exist_ok=True)
    sf.write(str(out), signal, args.sr, subtype="PCM_16")
    print(f"已生成 {out}（{len(signal)} 点，{len(signal) / args.sr:.3f} 秒）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
