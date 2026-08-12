"""RNNoise 48k 实时降噪演示（AX650 / AX620E 双芯）。

用法:
  python3 demo.py --chip ax650          # 默认处理 python/sample_speech.pcm
  python3 demo.py --chip ax620e --input in.pcm

输入格式：48kHz f32le PCM（16-bit 等价域，±32768，不做归一化）；也可传 16-bit WAV。
"""
import argparse
import sys
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CHIP_MODELS = {
    "ax650": ROOT / "rnnoise_ax650" / "model.axmodel",
    "ax620e": ROOT / "rnnoise_ax620e" / "model.axmodel",
}


def load_pcm(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".wav":
        with wave.open(str(path), "rb") as w:
            assert w.getframerate() == 48000, "仅支持 48kHz WAV"
            assert w.getsampwidth() == 2, "仅支持 16-bit PCM WAV"
            raw = w.readframes(w.getnframes())
        return np.frombuffer(raw, dtype="<i2").astype(np.float32)
    return np.fromfile(path, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="RNNoise 48k 实时降噪（AX650/AX620E）")
    parser.add_argument("--chip", choices=["ax650", "ax620e"], default="ax650",
                        help="目标芯片，决定使用哪个 axmodel")
    parser.add_argument("--input", default=str(ROOT / "python" / "sample_speech.pcm"),
                        help="48k f32 PCM 或 16-bit WAV")
    parser.add_argument("--output-dir", default="output")
    args = parser.parse_args()

    try:
        import axengine  # noqa: F401
        AX_AVAILABLE = True
    except Exception:
        AX_AVAILABLE = False
    if not AX_AVAILABLE:
        print("当前主机没有 AX 芯片（pyaxengine 不可用），无法运行 NPU 推理。")
        print("请在对应 AX 板端执行：python3 python/demo.py --chip ax650|ax620e")
        return

    sys.path.insert(0, str(ROOT / "python"))
    from rnnoise_sdk import RNNoiseDenoiser, dsp

    model = CHIP_MODELS[args.chip]
    if not model.is_file():
        print(f"模型不存在: {model}（请确认仓库完整）")
        sys.exit(1)
    pcm = load_pcm(Path(args.input))
    print(f"chip: {args.chip} | model: {model.name} | "
          f"input: {pcm.size / 48000:.2f}s ({pcm.size // dsp.FRAME_SIZE} 帧)")
    denoiser = RNNoiseDenoiser(str(model))
    out, vads = denoiser.process(pcm)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(exist_ok=True)
    out.astype(np.float32).tofile(out_dir / "out.pcm")
    np.save(out_dir / "vad.npy", vads)
    print(f"backend: {denoiser.backend}")
    print(f"语音存在比例: {float((vads > 0.5).mean()):.2f}")
    print(f"输出已保存: {out_dir / 'out.pcm'}")


if __name__ == "__main__":
    main()
