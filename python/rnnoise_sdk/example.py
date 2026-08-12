"""RNNoise 降噪示例：处理一段 48k 单声道音频，输出去噪结果。

用法:
  python example.py --model model.axmodel --input in.pcm [--output-dir out]

输入格式：48kHz f32le PCM（16-bit 等价域，±32768，不做归一化）；
也可传 16-bit PCM .wav（wave 标准库自动解码为 ±32768 域）。
"""
import argparse
import sys
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rnnoise_sdk import RNNoiseDenoiser, dsp  # noqa: E402


def load_pcm(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".wav":
        with wave.open(str(path), "rb") as w:
            assert w.getframerate() == 48000, "仅支持 48kHz WAV"
            assert w.getsampwidth() == 2, "仅支持 16-bit PCM WAV"
            raw = w.readframes(w.getnframes())
        return np.frombuffer(raw, dtype="<i2").astype(np.float32)
    return np.fromfile(path, dtype=np.float32)


def write_wav(path: Path, pcm: np.ndarray, sr: int = 48000) -> None:
    pcm = np.clip(pcm, -32768.0, 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def main() -> None:
    parser = argparse.ArgumentParser(description="RNNoise 48k 实时降噪示例")
    parser.add_argument("--model", required=True, help="model.axmodel 路径")
    parser.add_argument("--input", required=True, help="48k f32 PCM 或 16-bit WAV")
    parser.add_argument("--output-dir", default="output")
    args = parser.parse_args()

    pcm = load_pcm(Path(args.input))
    print(f"input: {pcm.size / 48000:.2f}s ({pcm.size // dsp.FRAME_SIZE} 帧)")
    denoiser = RNNoiseDenoiser(args.model)
    out, vads = denoiser.process(pcm)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out.astype(np.float32).tofile(out_dir / "out.pcm")
    write_wav(out_dir / "out.wav", out)
    np.save(out_dir / "vad.npy", vads)
    print(f"backend: {denoiser.backend}")
    print(f"frames: {vads.size}  vad_mean: {float(vads.mean()):.4f}")
    print(f"output RMS: {float(np.sqrt((out ** 2).mean())):.1f}")
    print(f"saved to: {out_dir}")


if __name__ == "__main__":
    main()
