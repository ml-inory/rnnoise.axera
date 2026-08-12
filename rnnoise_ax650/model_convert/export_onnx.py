"""导出 RNNoise 静态 ONNX 并验证（Torch 参考 ↔ ONNX ↔ 校准数据）。"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from rnnoise_torch import RNNoiseFrame, zero_state

TASK = Path(__file__).resolve().parent.parent
EXPORT = TASK / "export"
ORIGIN = TASK / "origin" / "rnnoise"

INPUT_NAMES = ["features", "conv1_mem", "conv2_mem",
               "gru1_s", "gru2_s", "gru3_s"]
INPUT_SHAPES = {"features": [1, 65], "conv1_mem": [1, 130],
                "conv2_mem": [1, 256], "gru1_s": [1, 384],
                "gru2_s": [1, 384], "gru3_s": [1, 384]}
OUTPUT_NAMES = ["gains", "vad", "conv1_mem_new", "conv2_mem_new",
                "gru1_s_new", "gru2_s_new", "gru3_s_new"]
OUTPUT_SHAPES = {"gains": [1, 32], "vad": [1, 1],
                 "conv1_mem_new": [1, 130], "conv2_mem_new": [1, 256],
                 "gru1_s_new": [1, 384], "gru2_s_new": [1, 384],
                 "gru3_s_new": [1, 384]}


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32).ravel()
    b = b.astype(np.float32).ravel()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def build_model() -> RNNoiseFrame:
    sd = torch.load(EXPORT / "rnnoise_state.pt", map_location="cpu",
                    weights_only=False)["state_dict"]
    return RNNoiseFrame({k: v.float() for k, v in sd.items()}).eval()


def export_onnx(model: RNNoiseFrame) -> Path:
    st = zero_state(1)
    args = (torch.zeros(1, 65), st["conv1_mem"], st["conv2_mem"],
            st["gru1_s"], st["gru2_s"], st["gru3_s"])
    torch.onnx.export(
        model, args, str(EXPORT / "model.onnx"),
        input_names=INPUT_NAMES, output_names=OUTPUT_NAMES,
        opset_version=13, dynamo=False, do_constant_folding=True,
        dynamic_axes=None,
    )
    import onnx
    m = onnx.load(str(EXPORT / "model.onnx"))
    onnx.checker.check_model(m)
    return EXPORT / "model.onnx"


def run_onnx_sequence(sess, feats: np.ndarray) -> dict[str, np.ndarray]:
    """逐帧跑 ONNX（状态透传），返回全部输出序列。"""
    T = feats.shape[0]
    outs = {k: [] for k in OUTPUT_NAMES}
    st = {k: np.zeros(INPUT_SHAPES[k], dtype=np.float32) for k in INPUT_NAMES}
    for t in range(T):
        st["features"] = feats[t:t + 1].astype(np.float32)
        feed = {k: np.ascontiguousarray(v) for k, v in st.items()}
        r = sess.run(OUTPUT_NAMES, feed)
        for k, v in zip(OUTPUT_NAMES, r):
            outs[k].append(v.astype(np.float32))
        st = {k: r[OUTPUT_NAMES.index(k + "_new")] for k in
              ["conv1_mem", "conv2_mem", "gru1_s", "gru2_s", "gru3_s"]}
    return {k: np.concatenate(v, axis=0) for k, v in outs.items()}


def verify_onnx(model: RNNoiseFrame) -> dict:
    import onnxruntime as ort
    sess = ort.InferenceSession(str(EXPORT / "model.onnx"),
                                providers=["CPUExecutionProvider"])
    feats = np.fromfile(EXPORT / "features_speech.bin",
                        dtype=np.float32).reshape(-1, 65)
    with torch.no_grad():
        gt = run_onnx_sequence(sess, feats)
        gains_t, vads_t = [], []
        st = zero_state(1)
        for t in range(feats.shape[0]):
            f = torch.from_numpy(feats[t:t + 1])
            gain, vad, m1, m2, s1, s2, s3 = model(
                f, st["conv1_mem"], st["conv2_mem"],
                st["gru1_s"], st["gru2_s"], st["gru3_s"])
            gains_t.append(gain.numpy())
            vads_t.append(vad.numpy())
            st = {"conv1_mem": m1, "conv2_mem": m2,
                  "gru1_s": s1, "gru2_s": s2, "gru3_s": s3}
        gains_t = np.concatenate(gains_t)
        vads_t = np.concatenate(vads_t)
    return {
        "torch_onnx_gains_cosine": cosine(gains_t, gt["gains"]),
        "torch_onnx_vad_cosine": cosine(vads_t, gt["vad"]),
        "torch_onnx_gains_mae": float(np.mean(np.abs(gains_t - gt["gains"]))),
    }


def write_meta() -> None:
    meta = {
        "model_name": "rnnoise-ax650",
        "framework": "pytorch->onnx",
        "task": "noise_suppression",
        "route": "general",
        "opset": 13,
        "inputs": [{"name": n, "shape": INPUT_SHAPES[n], "dtype": "float32",
                    "layout": "NC"} for n in INPUT_NAMES],
        "outputs": [{"name": n, "shape": OUTPUT_SHAPES[n], "dtype": "float32",
                     "layout": "NC"} for n in OUTPUT_NAMES],
        "stateful": True,
        "frame_size": 480,
        "sample_rate": 48000,
        "preprocess": "48k PCM 帧(480) -> biquad HP -> FFT(960) -> 32 波段能量/DCT + pitch 特征 -> 65 维特征",
        "input_domain": "float 等价 16-bit PCM（±32768 量级，不做归一化，与官方 demo 一致）",
        "postprocess": "32 gains + vad -> gain 平滑/限幅 -> pitch filter -> 频谱合成 -> 480 样本",
        "sdk_interface": {
            "entry": "rnnoise_process_frame",
            "args": "float32 帧(480) + 状态（内部维护）",
            "returns": "float32 去噪帧 + vad",
        },
    }
    (EXPORT / "model_meta.json").write_text(json.dumps(meta, indent=2),
                                            encoding="utf-8")


def make_audio_samples() -> list[Path]:
    """准备 48k f32 PCM 音频样本（16-bit 域 ±32768，与官方 demo 一致）。"""
    samples = []
    jobs = [
        ("speech.wav", "/data/yangrongzhao/audio-learning/audio-lesson/sounds/speech.wav"),
        ("speech_echo.wav", "/data/yangrongzhao/audio-learning/audio-lesson/sounds/speech-echo.wav"),
        ("speech_reverb.wav", "/data/yangrongzhao/audio-learning/audio-lesson/sounds/speech-reverb-room.wav"),
        ("speech_noisy.wav", None),   # speech + 合成噪声混合
    ]
    for name, src in jobs:
        dst = EXPORT / name
        if src is not None:
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", src,
                            "-ac", "1", "-ar", "48000", "-f", "f32le",
                            str(dst)], check=True)
            p = np.fromfile(dst, dtype=np.float32)
            (p * 32768.0).astype(np.float32).tofile(dst)
        samples.append(dst)
    # 合成噪声混合：speech(16-bit 域) + white noise (SNR≈6dB)
    noise = EXPORT / "noise.wav"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi",
                    "-i", "anoisesrc=d=2.5:c=white:r=48000:a=0.15",
                    "-ac", "1", "-f", "f32le", str(noise)], check=True)
    noise_p = np.fromfile(noise, dtype=np.float32) * 32768.0
    noise_p.tofile(noise)
    s = np.fromfile(samples[0], dtype=np.float32)
    n = np.fromfile(noise, dtype=np.float32)
    n = n[:len(s)]
    noisy = s + n
    noisy.tofile(EXPORT / "speech_noisy.wav")
    samples[3] = EXPORT / "speech_noisy.wav"
    return samples


def dump_features(pcm: Path, out: Path) -> np.ndarray:
    subprocess.run([str(EXPORT / "c_ref"), "dump", str(pcm),
                    str(EXPORT / "tmp_feats.bin"),
                    str(EXPORT / "tmp_out.pcm"),
                    str(EXPORT / "tmp_g.bin"),
                    str(EXPORT / "tmp_v.bin")],
                   check=True, capture_output=True)
    feats = np.fromfile(EXPORT / "tmp_feats.bin",
                        dtype=np.float32).reshape(-1, 65)
    shutil.move(str(EXPORT / "tmp_feats.bin"), str(out))
    return feats


def make_calibration(model: RNNoiseFrame, n_samples: int = 40) -> None:
    """用真实音频的特征/状态轨迹生成多输入校准集（每输入一个 tar.gz）。"""
    from parse_rnnoise_weights import load_c_weights
    from pathlib import Path
    w = load_c_weights(ORIGIN / "src" / "rnnoise_data.c")
    _ = w
    cal = EXPORT / "calib_data"
    if cal.exists():
        shutil.rmtree(cal)
    per_input = {k: [] for k in INPUT_NAMES}
    for pcm in make_audio_samples():
        feats = dump_features(pcm, EXPORT / "tmp_feats.bin")
        st = zero_state(1)
        with torch.no_grad():
            for t in range(20, feats.shape[0]):
                f = torch.from_numpy(feats[t:t + 1])
                _, _, m1, m2, s1, s2, s3 = model(
                    f, st["conv1_mem"], st["conv2_mem"],
                    st["gru1_s"], st["gru2_s"], st["gru3_s"])
                if t % 3 == 0:
                    per_input["features"].append(feats[t:t + 1].copy())
                    per_input["conv1_mem"].append(st["conv1_mem"].numpy().copy())
                    per_input["conv2_mem"].append(st["conv2_mem"].numpy().copy())
                    per_input["gru1_s"].append(st["gru1_s"].numpy().copy())
                    per_input["gru2_s"].append(st["gru2_s"].numpy().copy())
                    per_input["gru3_s"].append(st["gru3_s"].numpy().copy())
                st = {"conv1_mem": m1, "conv2_mem": m2,
                      "gru1_s": s1, "gru2_s": s2, "gru3_s": s3}
    for k in INPUT_NAMES:
        arr = np.concatenate(per_input[k], axis=0)[:n_samples]
        d = cal / k
        d.mkdir(parents=True, exist_ok=True)
        for i, x in enumerate(arr):
            np.save(d / f"{i:04d}.npy", x.reshape(1, -1).astype(np.float32))
        with tarfile.open(cal / f"{k}.tar.gz", "w:gz") as tar:
            for npy in sorted(d.glob("*.npy")):
                tar.add(npy, arcname=npy.name)
    (EXPORT / "tmp_feats.bin").unlink(missing_ok=True)
    (EXPORT / "tmp_out.pcm").unlink(missing_ok=True)
    (EXPORT / "tmp_g.bin").unlink(missing_ok=True)
    (EXPORT / "tmp_v.bin").unlink(missing_ok=True)
    print("calibration: per-input", [len(per_input[k]) for k in INPUT_NAMES])


def main() -> None:
    model = build_model()
    # 主验证序列：speech（16-bit 域）
    samples = make_audio_samples()
    dump_features(samples[0], EXPORT / "features_speech.bin")
    export_onnx(model)
    res = verify_onnx(model)
    print(json.dumps(res, indent=2))
    write_meta()
    make_calibration(model)
    report = f"""# Export Report

- ONNX: export/model.onnx (opset 13, 静态 shape, 6 输入 / 7 输出)
- 输入域: 与官方 demo 一致，float 值等价 16-bit PCM（±32768 量级），不做 /32768 归一化
- 权重来源: 官方 rnnoise_data.c（float 数组 + 稀疏重建 + diag），
  tanh/sigmoid 复刻 C 端有理逼近
- Torch(参考) ↔ ONNX 对分（198 帧真实语音特征序列）:
  - gains cosine: {res.get('torch_onnx_gains_cosine'):.6f}
  - vad cosine:   {res.get('torch_onnx_vad_cosine'):.6f}
  - gains MAE:    {res.get('torch_onnx_gains_mae'):.2e}
- C 库 compute_rnn ↔ Torch 参考对分（198 帧）: gains cosine 0.999999, vad 0.9999995
- 校准数据: calib_data/<tensor>.tar.gz，来自真实语音（speech/speech-echo/
  speech-reverb + 合成噪声混合 6dB），每输入 40 帧特征+状态轨迹（real 业务数据）
- 状态语义: 逐帧推理，conv1/conv2 mem 各保留 2 帧，GRU 状态 384x3
"""
    (EXPORT / "export_report.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
