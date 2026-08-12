"""验证 numpy DSP 移植：c_ref(dump) 特征/输出 vs 本模块逐帧结果。"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rnnoise_dsp import (FRAME_SIZE, NB_FEATURES, RNNoiseState,
                         compute_frame_features, features_from_pcm,
                         process_frame, rnn_biquad)
from export_onnx import INPUT_NAMES, INPUT_SHAPES, OUTPUT_NAMES

TASK = Path(__file__).resolve().parent.parent
EXPORT = Path(__file__).resolve().parent


def cosine(a, b):
    a = np.asarray(a, dtype=np.float32).ravel()
    b = np.asarray(b, dtype=np.float32).ravel()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def c_ref_dump(pcm: Path) -> dict:
    feats = EXPORT / "v_tmp_feats.bin"
    out = EXPORT / "v_tmp_out.pcm"
    g = EXPORT / "v_tmp_g.bin"
    v = EXPORT / "v_tmp_v.bin"
    subprocess.run([str(EXPORT / "c_ref"), "dump", str(pcm), str(feats),
                    str(out), str(g), str(v)], check=True, capture_output=True)
    return {
        "features": np.fromfile(feats, dtype=np.float32).reshape(-1, NB_FEATURES),
        "out": np.fromfile(out, dtype=np.float32),
        "gains": np.fromfile(g, dtype=np.float32).reshape(-1, 32),
        "vad": np.fromfile(v, dtype=np.float32),
    }


def onnx_full(feats: np.ndarray) -> dict:
    sess = ort.InferenceSession(str(EXPORT / "model.onnx"),
                                providers=["CPUExecutionProvider"])
    st = {k: np.zeros(INPUT_SHAPES[k], dtype=np.float32) for k in INPUT_NAMES}
    gains, vads, states = [], [], []
    for t in range(feats.shape[0]):
        st["features"] = feats[t:t + 1].astype(np.float32)
        r = sess.run(OUTPUT_NAMES, {k: np.ascontiguousarray(v) for k, v in st.items()})
        gains.append(r[0]); vads.append(r[1])
        states.append({k: r[OUTPUT_NAMES.index(k + "_new")] for k in
                       ["conv1_mem", "conv2_mem", "gru1_s", "gru2_s", "gru3_s"]})
        st = {k: r[OUTPUT_NAMES.index(k + "_new")] for k in
              ["conv1_mem", "conv2_mem", "gru1_s", "gru2_s", "gru3_s"]}
    return {"gains": np.concatenate(gains), "vad": np.concatenate(vads),
            "states": states}


def numpy_full(pcm: np.ndarray, nn) -> np.ndarray:
    st = RNNoiseState()
    outs = []
    for t in range(nn["gains"].shape[0]):
        frame = pcm[t * FRAME_SIZE:(t + 1) * FRAME_SIZE]
        out, _ = process_frame(st, frame, nn["gains"][t], nn["vad"][t])
        outs.append(out)
    return np.concatenate(outs)


def main() -> None:
    samples = ["speech.wav", "speech_noisy.wav", "speech_echo.wav",
               "speech_reverb.wav"]
    report = {}
    for name in samples:
        pcm = np.fromfile(EXPORT / name, dtype=np.float32)
        ref = c_ref_dump(EXPORT / name)
        feats_np = features_from_pcm(pcm)
        fcos = cosine(feats_np, ref["features"])
        fmae = float(np.abs(feats_np - ref["features"]).mean())
        pitch_ref = ref["features"][:, 64]
        pitch_np = feats_np[:, 64]
        pitch_match = float(np.mean(np.abs(pitch_ref - pitch_np) < 0.5))
        nn = onnx_full(feats_np)
        out_np = numpy_full(pcm, nn)
        ocos = cosine(out_np, ref["out"])
        omae = float(np.abs(out_np - ref["out"]).mean())
        gcos = cosine(nn["gains"], ref["gains"])
        vcos = cosine(nn["vad"], ref["vad"])
        entry = {
            "frames": feats_np.shape[0],
            "features_cosine": fcos,
            "features_mae": fmae,
            "pitch_match_ratio": pitch_match,
            "gains_cosine": gcos,
            "vad_cosine": vcos,
            "out_cosine": ocos,
            "out_mae": omae,
        }
        report[name] = entry
        print(name, json.dumps(entry, indent=1))
    (EXPORT / "dsp_verify.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
