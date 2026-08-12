"""RNNoise AX620E 推理会话（开发版：默认 AX Engine，本机可用 onnxruntime CPU 回退验证）。"""
import numpy as np

from . import dsp

DEFAULT_PROVIDER = "AxEngineExecutionProvider"

INPUT_NAMES = ["features", "conv1_mem", "conv2_mem",
               "gru1_s", "gru2_s", "gru3_s"]
INPUT_SHAPES = {"features": (1, 65), "conv1_mem": (1, 130),
                "conv2_mem": (1, 256), "gru1_s": (1, 384),
                "gru2_s": (1, 384), "gru3_s": (1, 384)}
OUTPUT_NAMES = ["gains", "vad", "conv1_mem_new", "conv2_mem_new",
                "gru1_s_new", "gru2_s_new", "gru3_s_new"]
_STATE_INPUTS = ["conv1_mem", "conv2_mem", "gru1_s", "gru2_s", "gru3_s"]


class RNNoiseDenoiser:
    """48k 单声道实时降噪器（默认 AX 芯片；本机验证时回退 onnxruntime CPU）。"""

    def __init__(self, model_path, providers=None):
        self.backend = "axengine"
        try:
            import axengine as axe
            self.session = axe.InferenceSession(
                model_path, providers=providers or [DEFAULT_PROVIDER])
        except Exception:
            import onnxruntime as ort
            self.session = ort.InferenceSession(
                model_path, providers=["CPUExecutionProvider"])
            self.backend = "onnxruntime"
        self.input_names = [i.name for i in self.session.get_inputs()]
        self.output_names = [o.name for o in self.session.get_outputs()]
        self.reset()

    def reset(self):
        self.st = dsp.RNNoiseState()
        self._states = {k: np.zeros(INPUT_SHAPES[k], dtype=np.float32)
                        for k in _STATE_INPUTS}

    def process_frame(self, frame):
        frame = np.asarray(frame, dtype=np.float32).reshape(-1)
        if frame.size != dsp.FRAME_SIZE:
            raise ValueError(f"帧长必须为 {dsp.FRAME_SIZE}，实际 {frame.size}")
        ana = dsp.analyze_frame(self.st, frame)
        if ana["silence"]:
            out, _ = dsp.synthesize_frame(self.st, ana, None, 0.0)
            return out, 0.0
        feeds = {
            "features": np.ascontiguousarray(
                ana["features"][None, :].astype(np.float32)),
        }
        feeds.update({k: np.ascontiguousarray(v)
                      for k, v in self._states.items()})
        outs = self.session.run(None, feeds)
        out_idx = {n: i for i, n in enumerate(self.output_names)}
        gains = outs[out_idx["gains"]]
        vad = outs[out_idx["vad"]]
        for k in _STATE_INPUTS:
            self._states[k] = np.asarray(
                outs[out_idx[k + "_new"]], dtype=np.float32)
        out, vad = dsp.synthesize_frame(self.st, ana, gains, vad)
        return out, float(vad)

    def process(self, pcm):
        pcm = np.asarray(pcm, dtype=np.float32)
        if pcm.ndim == 1:
            n = pcm.size // dsp.FRAME_SIZE
            frames = pcm[:n * dsp.FRAME_SIZE].reshape(n, dsp.FRAME_SIZE)
        else:
            frames = pcm.reshape(-1, dsp.FRAME_SIZE)
        outs, vads = [], []
        for fr in frames:
            o, v = self.process_frame(fr)
            outs.append(o)
            vads.append(v)
        return np.concatenate(outs), np.asarray(vads, dtype=np.float32)
