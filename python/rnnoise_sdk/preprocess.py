import numpy as np

def preprocess(frame, state):
    """48k PCM 帧(480, float32, 16-bit 域) -> (features(1,65), analysis)。
    对应原版 rnnoise_process_frame 的 biquad + 特征分析部分。
    state 为 dsp.RNNoiseState 实例（帧间状态，由 SDK 内部维护）。"""
    from .dsp import analyze_frame
    ana = analyze_frame(state, np.asarray(frame, dtype=np.float32).reshape(-1))
    return ana["features"][None, :].astype(np.float32), ana