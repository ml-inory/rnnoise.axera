import numpy as np

def postprocess(ana, gains, vad, state):
    """模型输出(gains/vad) + 分析结果 -> (去噪帧(480,), vad)。
    对应原版 rnnoise_process_frame 的 pitch filter / 增益平滑 / 频谱合成部分。"""
    from .dsp import synthesize_frame
    return synthesize_frame(state, ana,
                            np.asarray(gains, dtype=np.float32),
                            np.asarray(vad, dtype=np.float32))