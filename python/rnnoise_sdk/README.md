# rnnoise-ax Python SDK

48kHz 单声道实时降噪（RNNoise，AX650 NPU3 / AX620E NPU2 双芯编译）。

- 输入：480 采样/帧 float32（16-bit PCM 等价域 ±32768，不做归一化）
- 输出：去噪帧(480) + vad
- 模型：6 输入（features + 5 个状态）/ 7 输出（gains/vad + 5 个新状态），
  逐帧状态化推理，状态在 `RNNoiseDenoiser` 实例内维护
- 前后处理：numpy 移植原版 rnnoise C 管线（biquad/FFT/带能量 DCT/pitch/
  频谱合成），对照 `c_ref` 逐帧验证：特征 cosine≥0.995、去噪输出 cosine≥0.995

```python
from rnnoise_sdk import RNNoiseDenoiser
import numpy as np

denoiser = RNNoiseDenoiser("model.axmodel")
frame = np.random.randn(480).astype(np.float32) * 3000  # 16-bit 域
out, vad = denoiser.process_frame(frame)
```

NPU 专用发布版：仅依赖 numpy + pyaxengine（无 onnxruntime/torch/transformers 回退）。
