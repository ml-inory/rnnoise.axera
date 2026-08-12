本交付包为 rnnoise 双芯（AX650 / AX620E）合并仓库，两芯均已通过板端端到端 NPU 验证：
- AX650：AX650C 板，C++ 2.91ms/帧，输出 cosine 0.9988
- AX620E：AX630C 板（AX620E NPU2 同构），C++ 4.93ms/帧，输出 cosine 0.9989
Python SDK 为 NPU 专用版（仅 pyaxengine，无 onnxruntime/torch/transformers 回退）。
