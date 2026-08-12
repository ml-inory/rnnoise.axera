本交付包为 rnnoise 双芯（AX650 / AX620E）合并仓库。
AX650 已通过板端端到端 NPU 验证（C++ 2.91ms/帧，输出 cosine 0.9988）；
AX620E 尚未上板验证（Pulsar2 仿真 gains cosine 0.9987），交付 SDK 保留 onnxruntime CPU 回退用于本机验证。
