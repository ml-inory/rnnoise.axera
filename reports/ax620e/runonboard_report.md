# Run On Board Report（RNNoise AX620E）

- board: root@10.126.35.148（AX630C_CHIP（AX620E NPU2 同构），aarch64）
- model: compile/model.axmodel（U16 链路，NPU2，2.93MB，Pulsar2 7.0 编译）
- 测试音频: sample_speech.pcm（48k f32，16-bit 域，100 帧，语音+白噪声 6dB）

## Python SDK（rnnoise_ax620e_sdk，numpy DSP 移植）

- 每帧延迟: 106.0 ms（numpy DSP 为主；实时路径用 C++）
- 输出 vs c_ref（原版 C 全管线参考）: cosine 0.997419
- vad vs c_ref: cosine 0.999873

## C++ SDK（原版 C 信号处理 + AX Engine 替换，官方 ax620e_bsp_sdk v2.0.0 头文件/库编译）

- 每帧延迟: 4.933 ms（实时预算 10ms ✅）
- 输出 vs c_ref: cosine 0.998871，vad_mean 0.9924
- Python vs C++: cosine 0.997972

## 结论

- AX620E NPU2 axmodel 在 AX630C 板（同 NPU）端到端推理通过（Python/C++ cosine ≥ 0.98）
- 与 AX650 板端（C++ 2.91ms/帧，cosine 0.9988）同量级；AX630C 单核 NPU 为 4.93ms/帧，仍满足 10ms 实时预算
