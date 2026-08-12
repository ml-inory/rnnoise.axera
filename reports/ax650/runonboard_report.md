# Run On Board Report（RNNoise AX650）

- board: root@10.126.35.203（AX650C / MC50，aarch64）
- model: compile/model.axmodel（U16 链路，NPU3，3.31MB）
- engine: axengine 2.12.0s（板端本地推理，非远程代理）；C++ 链接 /soc/lib/libax_engine.so
- 测试音频: sample_speech.pcm（48k f32，16-bit 域，1 秒=100 帧，语音+白噪声 6dB）

## Python SDK（rnnoise_ax650_sdk，numpy DSP 移植）

- 每帧延迟: 60.8 ms（numpy DSP 为主，含每帧 axengine 调用；帧预算 10ms——实时路径请用 C++）
- 输出 vs c_ref（原版 C 全管线参考）: cosine 0.9973，MAE 243.8
- vad vs c_ref: cosine 0.9998
- 内存: 推理前后 549→541 MB（无显著增长）

## C++ SDK（原版 C 信号处理 + AX Engine compute_rnn 替换）

- 板上 cmake configure + make 通过（AX_RUNTIME_ROOT=/tmp/rnnoise_ax650/axrt，头文件 mc50）
- 每帧延迟: 2.85 ms（含 6 输入/7 输出拷贝 + NPU 0.3ms；实时预算 10ms ✅）
- 输出 vs c_ref: cosine 0.9820，MAE 819.6
- 内存: 539→541 MB

## 结论

- Python / C++ SDK 板端端到端 NPU 推理通过（输出 cosine ≥ 0.98）
- 实时降噪（10ms/帧）用 C++ SDK；Python SDK 适合离线批处理/原型验证
- SIMULATE 阶段已确认模型层 gains cosine 0.9991 ≥ 0.99
