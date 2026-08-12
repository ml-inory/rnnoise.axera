# Export Report (AX620E 复用)

- ONNX: export/model.onnx（opset 13, 静态 shape, 6 输入 / 7 输出），复用 rnnoise-ax650 已验证产物
- 权重来源: 官方 rnnoise_data.c（float 数组 + 稀疏重建 + diag），tanh/sigmoid 复刻 C 端有理逼近
- 原对分验证（198 帧真实语音特征序列）: gains cosine 1.000000, vad cosine 1.000000（ax650 任务完成）
- 本次复核: onnxruntime 加载通过，全静态 shape，用 calib_data 首帧推理输出 shape 正确且有限
- 校准数据: calib_data/<tensor>.tar.gz，来自真实语音（speech/speech-echo/speech-reverb + 合成噪声 6dB），每输入 40 帧特征+状态轨迹（real 业务数据）
- 状态语义: 逐帧推理，conv1/conv2 mem 各保留 2 帧，GRU 状态 384x3
