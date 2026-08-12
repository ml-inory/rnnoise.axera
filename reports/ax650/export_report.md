# Export Report

- ONNX: export/model.onnx (opset 13, 静态 shape, 6 输入 / 7 输出)
- 输入域: 与官方 demo 一致，float 值等价 16-bit PCM（±32768 量级），不做 /32768 归一化
- 权重来源: 官方 rnnoise_data.c（float 数组 + 稀疏重建 + diag），
  tanh/sigmoid 复刻 C 端有理逼近
- Torch(参考) ↔ ONNX 对分（198 帧真实语音特征序列）:
  - gains cosine: 1.000000
  - vad cosine:   1.000000
  - gains MAE:    1.42e-07
- C 库 compute_rnn ↔ Torch 参考对分（198 帧）: gains cosine 0.999999, vad 0.9999995
- 校准数据: calib_data/<tensor>.tar.gz，来自真实语音（speech/speech-echo/
  speech-reverb + 合成噪声混合 6dB），每输入 40 帧特征+状态轨迹（real 业务数据）
- 状态语义: 逐帧推理，conv1/conv2 mem 各保留 2 帧，GRU 状态 384x3
