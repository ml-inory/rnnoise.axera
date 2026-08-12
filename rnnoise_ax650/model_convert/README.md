# Model Convert（可复现编译）

本目录是从 RNNoise（xiph/rnnoise）导出 ONNX 并编译为 AX650 NPU3 AXMODEL 的
可复现流程。产物：`output/model.axmodel`（U16 链路，约 3.3MB）。

## 环境准备

- Docker
- Pulsar2 7.0 镜像（原始编译镜像
  `docker-registry.aitsw.axera-tech.com/pulsar2:20260810-temp-0d4427ff`，
  如本机已打成 `pulsar2:7.0` 标签可用 `PULSAR2_IMAGE=pulsar2:7.0` 覆盖）
- 爱芯 Pulsar2 license（如 license 目录不在默认位置，用
  `MAGNETAR_HASP_SRC=/path/to/.hasplm bash compile_pulsar2.sh` 挂载）

## 一键编译

在本目录执行：

```bash
bash compile_pulsar2.sh
```

成功后产物在 `output/model.axmodel`，编译日志在 `output/compile.log`（如需）。

## 模型结构说明

- 输入（逐帧状态化，float32，16-bit PCM 等价域 ±32768）：
  `features(1,65)`、`conv1_mem(1,130)`、`conv2_mem(1,256)`、
  `gru1_s/gru2_s/gru3_s(1,384)`
- 输出：`gains(1,32)`、`vad(1,1)` + 5 个新状态
- 网络：Conv1d(65→128)+tanh → Conv1d(128→384)+tanh → GRU(384)×3 →
  concat → Dense(32 sigmoid gains) + Dense(1 sigmoid vad)
- 量化：MinMax 校准（真实语音特征+状态轨迹 30 组/输入），
  U16 链路（MatMul/Conv/Add/Mul/Div/Sub/Concat/Clip/Slice，S8 权重），
  `highest_mix_precision=false`
- 校准数据：`calib_data/<tensor>.tar.gz`（speech/echo/reverb/噪声混合 6dB）

## 如何从零重新导出 ONNX

```bash
# 1. 获取权重（官方 rnnoise_data.c 内嵌权重，无需额外下载）
git clone https://github.com/xiph/rnnoise
# 2. 安装依赖
pip install torch numpy onnx onnxruntime
# 3. 在任务目录 export/ 下执行（会重新生成 model.onnx + calib_data/ + 验证报告）
python export_onnx.py
# 4. 将 model.onnx、calib_data/ 与 pulsar2_config.json 放回本目录后编译
```

注：`export_onnx.py` 依赖本目录的 `rnnoise_torch.py`（torch 参考模型，
复刻 C 端 tanh/sigmoid 有理逼近与稀疏权重）、`rnnoise_dsp.py`
（numpy 信号处理移植）与 `parse_rnnoise_weights.py`（C 权重解析），
已随包提供；音频样本生成需 ffmpeg。
