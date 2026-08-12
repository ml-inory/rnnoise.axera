---
license: isc
pipeline_tag: audio-to-audio
tags:
- axmodel
- axera
- rnnoise
- audio-denoising
- ax650
- ax620e
---

# RNNoise（爱芯 NPU 版）

Xiph RNNoise 语音降噪在爱芯 NPU 上的部署包，支持 **AX650（NPU3）/ AX620E（NPU2）** 双芯。
官方模型仓库：https://github.com/xiph/rnnoise

## 特性

- 48kHz 单声道实时语音降噪（逐帧 10ms，16-bit 等价域 ±32768）
- 原版 rnnoise C 信号处理（特征分析 / 合成）+ AX Engine NPU 推理
- Python 与 C++ 双 SDK；官方未提供 ONNX，本仓库 ONNX 由官方 `rnnoise_data.c`
  权重 1:1 复刻导出（Tanh/Sigmoid 有理逼近对齐 C 端）

## 支持平台与模型

| 芯片 | NPU | 模型目录 | gains cosine | 板端验证 |
|---|---|---|---|---|
| AX650 | NPU3 | `rnnoise_ax650/` | 0.9991（板端 198 帧） | ✅ C++ 2.91ms/帧，输出 cosine 0.9988 |
| AX620E | NPU2 | `rnnoise_ax620e/` | 0.9987（Pulsar2 仿真 100 帧） | 待上板 |

每个模型目录内含 `model.axmodel` + `model_meta.json`。
量化：U16 链路（MatMul/Conv/Add/Mul/Div/Sub/Concat/Clip/Slice，S8 权重），
MinMax，校准数据来自真实语音（speech/speech-echo/speech-reverb + 6dB 噪声）。

## 快速开始

### 1. 安装 Python 环境

```bash
pip install -r python/requirements.txt
```

### 2. 板端运行降噪

```bash
# AX650
python3 python/demo.py --chip ax650
# AX620E
python3 python/demo.py --chip ax620e
```

输出 `output/out.pcm`（48k f32le）+ `output/vad.npy`；演示样本为
`python/sample_speech.pcm`（约 4 秒语音）。

## C++ SDK

| 芯片 | 目录 | 说明 |
|---|---|---|
| AX650 | `cpp/ax650/` | cmake 链接 ax_engine/ax_sys，板端实测 2.91ms/帧 |
| AX620E | `cpp/ax620e/` | 同源码树，目标 AX620E BSP 交叉编译 |

## 精度说明

| 张量 | AX650 板端 cosine | AX620E 仿真 cosine |
|---|---|---|
| gains | 0.9991 | 0.9987 |
| vad | 0.99996 | 0.99997 |
| GRU 状态 | 0.992–0.996 | 0.996+ |

## 复现

完整导出与编译流程（含双芯 Pulsar2 配置与校准数据）见
GitHub：https://github.com/ml-inory/rnnoise.axera
