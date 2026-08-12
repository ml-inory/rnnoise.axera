# rnnoise-ax650 C++ SDK

48kHz 单声道实时降噪，RNNoise 原版 C 信号处理 + AX Engine（NPU3）网络推理。

## 编译（AX650 板端/交叉环境）

```bash
export AX_RUNTIME_ROOT=/path/to/axruntime   # 含 include/ax_engine_api.h 与 lib/libax_engine.so
mkdir -p build && cd build
cmake .. -DAX_RUNTIME_ROOT=$AX_RUNTIME_ROOT
make -j$(nproc)
```

## 运行

```bash
./build/model_example model.axmodel in.pcm out.pcm
```

输入为 48kHz f32le PCM（16-bit 等价域 ±32768，不做归一化）；
输出为同格式去噪 PCM。每帧 480 采样（10ms），vad 打印均值。

## 结构

- `include/rnnoise_ax.hpp`：`RNNoiseAX` 类（进程内单实例）
- `src/rnnoise_ax.cpp`：以 AX Engine 替换原版 `compute_rnn`（6 输入/7 输出逐帧状态化）
- `src/model_runner.cpp`：AX Engine 会话封装（按张量名映射，含缓存同步）
- `src/rnnoise/`：原版 rnnoise C 信号处理源码（denoise/pitch/FFT/LPC/表格，ISC 许可）；
  网络权重已内嵌 AXMODEL，`rnnoise_data.c` 为最小 stub（空权重表 + 零初始化）
