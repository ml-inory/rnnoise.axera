# rnnoise-ax620e C++ SDK

48kHz 单声道实时降噪，RNNoise 原版 C 信号处理 + AX Engine（NPU2，AX630C/AX620Q 同构）网络推理。

## 编译（AX620E 板端/交叉环境）

AX Engine 头文件与库来自官方 BSP SDK（AX620E NPU2 / AX630C 通用）：

```bash
# 官方 BSP：https://github.com/AXERA-TECH/ax620q_bsp_sdk  tag v2.0.0
# 解压后 msp/out/arm64_glibc/{include,lib} 即为头文件与库
export AX_RUNTIME_ROOT=/path/to/ax620e_bsp_sdk/msp/out/arm64_glibc
```

```bash
mkdir -p build && cd build
cmake .. -DAX_RUNTIME_ROOT=$AX_RUNTIME_ROOT
make -j$(nproc)
```

> 注：AX620E 官方 `libax_engine.so` 依赖 `libax_interpreter.so`
> （`AX_NPU_Get_*` 系列符号），CMakeLists 已链接 `ax_interpreter`。
> 运行前 `export LD_LIBRARY_PATH=$AX_RUNTIME_ROOT/lib:/opt/lib`。

## 运行

```bash
./build/model_example model.axmodel in.pcm out.pcm
```

输入为 48kHz f32le PCM（16-bit 等价域 ±32768，不做归一化）；
输出为同格式去噪 PCM。每帧 480 采样（10ms），vad 打印均值。
板端实测（AX630C，100 帧）：4.93ms/帧，输出 vs 原版 C 参考 cosine 0.9989。

## 结构

- `include/rnnoise_ax.hpp`：`RNNoiseAX` 类（进程内单实例）
- `src/rnnoise_ax.cpp`：以 AX Engine 替换原版 `compute_rnn`（6 输入/7 输出逐帧状态化）
- `src/model_runner.cpp`：AX Engine 会话封装（按张量名映射，含缓存同步）
- `src/rnnoise/`：原版 rnnoise C 信号处理源码（denoise/pitch/FFT/LPC/表格，ISC 许可）；
  网络权重已内嵌 AXMODEL，`rnnoise_data.c` 为最小 stub（空权重表 + 零初始化）
