# SIMULATE Report

- board: 10.126.35.203
- frames: 198
- 方法：逐帧状态化远程推理（ONNX FP32 参考 vs AXMODEL），远程每帧 ~0.3ms NPU
- 量化：U16 链路（MatMul/Conv/Add/Mul/Div/Sub/Concat/Clip/Slice，S8 权重），MinMax，calibration_size=30
- 指标：

| 张量 | cosine | MAE |
|------|--------|-----|
| gains | 0.999122 | 0.017231 |
| vad | 0.999962 | 0.001985 |
| conv1_mem_new | 0.999127 | 0.001900 |
| conv2_mem_new | 0.997568 | 0.020142 |
| gru1_s_new | 0.995698 | 0.030715 |
| gru2_s_new | 0.994793 | 0.036976 |
| gru3_s_new | 0.992189 | 0.050399 |

- gains max_abs_diff: 0.275927
- vad max_abs_diff: 0.078959

结论：gains cosine 0.9991 ≥ 0.99 ✅，GRU 状态 0.992-0.996 ✅，SIMULATE 通过。
