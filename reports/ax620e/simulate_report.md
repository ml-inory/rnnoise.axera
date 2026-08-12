# Simulate Report

Method: pulsar2 run（无 AX620E 板回退仿真）
target: AX620E NPU2
frames: 30~129（预热 30 帧后状态化推理 100 帧）

- gains_cosine: 0.998734
- gains_mae: 0.017887
- vad_cosine: 0.999971
- vad_mae: 0.002027
- conv1_mem_new_cosine: 0.999442
- conv1_mem_new_mae: 0.001393
- conv2_mem_new_cosine: 0.997951
- conv2_mem_new_mae: 0.019488
- gru1_s_new_cosine: 0.996999
- gru1_s_new_mae: 0.022802
- gru2_s_new_cosine: 0.996491
- gru2_s_new_mae: 0.026434
- gru3_s_new_cosine: 0.996274
- gru3_s_new_mae: 0.031290
- gains_max_abs_diff: 0.285138
- vad_max_abs_diff: 0.053815

结论：gains cosine 0.9987 ≥ 0.99 ✅（参考 AX650 板端 gains 0.9991，U16 链路一致）