/* RNNoise 网络权重占位（SDK 版）。
 *
 * 本交付包的网络权重已内嵌到 AXMODEL 中，compute_rnn 由 AX Engine 推理替换，
 * 不再需要原版 rnnoise_data.c 的 75MB 权重数组。此处仅保留空权重表与
 * init_rnnoise 空实现（memset 清零），以保持原版 denoise.c 的初始化链路。
 */
#include <string.h>

#include "rnnoise_data.h"

const WeightArray rnnoise_arrays[] = {{NULL, 0, 0, NULL}};
const WeightArray n[] = {{NULL, 0, 0, NULL}};

int init_rnnoise(RNNoise *model, const WeightArray *arrays) {
    (void)arrays;
    memset(model, 0, sizeof(*model));
    return 0;
}
