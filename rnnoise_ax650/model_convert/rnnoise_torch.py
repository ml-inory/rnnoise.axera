"""RNNoise 逐帧状态化 PyTorch 参考模型（与 C 端 compute_rnn 1:1）。

实现要点：
- conv1d 等效为 [mem(2帧) || 当前帧] 的 MatMul（C 端 compute_generic_conv1d）；
- GRU 按 C 端 compute_generic_gru 的 z/r/h 顺序手工展开；
- tanh/sigmoid 复刻 C 端 tanh_approx / sigmoid_approx 有理逼近；
- 输入输出均为静态 shape，可直接 torch.onnx.export。
"""
from __future__ import annotations

import torch
from torch import nn

N = 384          # GRU 尺寸
FEAT = 65        # 特征维
NB_BANDS = 32


def tanh_approx(x: torch.Tensor) -> torch.Tensor:
    N0, N1, N2 = 952.52801514, 96.39235687, 0.60863042
    D0, D1, D2 = 952.72399902, 413.36801147, 11.88600922
    x2 = x * x
    num = (N2 * x2 + N1) * x2 + N0
    den = (D2 * x2 + D1) * x2 + D0
    y = num * x / den
    return torch.clamp(y, -1.0, 1.0)


def sigmoid_approx(x: torch.Tensor) -> torch.Tensor:
    return 0.5 + 0.5 * tanh_approx(0.5 * x)


class RNNoiseFrame(nn.Module):
    """单帧推理：features[*,65] + 6 个状态 -> gains[*,32], vad[*,1], 新状态。"""

    def __init__(self, state_dict: dict[str, torch.Tensor]):
        super().__init__()
        self.p = {}
        for k, v in state_dict.items():
            self.p[k] = v

    def _w(self, key: str) -> torch.Tensor:
        return self.p[key]

    def _gru(self, w_in, w_rec, b_in, b_rec, x, s):
        zrh = x @ w_in.t() + b_in
        recur = s @ w_rec.t() + b_rec
        z = sigmoid_approx(zrh[:, :N] + recur[:, :N])
        r = sigmoid_approx(zrh[:, N:2 * N] + recur[:, N:2 * N])
        h = tanh_approx(zrh[:, 2 * N:] + r * recur[:, 2 * N:])
        return z * s + (1 - z) * h

    def forward(self, features, conv1_mem, conv2_mem,
                gru1_s, gru2_s, gru3_s):
        # conv1: [mem(2x65) || features] -> 128
        c1_in = torch.cat([conv1_mem, features], dim=-1)
        conv1_out = tanh_approx(c1_in @ self._w("conv1.w").t() + self._w("conv1.b"))
        conv1_mem_new = torch.cat([conv1_mem[:, FEAT:], features], dim=-1)

        # conv2: [mem(2x128) || conv1_out] -> 384
        c2_in = torch.cat([conv2_mem, conv1_out], dim=-1)
        conv2_out = tanh_approx(c2_in @ self._w("conv2.w").t() + self._w("conv2.b"))
        conv2_mem_new = torch.cat([conv2_mem[:, 128:], conv1_out], dim=-1)

        g1 = self._gru(self._w("gru1.w_in"), self._w("gru1.w_rec"),
                       self._w("gru1.b_in"), self._w("gru1.b_rec"),
                       conv2_out, gru1_s)
        g2 = self._gru(self._w("gru2.w_in"), self._w("gru2.w_rec"),
                       self._w("gru2.b_in"), self._w("gru2.b_rec"), g1, gru2_s)
        g3 = self._gru(self._w("gru3.w_in"), self._w("gru3.w_rec"),
                       self._w("gru3.b_in"), self._w("gru3.b_rec"), g2, gru3_s)

        cat = torch.cat([conv2_out, g1, g2, g3], dim=-1)  # 1536
        gain = sigmoid_approx(cat @ self._w("dense_out.w").t() + self._w("dense_out.b"))
        vad = sigmoid_approx(cat @ self._w("vad_dense.w").t() + self._w("vad_dense.b"))
        return gain, vad, conv1_mem_new, conv2_mem_new, g1, g2, g3


def zero_state(batch: int = 1, device="cpu") -> dict[str, torch.Tensor]:
    z = torch.zeros
    return {
        "conv1_mem": z(batch, 130, device=device),
        "conv2_mem": z(batch, 256, device=device),
        "gru1_s": z(batch, N, device=device),
        "gru2_s": z(batch, N, device=device),
        "gru3_s": z(batch, N, device=device),
    }


def run_sequence(model: RNNoiseFrame, features: torch.Tensor) -> tuple:
    """features: [T, 65] -> (gains[T,32], vad[T], states 轨迹)。"""
    st = zero_state(1, device=features.device)
    outs = []
    for t in range(features.shape[0]):
        f = features[t:t + 1]
        gain, vad, m1, m2, s1, s2, s3 = model(
            f, st["conv1_mem"], st["conv2_mem"],
            st["gru1_s"], st["gru2_s"], st["gru3_s"])
        outs.append((gain, vad))
        st = {"conv1_mem": m1, "conv2_mem": m2,
              "gru1_s": s1, "gru2_s": s2, "gru3_s": s3}
    gains = torch.cat([o[0] for o in outs], dim=0)
    vads = torch.cat([o[1] for o in outs], dim=0)
    return gains, vads
