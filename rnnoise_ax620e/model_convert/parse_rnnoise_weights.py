"""从官方 rnnoise_data.c 解析权重为 PyTorch 张量。

C 端线性层布局（compute_linear）：
  y = W @ x + b；GRU 稀疏权重(float_weights + weights_idx + diag)重建为稠密矩阵。
稀疏格式（sparse_sgemv8x4）：
  idx: 每 8 输出行为一组，组内: [count, pos0, w(32), pos1, w(32), ...]
  pos 为列基址(4 对齐)，w[8x4]: w[i*4+j] 乘 x[pos+j] 累加到 out[group_base+i]
diag（仅 GRU recurrent）：out[i] += diag[i]*state[i mod N]。
"""
from __future__ import annotations

import re
import numpy as np
import torch
from pathlib import Path


_FLOAT_RE = re.compile(
    r"static const float (\w+)\[(\d+)\] = \{(.*?)\};", re.S
)
_INT_RE = re.compile(
    r"static const int (\w+)\[(\d+)\] = \{(.*?)\};", re.S
)


def _parse_numbers(text: str) -> list[float]:
    out = []
    for tok in text.split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(float(tok))
    return out


def load_c_weights(c_path: Path) -> dict[str, np.ndarray]:
    src = c_path.read_text(encoding="utf-8")
    floats: dict[str, np.ndarray] = {}
    ints: dict[str, np.ndarray] = {}
    for name, n, body in _FLOAT_RE.findall(src):
        v = _parse_numbers(body)
        assert len(v) == int(n), f"{name}: {len(v)} != {n}"
        floats[name] = np.asarray(v, dtype=np.float32)
    for name, n, body in _INT_RE.findall(src):
        v = [int(float(t)) for t in _parse_numbers(body)]
        assert len(v) == int(n), f"{name}: {len(v)} != {n}"
        ints[name] = np.asarray(v, dtype=np.int32)
    return {"float": floats, "int": ints}


def reconstruct_sparse(float_w: np.ndarray, idx: np.ndarray,
                       rows: int, cols: int) -> np.ndarray:
    """重建稠密 [rows, cols] 权重矩阵（sparse_sgemv8x4 布局）。"""
    dense = np.zeros((rows, cols), dtype=np.float32)
    pos = 0
    wpos = 0
    for group in range(0, rows, 8):
        count = int(idx[pos]); pos += 1
        for _ in range(count):
            col = int(idx[pos]); pos += 1
            block = float_w[wpos:wpos + 32]
            for i in range(8):
                for j in range(4):
                    # 块内列主序: w[j*8 + i] 对应 (行=group+i, 列=col+j)
                    dense[group + i, col + j] = block[j * 8 + i]
            wpos += 32
    assert wpos == len(float_w), f"sparse 权重未消费: {wpos}/{len(float_w)}"
    return dense


def build_state_dict(c_path: Path) -> dict[str, torch.Tensor]:
    w = load_c_weights(c_path)
    f, i = w["float"], w["int"]

    def dense(name: str, rows: int, cols: int) -> np.ndarray:
        # C 稠密权重列主序: W[in*rows + out] -> torch [out, in]
        return f[name].reshape(cols, rows).T.copy()

    def sparse(name: str, rows: int, cols: int) -> np.ndarray:
        return reconstruct_sparse(f[name], i[name.replace("_weights_float", "_weights_idx")],
                                  rows, cols)

    sd = {}
    # 布局与 C 的 compute_linear 一致：y[out] = W[out, in] @ x[in] + b
    sd["conv1.w"] = torch.tensor(dense("conv1_weights_float", 128, 195))
    sd["conv1.b"] = torch.tensor(f["conv1_bias"])
    sd["conv2.w"] = torch.tensor(dense("conv2_weights_float", 384, 384))
    sd["conv2.b"] = torch.tensor(f["conv2_bias"])

    for g in ("gru1", "gru2", "gru3"):
        win = sparse(f"{g}_input_weights_float", 1152, 384)
        wrec = sparse(f"{g}_recurrent_weights_float", 1152, 384)
        # diag: dense[i, i mod 384] += diag[i]
        diag = f[f"{g}_recurrent_weights_diag"]
        assert len(diag) == 1152
        wrec[np.arange(1152), np.arange(1152) % 384] += diag
        sd[f"{g}.w_in"] = torch.tensor(win)
        sd[f"{g}.w_rec"] = torch.tensor(wrec)
        sd[f"{g}.b_in"] = torch.tensor(f[f"{g}_input_bias"])
        sd[f"{g}.b_rec"] = torch.tensor(f[f"{g}_recurrent_bias"])

    sd["dense_out.w"] = torch.tensor(dense("dense_out_weights_float", 32, 1536))
    sd["dense_out.b"] = torch.tensor(f["dense_out_bias"])
    sd["vad_dense.w"] = torch.tensor(dense("vad_dense_weights_float", 1, 1536))
    sd["vad_dense.b"] = torch.tensor(f["vad_dense_bias"])
    return sd


if __name__ == "__main__":
    import sys
    c_path = Path(sys.argv[1] if len(sys.argv) > 1 else
                  "origin/rnnoise/src/rnnoise_data.c")
    out = Path(sys.argv[2] if len(sys.argv) > 2 else "export/rnnoise_state.pt")
    sd = build_state_dict(c_path)
    torch.save({"state_dict": sd}, out)
    total = sum(v.numel() for v in sd.values())
    print(f"saved {out}: {total} params, keys={len(sd)}")
