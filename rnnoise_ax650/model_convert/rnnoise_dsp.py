"""RNNoise 信号处理链的 numpy 移植（1:1 对应 origin/rnnoise/src 的 C 实现）。

覆盖 denoise.c / pitch.c / kiss_fft / celt_lpc / rnnoise_tables 的浮点路径：
  PCM 帧(480) -> biquad HP -> FFT(960) -> 32 波段能量/DCT + pitch -> 65 维特征
  gains/vad -> gain 平滑/限幅 -> pitch filter -> 频谱合成 -> 480 样本

约定：输入音频为 16-bit PCM 等价 float（±32768 量级，不做 /32768 归一化，
与官方 rnnoise demo 一致）。所有运算尽量 float32，与 C float 语义对齐。
"""
from __future__ import annotations

import numpy as np

FRAME_SIZE = 480
WINDOW_SIZE = 2 * FRAME_SIZE
FREQ_SIZE = FRAME_SIZE + 1
NB_BANDS = 32
NB_FEATURES = 2 * NB_BANDS + 1

PITCH_MIN_PERIOD = 60
PITCH_MAX_PERIOD = 768
PITCH_FRAME_SIZE = 960
PITCH_BUF_SIZE = PITCH_MAX_PERIOD + PITCH_FRAME_SIZE

EBAND20MS = np.array(
    [0, 2, 4, 6, 8, 10, 12, 15, 18, 21, 24, 28, 32, 36, 41, 47, 53, 60,
     68, 77, 87, 98, 110, 124, 140, 157, 176, 198, 223, 251, 282, 317, 356,
     400], dtype=np.int32)

SECOND_CHECK = np.array(
    [0, 0, 3, 2, 3, 2, 5, 2, 3, 2, 3, 2, 5, 2, 3, 2], dtype=np.int32)


def f32(x):
    return np.asarray(x, dtype=np.float32)


def make_half_window() -> np.ndarray:
    """rnn_half_window（dump_rnnoise_tables.c 公式）。"""
    i = (np.arange(FRAME_SIZE, dtype=np.float64) + 0.5)
    s = np.sin(0.5 * np.pi * i / FRAME_SIZE)
    w = np.sin(0.5 * np.pi * s * s)
    return f32(w)


HALF_WINDOW = make_half_window()


def make_dct_table() -> np.ndarray:
    """rnn_dct_table：行 i、列 j 为 cos((i+.5)*j*pi/32)，j==0 乘 sqrt(.5)。"""
    i = np.arange(NB_BANDS, dtype=np.float64)[:, None]
    j = np.arange(NB_BANDS, dtype=np.float64)[None, :]
    t = np.cos((i + 0.5) * j * np.pi / NB_BANDS)
    t[:, 0] *= np.sqrt(0.5)
    return f32(t)


DCT_TABLE = make_dct_table()


def fft(x: np.ndarray) -> np.ndarray:
    """forward FFT：kiss_fft 无缩放正变换（960 点），返回复数组。"""
    return np.fft.fft(f32(x).astype(np.float64))


def forward_transform(x: np.ndarray) -> np.ndarray:
    """x[960] -> X[481]（kiss_fft 该 fork 正变换带 1/N 缩放）。"""
    y = fft(x) / WINDOW_SIZE
    return y[:FREQ_SIZE]


def inverse_transform(y: np.ndarray) -> np.ndarray:
    """X[481] -> x[960]（Hermitian 镜像 + 正变换实现逆变换）。"""
    y = np.asarray(y)
    n = WINDOW_SIZE
    x = np.zeros(n, dtype=np.complex128)
    x[:FREQ_SIZE] = y
    x[FREQ_SIZE:] = np.conj(y[1:(n - FREQ_SIZE) + 1][::-1])
    out = np.fft.fft(x)
    res = np.empty(n, dtype=np.float64)
    res[0] = out[0].real
    res[1:] = out[n:0:-1].real
    return res


def apply_window(x: np.ndarray) -> np.ndarray:
    x = x.copy()
    x[:FRAME_SIZE] *= HALF_WINDOW
    x[WINDOW_SIZE - 1:FRAME_SIZE - 1:-1] *= HALF_WINDOW
    return x


def compute_band_energy(X: np.ndarray) -> np.ndarray:
    """X[481] 复数 -> bandE[32]。"""
    X = np.asarray(X)
    power = (X.real ** 2 + X.imag ** 2).astype(np.float64)
    # C 实现用 sum[0..NB_BANDS+1]，sum[1] 和 sum[NB_BANDS] 修正后取 sum[1..NB_BANDS]
    s = np.zeros(NB_BANDS + 2)
    for i in range(NB_BANDS):
        b0, b1 = int(EBAND20MS[i]), int(EBAND20MS[i + 1])
        frac = (np.arange(b1 - b0, dtype=np.float64)) / (b1 - b0)
        p = power[b0:b1]
        s[i] += np.sum((1 - frac) * p)
        s[i + 1] += np.sum(frac * p)
    s[1] = (s[0] + s[1]) * 2.0 / 3.0
    s[NB_BANDS] = (s[NB_BANDS] + s[NB_BANDS + 1]) * 2.0 / 3.0
    return f32(s[1:NB_BANDS + 1])


def compute_band_corr(X: np.ndarray, P: np.ndarray) -> np.ndarray:
    X = np.asarray(X)
    P = np.asarray(P)
    prod = (X.real * P.real + X.imag * P.imag).astype(np.float64)
    s = np.zeros(NB_BANDS + 2)
    for i in range(NB_BANDS):
        b0, b1 = int(EBAND20MS[i]), int(EBAND20MS[i + 1])
        frac = (np.arange(b1 - b0, dtype=np.float64)) / (b1 - b0)
        p = prod[b0:b1]
        s[i] += np.sum((1 - frac) * p)
        s[i + 1] += np.sum(frac * p)
    s[1] = (s[0] + s[1]) * 2.0 / 3.0
    s[NB_BANDS] = (s[NB_BANDS] + s[NB_BANDS + 1]) * 2.0 / 3.0
    return f32(s[1:NB_BANDS + 1])


def dct(inp: np.ndarray) -> np.ndarray:
    """out[i] = sqrt(2/22) * sum_j in[j] * table[j, i]。"""
    inp = f32(inp)
    return f32(inp @ DCT_TABLE * np.sqrt(2.0 / 22.0))


def rnn_biquad(x: np.ndarray, mem: np.ndarray, b, a) -> tuple[np.ndarray, np.ndarray]:
    x = f32(x)
    y = np.empty_like(x)
    mem = mem.copy()
    for i in range(x.size):
        xi = float(x[i])
        yi = xi + mem[0]
        mem[0] = mem[1] + (b[0] * xi - a[0] * yi)
        mem[1] = b[1] * xi - a[1] * yi
        y[i] = yi
    return y, mem


def rnn_autocorr(x: np.ndarray, lag: int) -> np.ndarray:
    """rnn_autocorr(overlap=0)：ac[k] = sum_i x[i]*x[i+k]。"""
    n = x.size
    x = f32(x)
    ac = np.zeros(lag + 1, dtype=np.float64)
    for k in range(lag + 1):
        ac[k] = np.dot(x[:n - k].astype(np.float64), x[k:].astype(np.float64))
    return ac


def rnn_lpc(ac: np.ndarray, p: int) -> np.ndarray:
    ac = np.asarray(ac, dtype=np.float64)
    lpc = np.zeros(p, dtype=np.float64)
    error = ac[0]
    if ac[0] != 0:
        for i in range(p):
            rr = sum(lpc[j] * ac[i - j] for j in range(i)) + ac[i + 1]
            r = -rr / error
            lpc[i] = r
            for j in range((i + 1) // 2):
                tmp1 = lpc[j]
                tmp2 = lpc[i - 1 - j]
                lpc[j] = tmp1 + r * tmp2
                lpc[i - 1 - j] = tmp2 + r * tmp1
            error = error - r * r * error
            if error < 0.001 * ac[0]:
                break
    return f32(lpc)


def celt_fir5(x: np.ndarray, num: np.ndarray) -> np.ndarray:
    """5 抽头 FIR：y[i] = x[i] + sum_j num[j]*x[i-1-j]。"""
    x = f32(x)
    num = f32(num)
    n = x.size
    y = np.empty(n, dtype=np.float32)
    mem = np.zeros(5, dtype=np.float64)
    for i in range(n):
        xi = float(x[i])
        s = xi + float(num[0]) * mem[0] + float(num[1]) * mem[1] + \
            float(num[2]) * mem[2] + float(num[3]) * mem[3] + float(num[4]) * mem[4]
        mem[4] = mem[3]
        mem[3] = mem[2]
        mem[2] = mem[1]
        mem[1] = mem[0]
        mem[0] = xi
        y[i] = s
    return y


def rnn_pitch_downsample(x: np.ndarray) -> np.ndarray:
    """1728 采样 -> 864 低通（2:1 抽取 + LPC 预加重）。"""
    x = f32(x)
    length = x.size
    half = length // 2
    x_lp = np.empty(half, dtype=np.float32)
    x_lp[0] = 0.5 * (0.5 * float(x[1]) + float(x[0]))
    for i in range(1, half):
        x_lp[i] = 0.5 * (0.5 * (float(x[2 * i - 1]) + float(x[2 * i + 1])) + float(x[2 * i]))
    ac = rnn_autocorr(x_lp, 4)
    ac[0] *= 1.0001
    for i in range(1, 5):
        ac[i] -= ac[i] * (0.008 * i) * (0.008 * i)
    lpc = rnn_lpc(ac, 4)
    tmp = 1.0
    for i in range(4):
        tmp *= 0.9
        lpc[i] = lpc[i] * tmp
    lpc2 = np.zeros(5, dtype=np.float32)
    c1 = 0.8
    lpc2[0] = lpc[0] + 0.8
    lpc2[1] = lpc[1] + c1 * lpc[0]
    lpc2[2] = lpc[2] + c1 * lpc[1]
    lpc2[3] = lpc[3] + c1 * lpc[2]
    lpc2[4] = c1 * lpc[3]
    return celt_fir5(x_lp, lpc2)


def _find_best_pitch(xcorr: np.ndarray, y: np.ndarray, length: int, max_pitch: int):
    """find_best_pitch（浮点路径），返回 best_pitch[2]。"""
    xcorr = np.asarray(xcorr, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    Syy = 1.0 + np.dot(y[:length], y[:length])
    best_num = [-1.0, -1.0]
    best_den = [0.0, 0.0]
    best_pitch = [0, 1]
    for i in range(max_pitch):
        if xcorr[i] > 0:
            xcorr16 = xcorr[i] * 1e-12
            num = xcorr16 * xcorr16
            if num * best_den[1] > best_num[1] * Syy:
                if num * best_den[0] > best_num[0] * Syy:
                    best_num[1] = best_num[0]
                    best_den[1] = best_den[0]
                    best_pitch[1] = best_pitch[0]
                    best_num[0] = num
                    best_den[0] = Syy
                    best_pitch[0] = i
                else:
                    best_num[1] = num
                    best_den[1] = Syy
                    best_pitch[1] = i
        Syy += y[i + length] * y[i + length] - y[i] * y[i]
        Syy = max(1.0, Syy)
    return best_pitch


def rnn_pitch_search(x_lp: np.ndarray, y: np.ndarray, length: int, max_pitch: int) -> int:
    """rnn_pitch_search 返回 pitch（粗搜 + 细搜 + 伪插值）。"""
    x_lp = f32(x_lp)
    y = f32(y)
    lag = length + max_pitch
    len4 = length >> 2
    x_lp4 = x_lp[0:length:2][:len4]
    y_lp4 = y[0:lag:2][:(lag >> 2)]
    xcorr = np.array([np.dot(x_lp4.astype(np.float64),
                             y_lp4[i:i + len4].astype(np.float64))
                      for i in range(max_pitch >> 2)], dtype=np.float64)
    best_pitch = _find_best_pitch(xcorr, y_lp4, len4, max_pitch >> 2)
    half = max_pitch >> 1
    xcorr2 = np.zeros(half, dtype=np.float64)
    x_lp64 = x_lp.astype(np.float64)
    y64 = y.astype(np.float64)
    for i in range(half):
        if abs(i - 2 * best_pitch[0]) > 2 and abs(i - 2 * best_pitch[1]) > 2:
            continue
        s = np.dot(x_lp64[:length >> 1], y64[i:i + (length >> 1)])
        xcorr2[i] = max(-1.0, s)
    best_pitch = _find_best_pitch(xcorr2, y64, length >> 1, half)
    bp = best_pitch[0]
    if 0 < bp < half - 1:
        a, b, c = xcorr2[bp - 1], xcorr2[bp], xcorr2[bp + 1]
        if (c - a) > 0.7 * (b - a):
            offset = 1
        elif (a - c) > 0.7 * (b - c):
            offset = -1
        else:
            offset = 0
    else:
        offset = 0
    return 2 * bp - offset


def _compute_pitch_gain(xy, xx, yy):
    return xy / np.sqrt(1.0 + xx * yy)


def rnn_remove_doubling(x: np.ndarray, maxperiod: int, minperiod: int, n: int,
                        t0_in: int, prev_period: int, prev_gain: float):
    """rnn_remove_doubling：返回 (pg, T0)。"""
    x = f32(x)
    minperiod0 = minperiod
    maxperiod //= 2
    minperiod //= 2
    t0 = t0_in // 2
    prev_period //= 2
    n //= 2
    x = x[maxperiod:]
    if t0 >= maxperiod:
        t0 = maxperiod - 1
    T = T0 = t0
    x64 = x.astype(np.float64)
    xx = np.dot(x64[:n], x64[:n])
    xy = np.dot(x64[:n], x64[T0:T0 + n])
    yy_lookup = np.zeros(maxperiod + 1, dtype=np.float64)
    yy_lookup[0] = xx
    yy = xx
    for i in range(1, maxperiod + 1):
        yy = yy + float(x64[-i]) ** 2 - float(x64[n - i]) ** 2
        yy_lookup[i] = max(0.0, yy)
    yy = yy_lookup[T0]
    best_xy, best_yy = xy, yy
    g = g0 = _compute_pitch_gain(xy, xx, yy)
    for k in range(2, 16):
        T1 = (2 * T0 + k) // (2 * k)
        if T1 < minperiod:
            break
        if k == 2:
            T1b = T0 if T1 + T0 > maxperiod else T0 + T1
        else:
            T1b = (2 * int(SECOND_CHECK[k]) * T0 + k) // (2 * k)
        xy2 = np.dot(x64[:n], x64[T1b:T1b + n])
        xy = 0.5 * (xy + xy2)
        yy = 0.5 * (yy_lookup[T1] + yy_lookup[T1b])
        g1 = _compute_pitch_gain(xy, xx, yy)
        if abs(T1 - prev_period) <= 1:
            cont = prev_gain
        elif abs(T1 - prev_period) <= 2 and 5 * k * k < T0:
            cont = 0.5 * prev_gain
        else:
            cont = 0.0
        thresh = max(0.3, 0.7 * g0 - cont)
        if T1 < 3 * minperiod:
            thresh = max(0.4, 0.85 * g0 - cont)
        elif T1 < 2 * minperiod:
            thresh = max(0.5, 0.9 * g0 - cont)
        if g1 > thresh:
            best_xy, best_yy = xy, yy
            T = T1
            g = g1
    best_xy = max(0.0, best_xy)
    pg = 1.0 if best_yy <= best_xy else best_xy / (best_yy + 1.0)
    xcorr = np.zeros(3)
    for k in range(3):
        lag = T + k - 1
        xcorr[k] = np.dot(x64[:n], x64[lag:lag + n])
    if (xcorr[2] - xcorr[0]) > 0.7 * (xcorr[1] - xcorr[0]):
        offset = 1
    elif (xcorr[0] - xcorr[2]) > 0.7 * (xcorr[1] - xcorr[2]):
        offset = -1
    else:
        offset = 0
    if pg > g:
        pg = g
    T0 = 2 * T + offset
    if T0 < minperiod0:
        T0 = minperiod0
    return float(pg), int(T0)


def interp_band_gain(bandE: np.ndarray) -> np.ndarray:
    bandE = np.asarray(bandE, dtype=np.float64)
    g = np.zeros(FREQ_SIZE)
    for i in range(1, NB_BANDS):
        b0, b1 = int(EBAND20MS[i]), int(EBAND20MS[i + 1])
        frac = (np.arange(b1 - b0, dtype=np.float64)) / (b1 - b0)
        g[b0:b1] = (1 - frac) * bandE[i - 1] + frac * bandE[i]
    g[:int(EBAND20MS[1])] = bandE[0]
    g[int(EBAND20MS[NB_BANDS]):int(EBAND20MS[NB_BANDS + 1])] = bandE[NB_BANDS - 1]
    return f32(g)


def rnn_pitch_filter(X, P, Ex, Ep, Exp, g):
    """rnn_pitch_filter：就地修改 X[481]。"""
    X = np.asarray(X, dtype=np.complex128)
    P = np.asarray(P, dtype=np.complex128)
    Ex = np.asarray(Ex, dtype=np.float64)
    Ep = np.asarray(Ep, dtype=np.float64)
    Exp = np.asarray(Exp, dtype=np.float64)
    g = np.asarray(g, dtype=np.float64)
    r = np.zeros(NB_BANDS)
    for i in range(NB_BANDS):
        if Exp[i] > g[i]:
            r[i] = 1.0
        else:
            r[i] = (Exp[i] ** 2) * (1 - g[i] ** 2) / (0.001 + g[i] ** 2 * (1 - Exp[i] ** 2))
            r[i] = np.sqrt(min(1.0, max(0.0, r[i])))
        r[i] *= np.sqrt(Ex[i] / (1e-8 + Ep[i]))
    rf = interp_band_gain(r)
    X += rf * P
    newE = compute_band_energy(X)
    norm = np.sqrt(Ex / (1e-8 + newE))
    normf = interp_band_gain(norm)
    X *= normf
    return X


class RNNoiseState:
    """对应 DenoiseState：维护全部帧间状态。"""

    def __init__(self):
        self.analysis_mem = np.zeros(FRAME_SIZE, dtype=np.float32)
        self.synthesis_mem = np.zeros(FRAME_SIZE, dtype=np.float32)
        self.pitch_buf = np.zeros(PITCH_BUF_SIZE, dtype=np.float32)
        self.last_gain = 0.0
        self.last_period = 0
        self.mem_hp_x = np.zeros(2, dtype=np.float32)
        self.lastg = np.zeros(NB_BANDS, dtype=np.float32)
        self.delayed_X = np.zeros(FREQ_SIZE, dtype=np.complex128)
        self.delayed_P = np.zeros(FREQ_SIZE, dtype=np.complex128)
        self.delayed_Ex = np.zeros(NB_BANDS, dtype=np.float32)
        self.delayed_Ep = np.zeros(NB_BANDS, dtype=np.float32)
        self.delayed_Exp = np.zeros(NB_BANDS, dtype=np.float32)


def rnn_frame_analysis(st: RNNoiseState, inp: np.ndarray):
    """返回 (X[481], Ex[32])。"""
    x = np.concatenate([st.analysis_mem, f32(inp)])
    st.analysis_mem = f32(inp).copy()
    x = apply_window(x)
    X = forward_transform(x)
    Ex = compute_band_energy(X)
    return X, Ex


def compute_frame_features(st: RNNoiseState, inp: np.ndarray):
    """返回 (silence, X, P, Ex, Ep, Exp, features)。features 为空时清为 0。"""
    X, Ex = rnn_frame_analysis(st, inp)
    st.pitch_buf = np.concatenate([st.pitch_buf[FRAME_SIZE:], f32(inp)])
    pre = st.pitch_buf
    pitch_buf_lp = rnn_pitch_downsample(pre)
    x_lp = pitch_buf_lp[(PITCH_MAX_PERIOD >> 1):]
    y = pitch_buf_lp
    pitch_index = rnn_pitch_search(
        x_lp, y, PITCH_FRAME_SIZE, PITCH_MAX_PERIOD - 3 * PITCH_MIN_PERIOD)
    pitch_index = PITCH_MAX_PERIOD - pitch_index
    gain, pitch_index = rnn_remove_doubling(
        st.pitch_buf, PITCH_MAX_PERIOD, PITCH_MIN_PERIOD, PITCH_FRAME_SIZE,
        pitch_index, st.last_period, st.last_gain)
    st.last_period = pitch_index
    st.last_gain = gain
    p = np.array([
        st.pitch_buf[PITCH_BUF_SIZE - WINDOW_SIZE - pitch_index + i]
        for i in range(WINDOW_SIZE)], dtype=np.float32)
    p = apply_window(p)
    P = forward_transform(p)
    Ep = compute_band_energy(P)
    Exp = compute_band_corr(X, P)
    Exp = Exp / np.sqrt(0.001 + Ex * Ep)
    features = np.zeros(NB_FEATURES, dtype=np.float32)
    features[NB_BANDS:2 * NB_BANDS] = dct(Exp)
    features[2 * NB_BANDS] = 0.01 * (pitch_index - 300)
    logMax = -2.0
    follow = -2.0
    Ly = np.zeros(NB_BANDS)
    E = 0.0
    for i in range(NB_BANDS):
        Ly[i] = np.log10(1e-2 + float(Ex[i]))
        Ly[i] = max(logMax - 7, max(follow - 1.5, Ly[i]))
        logMax = max(logMax, Ly[i])
        follow = max(follow - 1.5, Ly[i])
        E += float(Ex[i])
    if E < 0.04:
        features[:] = 0.0
        return 1, X, P, Ex, Ep, Exp, features
    features[:NB_BANDS] = dct(Ly)
    features[0] -= 12.0
    features[1] -= 4.0
    return 0, X, P, Ex, Ep, Exp, features


def frame_synthesis(st: RNNoiseState, y) -> np.ndarray:
    x = inverse_transform(y)
    x = apply_window(x)
    out = f32(x[:FRAME_SIZE] + st.synthesis_mem)
    st.synthesis_mem = f32(x[FRAME_SIZE:])
    return out


def analyze_frame(st: RNNoiseState, inp: np.ndarray) -> dict:
    """C 端 rnnoise_process_frame 前半：biquad + 特征分析。

    返回 dict(silence, features, X, P, Ex, Ep, Exp)。会推进 st 的
    analysis_mem / pitch_buf / last_period / last_gain 等 DSP 状态。
    """
    a_hp = np.array([-1.99599, 0.99600], dtype=np.float32)
    b_hp = np.array([-2, 1], dtype=np.float32)
    x, st.mem_hp_x = rnn_biquad(f32(inp), st.mem_hp_x, b_hp, a_hp)
    silence, X, P, Ex, Ep, Exp, features = compute_frame_features(st, x)
    return {"silence": silence, "features": features,
            "X": X, "P": P, "Ex": Ex, "Ep": Ep, "Exp": Exp}


def synthesize_frame(st: RNNoiseState, ana: dict, gains, vad):
    """C 端 rnnoise_process_frame 后半：pitch filter + 增益 + 频谱合成。

    gains 为 None 表示静音帧（跳过模型相关部分，vad 记 0）。
    返回 (out_frame, vad_prob)。需在 analyze_frame 之后调用。
    """
    silence = ana["silence"]
    X, P, Ex, Ep, Exp = ana["X"], ana["P"], ana["Ex"], ana["Ep"], ana["Exp"]
    if not silence and gains is not None:
        gains = f32(gains).reshape(-1)
        st.delayed_X = rnn_pitch_filter(
            st.delayed_X, st.delayed_P, st.delayed_Ex, st.delayed_Ep,
            st.delayed_Exp, gains)
        for i in range(NB_BANDS):
            alpha = 0.6
            gains[i] = max(gains[i], alpha * st.lastg[i])
            st.lastg[i] = min(1.0, gains[i] * (st.delayed_Ex[i] + 1e-3) / (Ex[i] + 1e-3))
        gf = interp_band_gain(gains)
        st.delayed_X = st.delayed_X * gf
    out = frame_synthesis(st, st.delayed_X)
    st.delayed_X = X
    st.delayed_P = P
    st.delayed_Ex = Ex
    st.delayed_Ep = Ep
    st.delayed_Exp = Exp
    vad_prob = 0.0 if (silence or gains is None) else float(np.asarray(vad).reshape(-1)[0])
    return out, vad_prob


def process_frame(st: RNNoiseState, inp: np.ndarray, gains, vad):
    """完整一帧（C 端 rnnoise_process_frame 等价），返回 (out, vad_prob)。"""
    ana = analyze_frame(st, inp)
    if ana["silence"]:
        return synthesize_frame(st, ana, None, 0.0)
    return synthesize_frame(st, ana, gains, vad)


def features_from_pcm(pcm: np.ndarray) -> np.ndarray:
    """批量：48k f32 PCM（16-bit 域）-> (T, 65) 特征矩阵（含 silence 清零逻辑）。"""
    st = RNNoiseState()
    feats = []
    pcm = f32(pcm)
    for t in range(0, pcm.size - FRAME_SIZE + 1, FRAME_SIZE):
        frame = pcm[t:t + FRAME_SIZE]
        a_hp = np.array([-1.99599, 0.99600], dtype=np.float32)
        b_hp = np.array([-2, 1], dtype=np.float32)
        x, st.mem_hp_x = rnn_biquad(frame, st.mem_hp_x, b_hp, a_hp)
        _, _, _, _, _, _, features = compute_frame_features(st, x)
        feats.append(features.copy())
    return np.stack(feats)
