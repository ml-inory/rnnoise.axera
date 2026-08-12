#pragma once

#include <memory>
#include <string>

// RNNoise AX650 实时降噪器：原版 rnnoise C 信号处理 + AX Engine 网络推理。
class RNNoiseAX {
public:
    explicit RNNoiseAX(const std::string& model_path);
    ~RNNoiseAX();

    RNNoiseAX(const RNNoiseAX&) = delete;
    RNNoiseAX& operator=(const RNNoiseAX&) = delete;

    static int FrameSize() { return 480; }

    void Reset();

    // 处理一帧 48k PCM（16-bit 等价 float，±32768 域）。
    // in/out 各至少 FrameSize() 个 float；返回 vad（0~1）。
    float ProcessFrame(float* out, const float* in);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
