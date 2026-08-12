// RNNoise AX SDK：用 AX Engine 替换原版 compute_rnn（网络推理），
// 其余信号处理（biquad/FFT/pitch/合成）沿用原版 C 实现。
#include "rnnoise_ax.hpp"

#include "model_runner.hpp"

#include <cstring>
#include <mutex>
#include <stdexcept>
#include <vector>

extern "C" {
#include "denoise.h"
#include "rnn.h"
}

namespace {

// 单实例模型 runner（compute_rnn 无状态上下文可用，采用全局注册方式；
// 同一时刻只允许一个 RNNoiseAX 实例执行推理）。
std::mutex g_runner_mu;
ModelRunner* g_ax_runner = nullptr;

extern "C" void rnnoise_ax_set_runner(ModelRunner* runner) {
    std::lock_guard<std::mutex> lk(g_runner_mu);
    g_ax_runner = runner;
}

extern "C" void compute_rnn(const RNNoise* model, RNNState* rnn,
                            float* gains, float* vad,
                            const float* input, int arch) {
    (void)model;
    (void)arch;
    ModelRunner* runner = nullptr;
    {
        std::lock_guard<std::mutex> lk(g_runner_mu);
        runner = g_ax_runner;
    }
    if (runner == nullptr) {
        throw std::runtime_error("compute_rnn: AX runner 未初始化");
    }

    std::vector<std::vector<float>> feeds = {
        std::vector<float>(input, input + NB_FEATURES),
        std::vector<float>(rnn->conv1_state,
                           rnn->conv1_state + CONV1_STATE_SIZE),
        std::vector<float>(rnn->conv2_state,
                           rnn->conv2_state + CONV2_STATE_SIZE),
        std::vector<float>(rnn->gru1_state,
                           rnn->gru1_state + GRU1_OUT_SIZE),
        std::vector<float>(rnn->gru2_state,
                           rnn->gru2_state + GRU2_OUT_SIZE),
        std::vector<float>(rnn->gru3_state,
                           rnn->gru3_state + GRU3_OUT_SIZE),
    };
    std::vector<std::vector<float>> outs = runner->Run(feeds);

    std::memcpy(gains, outs[0].data(), NB_BANDS * sizeof(float));
    *vad = outs[1][0];
    std::memcpy(rnn->conv1_state, outs[2].data(),
                CONV1_STATE_SIZE * sizeof(float));
    std::memcpy(rnn->conv2_state, outs[3].data(),
                CONV2_STATE_SIZE * sizeof(float));
    std::memcpy(rnn->gru1_state, outs[4].data(),
                GRU1_OUT_SIZE * sizeof(float));
    std::memcpy(rnn->gru2_state, outs[5].data(),
                GRU2_OUT_SIZE * sizeof(float));
    std::memcpy(rnn->gru3_state, outs[6].data(),
                GRU3_OUT_SIZE * sizeof(float));
}

}  // namespace

struct RNNoiseAX::Impl {
    std::unique_ptr<ModelRunner> runner;
    DenoiseState* state = nullptr;

    explicit Impl(const std::string& model_path)
        : runner(new ModelRunner(model_path)),
          state(rnnoise_create(nullptr)) {
        if (state == nullptr) {
            throw std::runtime_error("rnnoise_create 失败");
        }
        rnnoise_ax_set_runner(runner.get());
    }

    ~Impl() {
        rnnoise_ax_set_runner(nullptr);
        if (state) {
            rnnoise_destroy(state);
        }
    }
};

RNNoiseAX::RNNoiseAX(const std::string& model_path)
    : impl_(new Impl(model_path)) {}

RNNoiseAX::~RNNoiseAX() = default;

void RNNoiseAX::Reset() {
    if (!impl_) {
        return;
    }
    rnnoise_ax_set_runner(nullptr);
    rnnoise_destroy(impl_->state);
    impl_->state = rnnoise_create(nullptr);
    if (impl_->state == nullptr) {
        throw std::runtime_error("rnnoise_create 失败");
    }
    rnnoise_ax_set_runner(impl_->runner.get());
}

float RNNoiseAX::ProcessFrame(float* out, const float* in) {
    return rnnoise_process_frame(impl_->state, out, in);
}
