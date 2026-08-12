#pragma once

#include <string>
#include <vector>

// AX Engine 会话封装：加载 AXMODEL，按张量名映射输入输出。
class ModelRunner {
public:
    explicit ModelRunner(const std::string& model_path,
                         const std::string& model_name = "rnnoise");
    ~ModelRunner();

    ModelRunner(const ModelRunner&) = delete;
    ModelRunner& operator=(const ModelRunner&) = delete;

    // 6 输入（features/conv1_mem/conv2_mem/gru1_s/gru2_s/gru3_s）
    // -> 7 输出（gains/vad/conv1_mem_new/conv2_mem_new/gru1_s_new/gru2_s_new/gru3_s_new）
    std::vector<std::vector<float>> Run(
        const std::vector<std::vector<float>>& inputs);

private:
    struct Impl;
    Impl* impl_;
};
