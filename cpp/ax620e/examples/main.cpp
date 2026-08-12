// RNNoise AX620E 示例：处理 48k f32 PCM（16-bit 等价域），输出去噪 PCM。
// 用法: ./model_example model.axmodel in.pcm out.pcm
#include "rnnoise_ax.hpp"

#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 4) {
        std::fprintf(stderr, "用法: %s model.axmodel in.pcm out.pcm\n", argv[0]);
        return 1;
    }
    FILE* in = std::fopen(argv[2], "rb");
    if (!in) {
        std::fprintf(stderr, "无法打开输入 %s\n", argv[2]);
        return 1;
    }
    std::fseek(in, 0, SEEK_END);
    long bytes = std::ftell(in);
    std::fseek(in, 0, SEEK_SET);
    std::vector<float> pcm(bytes / sizeof(float));
    if (!pcm.empty()) {
        std::fread(pcm.data(), sizeof(float), pcm.size(), in);
    }
    std::fclose(in);

    const int frame = RNNoiseAX::FrameSize();
    const int frames = static_cast<int>(pcm.size() / frame);
    if (frames == 0) {
        std::fprintf(stderr, "输入过短\n");
        return 1;
    }

    RNNoiseAX denoiser(argv[1]);
    std::vector<float> out(pcm.size());
    double vad_sum = 0.0;
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < frames; ++i) {
        vad_sum += denoiser.ProcessFrame(
            &out[i * frame], &pcm[i * frame]);
    }
    auto t1 = std::chrono::steady_clock::now();
    double secs = std::chrono::duration<double>(t1 - t0).count();

    FILE* of = std::fopen(argv[3], "wb");
    if (!of) {
        std::fprintf(stderr, "无法写入输出 %s\n", argv[3]);
        return 1;
    }
    std::fwrite(out.data(), sizeof(float), out.size(), of);
    std::fclose(of);

    std::printf("frames=%d vad_mean=%.4f per_frame_ms=%.3f out=%s\n",
                frames, vad_sum / frames, secs / frames * 1000.0, argv[3]);
    return 0;
}
