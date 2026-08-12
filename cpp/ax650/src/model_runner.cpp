#include "model_runner.hpp"

#include <ax_engine_api.h>
#include <ax_sys_api.h>

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <unordered_map>

namespace {

// 老版本 ax_sys_api.h 未声明这两个缓存同步接口，这里补充声明（板端 libax_sys 已导出）。
extern "C" AX_S32 AX_SYS_MflushCache(AX_U64 phy_addr, AX_VOID* vir_addr, AX_U32 size);
extern "C" AX_S32 AX_SYS_MinvalidateCache(AX_U64 phy_addr, AX_VOID* vir_addr, AX_U32 size);

std::vector<char> read_binary(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("failed to open " + path);
    }
    return std::vector<char>(
        std::istreambuf_iterator<char>(file),
        std::istreambuf_iterator<char>());
}

void check_ax(int ret, const char* message) {
    if (ret != 0) {
        throw std::runtime_error(message);
    }
}

const char* kInputNames[6] = {
    "features", "conv1_mem", "conv2_mem",
    "gru1_s", "gru2_s", "gru3_s",
};
const char* kOutputNames[7] = {
    "gains", "vad",
    "conv1_mem_new", "conv2_mem_new",
    "gru1_s_new", "gru2_s_new", "gru3_s_new",
};

// 兼容不同 AX SDK 版本：新版 AX_ENGINE_RunSyncV2(handle, context, io)，
// 旧版 AX_ENGINE_Run(context, io)。
namespace axrun {
template <typename H, typename C, typename IO>
auto run(H h, C c, IO* io, int)
    -> decltype(AX_ENGINE_RunSyncV2(h, c, io)) {
    return AX_ENGINE_RunSyncV2(h, c, io);
}
template <typename H, typename C, typename IO>
auto run(H /*h*/, C c, IO* io, long)
    -> decltype(AX_ENGINE_Run(c, io)) {
    return AX_ENGINE_Run(c, io);
}
}  // namespace axrun

}  // namespace

struct ModelRunner::Impl {
    AX_ENGINE_HANDLE handle = nullptr;
    AX_ENGINE_CONTEXT_T context = nullptr;
    AX_ENGINE_IO_INFO_T* info = nullptr;
    AX_ENGINE_IO_T io {};
    std::vector<AX_ENGINE_IO_BUFFER_T> buffers;
    std::vector<int> input_idx;
    std::vector<int> output_idx;
    std::vector<char> model;

    explicit Impl(const std::string& model_path, const std::string& model_name)
        : model(read_binary(model_path)) {
        check_ax(AX_SYS_Init(), "AX_SYS_Init failed");

        AX_ENGINE_NPU_ATTR_T npu_attr;
        std::memset(&npu_attr, 0, sizeof(npu_attr));
        npu_attr.eHardMode = static_cast<AX_ENGINE_NPU_MODE_T>(0);
        check_ax(AX_ENGINE_Init(&npu_attr), "AX_ENGINE_Init failed");

        AX_ENGINE_HANDLE_EXTRA_T extra;
        std::memset(&extra, 0, sizeof(extra));
        extra.pName = const_cast<AX_S8*>(
            reinterpret_cast<const AX_S8*>(model_name.c_str()));
        check_ax(
            AX_ENGINE_CreateHandleV2(
                &handle, model.data(),
                static_cast<AX_U32>(model.size()), &extra),
            "AX_ENGINE_CreateHandleV2 failed");
        check_ax(
            AX_ENGINE_CreateContextV2(handle, &context),
            "AX_ENGINE_CreateContextV2 failed");
        check_ax(AX_ENGINE_GetIOInfo(handle, &info), "AX_ENGINE_GetIOInfo failed");
        if (!info || info->nInputSize < 6 || info->nOutputSize < 7) {
            throw std::runtime_error("model IO mismatch (expect 6 in / 7 out)");
        }

        // 按张量名建立索引映射
        input_idx.resize(6, -1);
        output_idx.resize(7, -1);
        std::unordered_map<std::string, int> in_map, out_map;
        for (AX_U32 i = 0; i < info->nInputSize; ++i) {
            const char* nm = info->pInputs[i].pName;
            in_map[nm ? nm : ""] = static_cast<int>(i);
        }
        for (AX_U32 i = 0; i < info->nOutputSize; ++i) {
            const char* nm = info->pOutputs[i].pName;
            out_map[nm ? nm : ""] = static_cast<int>(i);
        }
        for (int i = 0; i < 6; ++i) {
            auto it = in_map.find(kInputNames[i]);
            if (it == in_map.end()) {
                throw std::runtime_error(std::string("missing input ") + kInputNames[i]);
            }
            input_idx[i] = it->second;
        }
        for (int i = 0; i < 7; ++i) {
            auto it = out_map.find(kOutputNames[i]);
            if (it == out_map.end()) {
                throw std::runtime_error(std::string("missing output ") + kOutputNames[i]);
            }
            output_idx[i] = it->second;
        }

        buffers.resize(info->nInputSize + info->nOutputSize);
        io.pInputs = buffers.data();
        io.nInputSize = info->nInputSize;
        io.pOutputs = buffers.data() + info->nInputSize;
        io.nOutputSize = info->nOutputSize;
        for (AX_U32 i = 0; i < info->nInputSize; ++i) {
            std::memset(&buffers[i], 0, sizeof(buffers[i]));
            buffers[i].nSize = info->pInputs[i].nSize;
            check_ax(
                AX_SYS_MemAllocCached(
                    &buffers[i].phyAddr, &buffers[i].pVirAddr,
                    buffers[i].nSize, 128,
                    reinterpret_cast<const AX_S8*>("model_input")),
                "AX_SYS_MemAllocCached(input) failed");
        }
        for (AX_U32 i = 0; i < info->nOutputSize; ++i) {
            AX_ENGINE_IO_BUFFER_T& buf = buffers[info->nInputSize + i];
            std::memset(&buf, 0, sizeof(buf));
            buf.nSize = info->pOutputs[i].nSize;
            check_ax(
                AX_SYS_MemAllocCached(
                    &buf.phyAddr, &buf.pVirAddr, buf.nSize, 128,
                    reinterpret_cast<const AX_S8*>("model_output")),
                "AX_SYS_MemAllocCached(output) failed");
        }
    }

    ~Impl() {
        for (auto& item : buffers) {
            if (item.phyAddr) {
                AX_SYS_MemFree(item.phyAddr, item.pVirAddr);
            }
        }
        if (handle) {
            AX_ENGINE_DestroyHandle(handle);
        }
        AX_ENGINE_Deinit();
        AX_SYS_Deinit();
    }
};

ModelRunner::ModelRunner(const std::string& model_path,
                         const std::string& model_name)
    : impl_(new Impl(model_path, model_name)) {}

ModelRunner::~ModelRunner() {
    delete impl_;
}

std::vector<std::vector<float>> ModelRunner::Run(
    const std::vector<std::vector<float>>& inputs) {
    if (inputs.size() != 6) {
        throw std::runtime_error("rnnoise expects 6 inputs");
    }
    for (int i = 0; i < 6; ++i) {
        const size_t bytes = inputs[i].size() * sizeof(float);
        AX_ENGINE_IO_BUFFER_T& buf = impl_->buffers[impl_->input_idx[i]];
        if (bytes > buf.nSize) {
            throw std::runtime_error("input larger than model tensor");
        }
        std::memcpy(buf.pVirAddr, inputs[i].data(), bytes);
        AX_SYS_MflushCache(buf.phyAddr, buf.pVirAddr,
                           static_cast<AX_U32>(bytes));
    }
    check_ax(axrun::run(impl_->handle, impl_->context, &impl_->io, 0),
             "AX_ENGINE_Run failed");

    std::vector<std::vector<float>> outputs(7);
    for (int i = 0; i < 7; ++i) {
        const AX_ENGINE_IO_BUFFER_T& buf =
            impl_->buffers[impl_->info->nInputSize + impl_->output_idx[i]];
        AX_SYS_MinvalidateCache(buf.phyAddr, buf.pVirAddr, buf.nSize);
        const size_t count = buf.nSize / sizeof(float);
        const auto* src = static_cast<const float*>(buf.pVirAddr);
        outputs[i].assign(src, src + count);
    }
    return outputs;
}
