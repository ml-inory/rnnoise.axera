cmake_minimum_required(VERSION 3.15)
project(rnnoise_ax620e_sdk LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(include src/rnnoise)

# 原版 rnnoise 信号处理（denoise/pitch/FFT/LPC/表格）。
# 注意：不编译 rnn.c（其 compute_rnn 由 src/rnnoise_ax.cpp 用 AX Engine 替换）。
add_library(rnnoise_c STATIC
  src/rnnoise/denoise.c
  src/rnnoise/pitch.c
  src/rnnoise/kiss_fft.c
  src/rnnoise/celt_lpc.c
  src/rnnoise/nnet.c
  src/rnnoise/nnet_default.c
  src/rnnoise/parse_lpcnet_weights.c
  src/rnnoise/rnnoise_data.c
  src/rnnoise/rnnoise_tables.c
)

add_library(rnnoise_ax620e_sdk STATIC
  src/model_runner.cpp
  src/rnnoise_ax.cpp
)
target_include_directories(rnnoise_ax620e_sdk PUBLIC include)
target_link_libraries(rnnoise_ax620e_sdk PRIVATE rnnoise_c)

add_executable(model_example examples/main.cpp)
target_link_libraries(model_example PRIVATE rnnoise_ax620e_sdk)

if(AX_RUNTIME_ROOT)
  target_include_directories(rnnoise_ax620e_sdk PUBLIC ${AX_RUNTIME_ROOT}/include)
  target_link_directories(rnnoise_ax620e_sdk PUBLIC ${AX_RUNTIME_ROOT}/lib)
  # AX620E/AX630C（AX620E NPU2）官方 BSP（ax620e_bsp_sdk v2.0.0）的
  # libax_engine.so 依赖 libax_interpreter.so（AX_NPU_Get_* 系列符号），须一并链接。
  target_link_libraries(rnnoise_ax620e_sdk PUBLIC ax_engine ax_interpreter ax_sys pthread dl atomic)
endif()
