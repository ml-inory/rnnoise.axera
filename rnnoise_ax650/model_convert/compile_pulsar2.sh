#!/usr/bin/env bash
set -euo pipefail
# 用 Pulsar2 7.0 Docker 编译：model.onnx + calib_data -> output/model.axmodel
# 原始编译镜像：docker-registry.aitsw.axera-tech.com/pulsar2:20260810-temp-0d4427ff
# （如本机已打成 pulsar2:7.0 标签，也可用 pulsar2:7.0）
IMAGE="${PULSAR2_IMAGE:-docker-registry.aitsw.axera-tech.com/pulsar2:20260810-temp-0d4427ff}"
HASP="${MAGNETAR_HASP_SRC:-}"
EXTRA=()
if [ -n "$HASP" ]; then EXTRA+=(-v "$HASP:/root/.hasplm"); fi
docker run --rm -v "$(pwd)":/workspace "${EXTRA[@]}" "$IMAGE" \
  pulsar2 build --config /workspace/pulsar2_config.json
echo "✅ 产物: output/model.axmodel"
