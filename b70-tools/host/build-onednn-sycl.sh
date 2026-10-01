#!/bin/bash
# oneDNN with SYCL CPU and GPU runtimes, for llama.cpp's GGML_SYCL_DNN (oneDNN flash attention on the B70).
#   b70-tools/host/build-onednn-sycl.sh [PREFIX]      (default /opt/onednn-sycl; run the install step with rights)
# Reconstructs the reference machine's install (oneDNN 3.11, DNNL_CPU_RUNTIME=SYCL, DNNL_GPU_RUNTIME=SYCL); that
# install predates this script, which has not been run as written.
set -e
PREFIX=${1:-/opt/onednn-sycl}
TAG=${ONEDNN_TAG:-v3.11}
SRC=${ONEDNN_SRC:-$HOME/src/oneDNN}
[ -d "$SRC" ] || git clone --depth 1 --branch $TAG https://github.com/uxlfoundation/oneDNN.git "$SRC"
set +u; source ${ONEAPI:-/opt/intel/oneapi/setvars.sh} --force >/dev/null 2>&1; set -u
cmake -S "$SRC" -B "$SRC/build-sycl" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx \
  -DDNNL_CPU_RUNTIME=SYCL -DDNNL_GPU_RUNTIME=SYCL -DDNNL_BUILD_TESTS=OFF -DDNNL_BUILD_EXAMPLES=OFF \
  -DCMAKE_INSTALL_PREFIX="$PREFIX"
cmake --build "$SRC/build-sycl" -j "$(nproc)"
cmake --install "$SRC/build-sycl"
echo "installed: $PREFIX (DNNL_DIR=$PREFIX/lib/cmake/dnnl)"
