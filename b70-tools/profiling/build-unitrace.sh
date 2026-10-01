#!/bin/bash
# Intel's unitrace (intel/pti-gpu): per-kernel GPU time on Level Zero, how the decode work was profiled.
#   b70-tools/profiling/build-unitrace.sh [DIR]      (default ~/src/pti-gpu)
# unitrace builds with XPTI; oneAPI ships it, so the oneAPI environment is sourced and its paths are passed.
set -e
DIR=${1:-$HOME/src/pti-gpu}
[ -d "$DIR" ] || git clone --depth 1 https://github.com/intel/pti-gpu.git "$DIR"
set +u; source ${ONEAPI:-/opt/intel/oneapi/setvars.sh} --force >/dev/null 2>&1; set -u
cd "$DIR/tools/unitrace" && rm -rf build && mkdir build && cd build
C=/opt/intel/oneapi/compiler/latest
cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_WITH_MPI=0 -DXptifw_LIBRARY=$C/lib/libxptifw.so -DXptifw_INCLUDE_DIR=$C/include ..
nice make -j"$(nproc)"
echo "built: $DIR/tools/unitrace/build/unitrace"
