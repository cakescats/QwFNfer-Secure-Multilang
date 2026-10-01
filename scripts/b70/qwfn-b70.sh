#!/bin/bash
# qwfn-server with the B70 configuration (docs/B70-config.md). Extra arguments are appended, so a supervisor
# can add --host/--port (the server takes the last value of a repeated flag) or --mmproj.
#   QWFN_B70_LLAMA=~/src/llama-b70 QWFN_B70_GSQ=~/models/.../gsq-rco QWFN_B70_HEAD=<overlay head> scripts/b70/qwfn-b70.sh
# exec, deliberately: a supervisor that treats this process exiting as "VRAM released" must see qwfn-server
# itself, not a wrapper that could outlive it. QWFN_LOCK_HOST needs the memlock limit raised
# (LimitMEMLOCK=infinity under systemd); check the log for "locked (VmLck".
set -e
HERE=$(cd "$(dirname "$0")/../.." && pwd)
LLAMA=${QWFN_B70_LLAMA:-$HOME/src/llama-b70}
GSQ=${QWFN_B70_GSQ:?set QWFN_B70_GSQ to the directory with the GSQ-RCO Q2_0 shards}
STOCK=$GSQ/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf
HEAD=${QWFN_B70_HEAD:-$STOCK}                 # an overlay head (scripts/b70/build-overlay.sh), or the stock model
set +u; source ${ONEAPI:-/opt/intel/oneapi/setvars.sh} --force >/dev/null 2>&1; set -u
export QWFN_GGML_BACKENDS=$LLAMA/build-sycl/bin QWFN_REQUIRE_GPU=1 QWFN_VOCAB_MODEL=$STOCK
export ONEAPI_DEVICE_SELECTOR=level_zero:0 UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1 SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
export GGML_SYCL_FA_ONEDNN=1 GGML_SYCL_ENABLE_MKL_FA=0          # oneDNN flash attention
export QWFN_DEV_MASK=1 QWFN_QSA_PACK=1 QWFN_LOCK_HOST=1          # prefill inputs on the device; locked host memory
export GGML_SYCL_FUSE_IDX=1                                      # prefill: the sparse-attention indexer's head sum as one kernel (patch 16)
export GGML_SYCL_FUSE_HC=1 GGML_SYCL_FUSE_HC_DECODE=1            # hyper-connection combine+norm fusion
export KMP_BLOCKTIME=0 GGML_SYCL_MMVW=1 GGML_SYCL_SMALLK=1 GGML_SYCL_MOE_Q2W=1 QWFN_PREDICT_CUR2=1   # decode
export QWFN_Q2_SOA=1 GGML_SYCL_FUSE_SPARSE_DECODE=1 GGML_SYCL_FUSE_HC_GATE=1                         # decode (patches 09-11)
export QWFN_IQ4_SOA=1                                                                                # iq4_nl experts in VRAM as IQ4_NL_SOA (patch 17)
export GGML_SYCL_Q8W=1                                                                               # one-token Q8_0 matvec, a block per lane (patch 18)
export GGML_SYCL_Q8_REUSE=1 GGML_SYCL_Q8_DIRECT=1                                                    # one-token Q8_0: q8_1 input reused, direct dispatch (patches 19-20)
export GGML_SYCL_Q8_EPILOGUE=1                                                                       # Q8_0 MUL_MAT -> [SCALE ->] SILU as one launch (patch 21; overlay v4's mixers)
export GGML_SYCL_TOPK_WG=1 GGML_SYCL_FUSE_HC_MIX=1 GGML_SYCL_FUSE_ADDCHAIN=1 GGML_SYCL_FUSE_MOESUM=1 GGML_SYCL_FUSE_CONV=1   # decode fusions (patches 12-15)
exec "$HERE/build/qwfn-server" "$HEAD" \
    --ctx 131072 --kv q8_0 --vram 25 --ram 8 --batch 16384 --prefill-chunk 6144 --reserve 2048 --prefix-cache 3 \
    "$@"
