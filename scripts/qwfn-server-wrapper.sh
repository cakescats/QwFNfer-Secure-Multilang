#!/usr/bin/env bash
# Example engine wrapper for the console (QWFN_SERVER points here, see qwfn-console-engine.conf).
# The console runs the engine as `qwfn-server <model.gguf> <flags...>`; the wrapper picks the
# build, sets the environment and adds flags, then execs the real binary with the same arguments.
# Copy it out of the repository (the drop-in expects ~/qwfn-server-wrapper.sh), set BIN and
# make it executable.
BIN="$HOME/QwFNfer/build/qwfn-server"

# A release bundle keeps libggml next to the binary.
export LD_LIBRARY_PATH="$(dirname "$BIN")${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# An expert pack per model, for a console that does not pass one itself (current consoles set
# QWFN_EXPERT_PACK and --io-uring when the model has a pack). The engine checks the pack's
# fingerprint and ignores a pack that does not match the model.
#case "$(basename "$1")" in
#  Qwen3.8-Flash-Next-Abliterated-Q4_K_M.gguf)
#    export QWFN_EXPERT_PACK="$HOME/qwfn-pack/Qwen3.8-Flash-Next-Abliterated-Q4_K_M.qwpk" ;;
#  Qwen3.8-Flash-Next-UD-Q4_K_XL-0000?-of-00004.gguf)
#    export QWFN_EXPERT_PACK="$HOME/qwfn-pack/Qwen3.8-Flash-Next-UD-Q4_K_XL.qwpk" ;;
#esac

# Early routing; with stock ggml it is slower (needs patches/ggml-cuda/01).
#export QWFN_EARLY_ROUTE=1

EXTRA=()
#EXTRA+=(--io-uring)   # expert reads over io_uring, the pack needs it
exec "$BIN" "$@" "${EXTRA[@]}"
