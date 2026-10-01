#!/bin/bash
# Natural-text replay NLL: how well a model predicts real text it did not write (lower is better). The quality
# check behind docs/dense-overlay.md (v2: -0.034). Replaying the model's own greedy output instead compresses
# differences between quants.
#   nll_replay.sh IDS_FILE HEAD [HEAD ...]
#     IDS_FILE  one token id per line (b70-tools/fixtures: make_fixtures.py text, then qwfn-tok --raw)
#     HEAD      model head shards to compare (the stock model, an overlay head, ...)
# Env: OFFSETS (default "8000 24000"), PROMPT (64), REPLAY (1024), REPS (2), QWFN (the engine's build dir,
# default ./build), plus the usual backend env (QWFN_GGML_BACKENDS, QWFN_VOCAB_MODEL for overlays, ...).
# Per head, offset and rep: a PROMPT-token prompt, then REPLAY tokens replayed with --ppl. Heads alternate
# within each rep, so drift hits all of them. The run-to-run spread on the B70 is ~0.004.
set -uo pipefail
IDS=$1; shift
OFFSETS=${OFFSETS:-"8000 24000"}; PROMPT=${PROMPT:-64}; REPLAY=${REPLAY:-1024}; REPS=${REPS:-2}
QWFN=${QWFN:-./build}
TMP=$(mktemp -d); trap 'rm -rf $TMP' EXIT
for off in $OFFSETS; do
  sed -n "$((off + 1)),$((off + PROMPT))p" "$IDS" > $TMP/prompt-$off.ids
  sed -n "$((off + PROMPT + 1)),$((off + PROMPT + REPLAY))p" "$IDS" > $TMP/replay-$off.ids
done
for rep in $(seq 1 $REPS); do
  for head in "$@"; do
    for off in $OFFSETS; do
      nll=$("$QWFN/qwfn-gen" "$head" --prompt-file $TMP/prompt-$off.ids --replay-file $TMP/replay-$off.ids \
              --gen $REPLAY --ppl --ctx 32768 --kv q8_0 --vram 24 --ram 8 2>&1 | grep -a -m1 "replay NLL")
      echo "$(basename "$head") offset=$off rep=$rep | ${nll:-FAILED}"
    done
  done
done
