#!/bin/bash
# Assemble the dense overlays over the stock GSQ-RCO Q2_0 shards (docs/dense-overlay.md). Nothing of the 66 GB
# model is copied: an overlay is a metadata head with split.count raised, the replacement GGUFs (tensors fetched
# from Unsloth's published quants by HTTP range), then the stock shards as symlinks. The engine keeps the first
# tensor of a name across shards, so the replacements win.
#   scripts/b70/build-overlay.sh GSQ_DIR OUT_DIR [v1|v2|v3|v4]  (default v2; v4 is the preferred config, README)
# GSQ_DIR holds Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-0000{1,2}-of-00002.gguf; needs build/gguf-requant.
# Run the server with QWFN_VOCAB_MODEL=GSQ_DIR/...-00001-of-00002.gguf (the tokenizer comes from the stock head).
set -euo pipefail
HERE=$(cd "$(dirname "$0")/../.." && pwd)
GSQ=$1; OUT=$2; VER=${3:-v2}
S1=$GSQ/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf
S2=$GSQ/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf
FETCH="python3 $HERE/tools/overlay/gguf_fetch.py --jobs 8"
REQUANT=${REQUANT:-$HERE/build/gguf-requant}
mkdir -p "$OUT"
fetch_v2() {
  # v2: the tensors Unsloth never cuts, at their bits. From UD-Q2_K_XL: shared-expert down Q8_0 (all layers),
  # layer 1 ple_key Q8_0, layer 2 gate/up IQ3_XXS. From UD-Q3_K_XL: token_embd Q8_0, output Q6_K, expert down
  # Q8_0 on layers 2, 4, 30, 46, 47.
  [ -f "$OUT/q2kxl.gguf" ] || $FETCH UD-Q2_K_XL "$OUT/q2kxl.gguf" \
    'ffn_down_shexp\.weight$' '^blk\.1\.ple_key\.weight$' '^blk\.2\.ffn_(gate|up)_exps\.weight$'
  [ -f "$OUT/q3kxl.gguf" ] || $FETCH UD-Q3_K_XL "$OUT/q3kxl.gguf" \
    '^token_embd\.weight$' '^output\.weight$' '^blk\.(2|4|30|46|47)\.ffn_down_exps\.weight$'
}
if [ "$VER" = v3 ] || [ "$VER" = v4 ]; then
  # v3: v2 plus, from UD-Q3_K_XL, expert down IQ4_NL on the 43 layers still at Q2_0, and the attention,
  # shared-expert gate/up and ssm_out tensors at Q8_0 (v1/v2 take them at Q5_K/Q6_K, so dense5u.gguf is not
  # used). Expert gate/up stay Q2_0; the hyper-connection mixers stay BF16.
  fetch_v2
  DOWN_LAYERS='0|1|3|5|6|7|8|9|10|11|12|13|14|15|16|17|18|19|20|21|22|23|24|25|26|27|28|29|31|32|33|34|35|36|37|38|39|40|41|42|43|44|45'
  [ -f "$OUT/q3down.gguf" ] || $FETCH UD-Q3_K_XL "$OUT/q3down.gguf" "^blk\.($DOWN_LAYERS)\.ffn_down_exps\.weight\$"
  [ -f "$OUT/dense8.gguf" ] || $FETCH UD-Q3_K_XL "$OUT/dense8.gguf" \
    'attn_qkv\.weight$' 'attn_gate\.weight$' 'attn_q\.weight$' 'attn_k\.weight$' 'attn_v\.weight$' \
    'attn_output\.weight$' 'ffn_gate_shexp\.weight$' 'ffn_up_shexp\.weight$' 'ssm_out\.weight$'
  parts=("$OUT/q3down.gguf" "$OUT/dense8.gguf" "$OUT/q2kxl.gguf" "$OUT/q3kxl.gguf")
  if [ "$VER" = v4 ]; then
    # v4: v3 plus the hyper-connection mixers' up/down at Q8_0 as Unsloth ships them at every tier (their inject
    # weights, F32 there, stay at the stock BF16). Runs best with patch 21 (GGML_SYCL_Q8_EPILOGUE).
    [ -f "$OUT/hcq8.gguf" ] || $FETCH UD-Q3_K_XL "$OUT/hcq8.gguf" \
      '^blk\.[0-9]+\.hc_(attn|ffn)_(up|down)\.weight$' '^output_hc_(up|down)\.weight$'
    parts=("$OUT/hcq8.gguf" "${parts[@]}")
  fi
else
  # v1: Unsloth's dense formats (Q5_K attention and shared-expert gate/up, Q6_K ssm_out) from UD-Q2_K_XL
  [ -f "$OUT/dense5u.gguf" ] || $FETCH UD-Q2_K_XL "$OUT/dense5u.gguf" \
    'attn_qkv\.weight$' 'attn_gate\.weight$' 'attn_q\.weight$' 'attn_k\.weight$' 'attn_v\.weight$' \
    'attn_output\.weight$' 'ffn_gate_shexp\.weight$' 'ffn_up_shexp\.weight$' 'ssm_out\.weight$'
  parts=("$OUT/dense5u.gguf")
  if [ "$VER" = v2 ]; then fetch_v2; parts=("$OUT/q2kxl.gguf" "$OUT/q3kxl.gguf" "$OUT/dense5u.gguf"); fi
fi
parts+=("$S1" "$S2")
N=$(( ${#parts[@]} + 1 ))
D=$OUT/$VER; mkdir -p "$D"
NAME=Qwen3.8-Flash-Next-GSQ-RCO-Q2_0
"$REQUANT" meta "$S1" "$D/$NAME-00001-of-0000$N.gguf" $N
i=2; for p in "${parts[@]}"; do ln -sf "$(readlink -f "$p")" "$D/$NAME-0000$i-of-0000$N.gguf"; i=$((i + 1)); done
ls -la "$D"
echo "head: $D/$NAME-00001-of-0000$N.gguf"
