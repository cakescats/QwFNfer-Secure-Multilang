# Dense overlay: Unsloth's bits where GSQ-RCO cut too deep

GSQ-RCO Q2_0 stores most of the model at 2 bits, including tensors that Unsloth's own low tiers never cut
that far. An overlay puts those tensors back at the bits Unsloth uses, without copying the 66 GB model: a
metadata head with `split.count` raised, the replacement GGUFs, then the stock shards as symlinks. The engine
keeps the **first** tensor of a name across shards, so the replacements win.

`scripts/b70/build-overlay.sh GSQ_DIR OUT_DIR [v1|v2|v3|v4]` fetches the replacement tensors from
`unsloth/Qwen3.8-Flash-Next-GGUF` by HTTP range (`tools/overlay/gguf_fetch.py`, only those tensors' bytes)
and assembles the head (`gguf-requant meta`). Run the server on the new head with
`QWFN_VOCAB_MODEL=<stock head shard>`: llama.cpp's split loader, which reads the tokenizer, rejects shards
that carry no `split.no`.

What each overlay contains and what it measured: v2 in the README's validated run configuration, v3 and v4 in its
preferred config. v1, the base of both, is attention and shared-expert gate/up at Q5_K and `ssm_out` at Q6_K from
UD-Q2_K_XL (replay NLL -1.2%, no speed cost). v4 is v3 plus `hcq8.gguf`: the hyper-connection mixers' up/down
(`hc_{attn,ffn}_{up,down}`, `output_hc_{up,down}`) at Q8_0 from UD-Q3_K_XL -- Unsloth keeps them at Q8_0 at every
tier; their 4-row `*_inject` weights are F32 there and stay at the stock BF16 here. It needs patch 21
(`GGML_SYCL_Q8_EPILOGUE`) to be faster than v3: without it the Q8_0 path adds kernel launches that cancel the smaller
reads. The engine folds the mixers' 1/hc scale into BF16/F16/F32 weights, and into Q8_0 block scales only when every
block stays exact -- Unsloth's do not, so v4 keeps the scale, which patch 21 applies in the matvec's epilogue.

`tools/overlay/shard_prune.py OVERLAY_DIR SHARD OUT` rewrites the stock first shard keeping only the tensors that
overlay's head still reads from it: against v3 it drops 14 GB (v1/v2 heads stop working), against v4 15.3 GB (the
stock BF16 mixers too: v1-v3 heads stop working). To keep a v3 rollback, prune against v3 -- v4 runs on a
v3-pruned shard, since its own parts shadow the mixers.

v2's cost is its heavier per-token reads (Q8_0 expert down on five layers, Q8_0 shared-expert down on all,
Q6_K output), not the expert tier: the VRAM tier holds 16437 instead of 17328 blocks, but the share of
expert reads served from VRAM barely moves.

`tools/overlay/quant_compare.py GGML_H HEAD [TIER ...]` prints the effective type of every tensor role of an
assembled model next to Unsloth's published tiers (headers read by range, nothing downloaded): the check that
an overlay took. `gguf_fetch_verify.py` spot-checks a fetched file against its source.

`gguf-requant conv` re-encodes tensors locally (for formats no published quant ships); quantizing
already-quantized weights keeps their error, so it is for speed experiments, and `gguf-requant err`
measures what the second rounding adds.
