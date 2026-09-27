# QwFNfer-xpu

A heavily optimized build of [QwFNfer](https://github.com/Apolog1ze-Dev/QwFNfer), the purpose-built
Qwen3.8-Flash-Next inference engine, for the **Intel Arc Pro B70** (32 GB, Battlemage) through ggml's SYCL
backend. The engine is upstream's; this fork adds a ggml-sycl patch series tuned for the card
(`patches/ggml-sycl/`), engine changes for the B70's memory layout and launch costs, a dense-weight overlay, and a
measured run configuration. Everything else (the three-tier expert cache, the sparse attention, the server
and its OpenAI and Anthropic APIs) works as upstream documents it.

## Major improvements

Against the first B70 enablement (2026-09-15: stock llama.cpp SYCL backend, Unsloth UD-Q3_K_XL, no switches):
**15-16 tok/s warm chat and 222-248 tok/s prefill at 21K**. Now, with the validated run configuration below, at a
110 W power cap: **29-31 tok/s short-prompt decode, 27 tok/s decode at 40K context, 430-450 tok/s prefill at
20-40K, ~386 tok/s at 89K**. The [preferred config](#preferred-config) (overlay v4) spends some of that on quality --
natural-text NLL ~1.5 against 2.02 for the validated overlay -- and with patches 18-21 and `--vram 25` runs at
**28-33 tok/s short-prompt decode, ~510 / ~600 / ~500-515 tok/s prefill at 20K / 40K / 89K**.

| Change | Where | Measured effect |
|---|---|---|
| q2_0 experts stored as [codes][scales] so the MoE kernel uses aligned vector loads | patch 09 + engine (`QWFN_Q2_SOA`) | decode layer graphs -18%, 40K decode +20% |
| wide-load bf16/f16 one-token matvec (16-byte loads, f32 accumulate) | patch 05 (`GGML_SYCL_MMVW`) | decode +23% |
| one-token sparse attention fused into one kernel over the q8_0 cache | patch 10 (`GGML_SYCL_FUSE_SPARSE_DECODE`) | decode layer graphs -8.5% |
| small-K f32 matmul without oneMKL's per-call host cost | patch 05 (`GGML_SYCL_SMALLK`) | decode layer graphs -4.9% |
| wide q2_0 MoE matvec + fused gate/up/SwiGLU | patches 07-08 (`GGML_SYCL_MOE_Q2W`) | decode layer graphs -4% |
| hyper-connection combine + norm (and its gate) as one kernel | patches 04, 06, 11 | decode layer graphs -2.3%; with the device-built inputs, 89K prefill -14% |
| small-kernel fusions: router top-k, hyper-connection mixer, ADD chains, MoE weighted sum, DeltaNet conv | patches 12-15 | decode layer graphs -8% (~300 fewer kernels per token), decode ~+5% |
| IQ4_NL experts in an aligned layout, decoded through a local-memory table (overlay v3's expert down) | patch 17 + engine (`QWFN_IQ4_SOA`) | one-token IQ4_NL MoE matvec 2.75x; v3 decode graph -12% |
| one-token Q8_0 matvec with a whole block per lane (16-byte loads of quants and activations) for the dense Q8_0 weights | patch 18 (`GGML_SYCL_Q8W`) | dense Q8_0 matvecs -13% (~405 -> ~467 GB/s), v3 decode graph -1.0 ms/token |
| overlay v4's hyper-connection mixers at Q8_0, their down projection's scale and SiLU as the Q8_0 matvec's epilogue | overlay v4 + patch 21 (`GGML_SYCL_Q8_EPILOGUE`) | v4 vs v3: decode layer graphs -0.62 ms/token, dense weights -0.6 GB, NLL unchanged |
| one-token Q8_0 matvecs without the generic matmul routine's per-call host work, and one q8_1 quantization per shared input (q/k/v, qkv/gate, shared-expert gate/up) | patches 19-20 (`GGML_SYCL_Q8_REUSE`, `GGML_SYCL_Q8_DIRECT`) | 108 fewer quantize launches per token, host launch time -20 us per layer graph, v3 decode graph -0.32 ms/token; bit-identical |
| the prefill sparse-attention indexer's per-head score sum as one kernel | patch 16 (`GGML_SYCL_FUSE_IDX`) | attention -10% at 89K (78 -> 70 s), -1.8 s at 40K; bit-identical |
| OpenMP pool stops spinning next to the launch thread | `KMP_BLOCKTIME=0` | decode +5% |
| next-layer expert prediction from the FFN input | engine (`QWFN_PREDICT_CUR2`) | decode +4.7% |
| oneDNN flash attention for prefill | `GGML_SYCL_FA_ONEDNN` | 89K prefill 90-192 -> ~300 tok/s |
| causal mask and sparse-attention bias built on the device | engine (`QWFN_DEV_MASK`, `QWFN_QSA_PACK`) | 89K prefill 361 -> 389 tok/s |
| locked, driver-registered host memory for the RAM tier and prefill staging | engine (`QWFN_LOCK_HOST`) | 89K prefill 425 -> 482 tok/s |
| fix: the prefill's sparse-attention selection no longer spends its slots on blocks after the query (backend-neutral; offered upstream) | engine | usable cells per query 823 -> 1,998 of 2,052 (40K); 118K prompt, 8 notes to list in order: 0-1/8 placed right in 5 of 5 runs without it, 8/8 in 7 of 7 with it; no speed cost |
| dense overlay v2: Unsloth's bits for the tensors GSQ-RCO cut to 2 bits | `scripts/b70/build-overlay.sh` | natural-text NLL -0.034 (about 10x the run spread) |

Every change is off by default in the patched ggml tree and checked with `test-backend-ops`; the details and the
rejected alternatives are in [`docs/B70-SYCL.md`](docs/B70-SYCL.md) and [`docs/B70-config.md`](docs/B70-config.md), and
everything tried so far (adopted, rejected, and ideas not yet tried) in [`docs/B70-registry.md`](docs/B70-registry.md).

## Preferred config

**Overlay v4 with vision**: the heaviest overlay this card runs at a usable speed. It is the validated
configuration below with more bits where the stock quantization is thinnest: expert down and the dense tensors.
Same build, flags and launcher, a different head. Its IQ4_NL expert down runs through patch 17's kernel
(`QWFN_IQ4_SOA`), which cuts the GPU time per decode token by 12%.

v4 is v3 with the hyper-connection mixers' up/down projections at Q8_0 -- the format Unsloth ships them in at every
tier -- instead of the stock BF16: 0.6 GB less on the GPU, and with patch 21 running each down projection's scale and
SiLU inside the Q8_0 matvec, decode layer graphs 0.62 ms/token faster than v3 (A-B-B-A); natural-text NLL unchanged.
v4's smaller dense weights also leave room for a bigger expert tier: `--vram 25` (53% of the expert blocks in VRAM),
checked at the limit -- an image plus a 126K-token document -- with nothing evicted and ~0.7 GB of VRAM still free
([`docs/B70-config.md`](docs/B70-config.md)). On the hardware below (the v2/v3 columns are older builds; the v3-v4
comparison on one build is the A-B-B-A above):

| | v2 (validated config) | v3 | v4 (preferred) |
|---|---|---|---|
| build, `--vram` | patches 01-17, 24 | patches 01-17, 24 | patches 01-21, 25 |
| decode, short prompt (several starts) | 23-30 tok/s | 23-26 tok/s | 28-33 tok/s |
| prefill at 20K / 40K / 89K | 514 / 522 / 416-431 tok/s (before patches 16-17) | 454 / 482 / 401-408 tok/s (before patches 16-17) | 511 / 604 / 501-516 tok/s |
| expert blocks resident in VRAM | 67% | 51% | 53% |
| 89K needle / 118K prompt, 8 notes recalled | correct / 7 of 8 | correct / 7 of 8 | correct / not re-run |
| image + 126K-token document | -- | -- | correct |

Decode varies with the start as well: how fast missed experts come off the NVMe differs between server starts
(7-13 ms per token on this drive), which moves decode by up to ~13%.

**Weights.** v2, plus from Unsloth's UD-Q3_K_XL:
- expert down at IQ4_NL on the 43 layers v2 leaves at Q2_0 (+20.3 GB);
- the attention, shared-expert gate/up and `ssm_out` tensors at Q8_0 instead of Q5_K/Q6_K (+3.0 GB).

Expert gate/up stay Q2_0; v4 adds the hyper-connection mixers' up/down at Q8_0 (0.7 GB). Vision is the BF16 `mmproj`.
That is 97 GB on disk. Pruning the stock first shard frees what the head shadows: 14 GB for v3 (v1/v2 heads stop
working), 15.3 GB for v4 (the stock BF16 mixers too, so v1-v3 heads stop working). Keep the v3-level pruning while you
may want to roll back to v3.

Quality, as natural-text NLL over 1,024 tokens after an 8K prompt (lower is better):

| | v2 | v3 |
|---|---|---|
| the text through the decode path, token by token (the reference) | 2.02 | **1.51** |
| prompt prefill, then the 1,024 tokens decoded | 2.00-2.03 | **1.46-1.49** |

v4, same prefill-then-decode measure, 4 runs interleaved with v3 on one build: **1.44-1.47** (v3 1.46-1.50 in that
session) -- no loss from the Q8_0 mixers.

The two paths agree, which is what the prefill sparse-attention fix (in the table above) is for. Before that fix,
prefill disagreed with decode and v2's prefill number came out near 1.5.

```sh
scripts/b70/build-overlay.sh <GSQ-RCO dir> <overlay dir> v4     # v3 parts + hcq8.gguf (0.7 GB), head 00001-of-00008
# optional: drop the shadowed tensors from the stock first shard -- against v4 this also drops the stock BF16 mixers,
# so v1-v3 heads stop working; prune against <overlay dir>/v3 instead to keep a v3 rollback (14 GB instead of 15.3 GB)
python3 tools/overlay/shard_prune.py <overlay dir>/v4 <GSQ-RCO dir>/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf pruned.gguf \
  && mv pruned.gguf <GSQ-RCO dir>/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf

QWFN_B70_LLAMA=~/src/llama-b70 QWFN_B70_GSQ=<GSQ-RCO dir> \
QWFN_B70_HEAD=<overlay dir>/v4/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00008.gguf \
  scripts/b70/qwfn-b70.sh --mmproj <GSQ-RCO dir>/mmproj-Qwen3.8-Flash-Next-BF16.gguf
```

`qwfn-b70.sh` sets `--vram 25`, sized for v4. The v2 and v3 heads carry more dense weight: append `--vram 24` (the
server takes the last value of a repeated flag). Upgrading an existing v3 install: `build-overlay.sh ... v4` fetches
only `hcq8.gguf` (the other parts are reused) and writes the new head; point `QWFN_B70_HEAD` at it.

## Validated run configuration

The configuration every number above was measured with (overlay v2), and the base the preferred config builds on.

**Hardware and system.** Arc Pro B70 (32 GB) on PCIe 3.0 x16, Ryzen 7 5700 (8C/16T), 32 GB RAM, NVMe Gen3 x4;
Linux 7.1 with the xe driver, oneAPI 2026.0, oneDNN built for SYCL. GPU power cap 110 W (140-160 W is ~6% faster).

**Weights.**

| Part | Source |
|---|---|
| base model | [`ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF`](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF), Q2_0 (2 shards, 66 GB) |
| dense overlay v2 (+5.7 GB) | attention, shared-expert and `ssm_out` tensors at Q5_K/Q6_K/Q8_0, `token_embd` Q8_0, `output` Q6_K, expert down Q8_0 on layers 2, 4, 30, 46, 47, layer-2 expert gate/up IQ3_XXS: fetched by HTTP range from [Unsloth's](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) UD-Q2_K_XL and UD-Q3_K_XL ([`docs/dense-overlay.md`](docs/dense-overlay.md)) |
| vision (optional) | `mmproj-Qwen3.8-Flash-Next-BF16.gguf` |

**Build.** llama.cpp `bbdd9f2` + `patches/ggml-sycl/01-21`, then the engine against it:

```sh
scripts/b70/build-llama-sycl.sh ~/src/llama-b70
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_CPP_ROOT=$HOME/src/llama-b70 -DLLAMA_CPP_BUILD=$HOME/src/llama-b70/build-sycl/bin
cmake --build build -j
scripts/b70/build-overlay.sh <GSQ-RCO dir> <overlay dir> v2
```

**Run.** `scripts/b70/qwfn-b70.sh` sets everything below and execs the server (extra arguments are appended):

```sh
QWFN_B70_LLAMA=~/src/llama-b70 QWFN_B70_GSQ=<GSQ-RCO dir> \
QWFN_B70_HEAD=<overlay dir>/v2/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00006.gguf \
  scripts/b70/qwfn-b70.sh --mmproj <GSQ-RCO dir>/mmproj-Qwen3.8-Flash-Next-BF16.gguf --vram 24
```

| Setting | Value |
|---|---|
| server flags | `--ctx 131072 --kv q8_0 --vram 25 --ram 8 --batch 16384 --prefill-chunk 6144 --reserve 2048 --prefix-cache 3` (`--vram 24` with the v2/v3 heads) |
| backend | `QWFN_GGML_BACKENDS=<llama>/build-sycl/bin QWFN_REQUIRE_GPU=1 ONEAPI_DEVICE_SELECTOR=level_zero:0 UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1 SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1` |
| tokenizer | `QWFN_VOCAB_MODEL=<GSQ-RCO dir>/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf` (needed with the overlay head) |
| prefill | `GGML_SYCL_FA_ONEDNN=1 GGML_SYCL_ENABLE_MKL_FA=0 QWFN_DEV_MASK=1 QWFN_QSA_PACK=1 QWFN_LOCK_HOST=1 GGML_SYCL_FUSE_IDX=1` |
| fusions | `GGML_SYCL_FUSE_HC=1 GGML_SYCL_FUSE_HC_DECODE=1 GGML_SYCL_FUSE_HC_GATE=1 GGML_SYCL_FUSE_SPARSE_DECODE=1 GGML_SYCL_TOPK_WG=1 GGML_SYCL_FUSE_HC_MIX=1 GGML_SYCL_FUSE_ADDCHAIN=1 GGML_SYCL_FUSE_MOESUM=1 GGML_SYCL_FUSE_CONV=1` |
| decode | `KMP_BLOCKTIME=0 GGML_SYCL_MMVW=1 GGML_SYCL_SMALLK=1 GGML_SYCL_MOE_Q2W=1 QWFN_PREDICT_CUR2=1 QWFN_Q2_SOA=1 QWFN_IQ4_SOA=1 GGML_SYCL_Q8W=1 GGML_SYCL_Q8_REUSE=1 GGML_SYCL_Q8_DIRECT=1 GGML_SYCL_Q8_EPILOGUE=1` |
| host | memlock unlimited (`LimitMEMLOCK=infinity` under systemd) for `QWFN_LOCK_HOST`; a writable `$HOME` so the GPU compiler cache persists (the first request after a new build compiles the kernels) |
| compute runtime | recommended: `NEOReadDebugKeys=1 EnableSharedSystemUsmSupport=0`. By default the runtime may route GPU copies from ordinary host memory through the kernel's shared virtual memory (xe SVM, device-private pages); a process killed in the middle of such a copy can leave the xe driver unable to unbind. Measured at no cost for this engine (interleaved starts, decode and prefill unchanged) |

What not to turn on, and why (`--spec-block`, `--mtp`, `--batch 32768`, `--vram` above 25): see
[`docs/B70-SYCL.md`](docs/B70-SYCL.md).

## Built around the Qwen4 architecture

Qwen3.8-Flash-Next ships the Qwen4-generation design, `qwen4exp` in the GGUF, and the engine is shaped by what that checkpoint actually contains rather than by its parameter count:

- **48 layers, 36 Gated DeltaNet + 12 sparse attention** (every fourth layer), a residual of 4 hyper-connected streams, 512 routed experts with top-10 routing plus one shared expert, a lightning indexer (4 × 128, top-2048) with 4-way pooled keys, and a 51B-parameter per-layer n-gram embedding table (PLE), 28.8 GB on its own.
- **Placement follows the shape.** The dense core (about 5 GB) is resident in VRAM. The routed experts are the only weights that need bandwidth, so they get the three-tier cache. The PLE table stays on the NVMe: a token reads 16 rows of 90 bytes from it (181 µs), so the largest tensor in the file costs no RAM at all.
- **The hybrid layer mix is what makes decode flat.** DeltaNet layers carry a fixed recurrent state and no KV, so their decode graphs reference nothing that changes with position and replay as CUDA graphs; the 12 attention layers select over pooled block keys, so their cost does not grow with context up to the trained 262K.
- **The MoE's routing skew is what makes a small GPU enough.** With 512 fine-grained experts and 10 active, routing is far from uniform, so a VRAM tier of ~2,600–3,900 experts serves 62–77% of lookups at 160K context, and the next layer's routing can be computed a layer early by running its block on the residual stream and prefetched while the current layer runs.
- **The model card is followed** for the thinking template, the tool-call format, mrope for images and the sampling presets; the forward pass is checked node by node against llama.cpp's `qwen4exp`, which matters because this architecture is unusually sensitive to accumulation order.

**What this means for the next Qwen releases.** Every hyper-parameter the engine uses is read from the GGUF metadata (layer count and interval, expert count and top-k, indexer geometry, DeltaNet sizes, PLE geometry, context and rope). Nothing is hard-coded to this checkpoint. A future checkpoint built from the same blocks at a different size (more experts, more layers, a bigger PLE, a longer context) is a metadata change; a new block is a graph change, validated against the reference the same way. What the engine needs from the hardware is set by the *active* path per token and by the tiers you can afford, not by the file size: the dense core and the KV/indexer state must fit in VRAM, and everything else streams through cache tiers sized to the GPU and RAM present. The console's cost model does that sizing for whatever machine it finds. Two things are still on the list: the checkpoint's multi-token-prediction head (the *Draft head* setting: the trunk verifies every draft, so the output is its own) pays in chat and in agentic coding but not yet on a 155K-token document, and the tiers have only been measured on the 16 GB / 30 GB reference machine; the console's auto-tune is what carries the sizing to other machines.

## Acknowledgment

Built on [ggml](https://github.com/ggml-org/ggml) (quantized kernels, CUDA backend) and [llama.cpp](https://github.com/ggml-org/llama.cpp) (tokenizer, and the bit-exact reference the forward pass is validated against). Model: Qwen3.8-Flash-Next by the [Qwen](https://huggingface.co/Qwen) team; quantized GGUFs by [Unsloth](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF), whose Studio served as the harness for testing. The engine itself is [QwFNfer](https://github.com/Apolog1ze-Dev/QwFNfer); this fork only adds the Intel
work on top of it. Base weights: [GSQ-RCO Q2_0](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF)
by ISTA-DASLab. The SYCL backend is ggml's, built with Intel oneAPI and oneDNN.

## License

[Apache License 2.0](LICENSE).
