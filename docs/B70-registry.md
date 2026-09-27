# B70 experiment registry

What has been tried for speed and quality on the Arc Pro B70, what it measured, and what is left to try. The point
is to avoid re-running dead ends: **before retrying a rejected item, check its "revisit when" column.** The patch
series is described in [`B70-SYCL.md`](B70-SYCL.md), the run configuration in the README.

Measurement notes that apply throughout:

- **Decode varies between server starts** by up to ~13% on the reference machine: how fast missed experts come off
  the NVMe differs from start to start. Compare configurations with interleaved arms (A-B-A-B), three or more
  starts each.
- **A change that alters rounding also alters routing.** Greedy decode drifts onto other tokens and other experts, so
  per-token counters stop being comparable. For kernel changes, time a teacher-forced replay
  (`qwfn-gen --replay-file`), which reports the decode split over identical tokens.
- **Quality** is natural-text NLL (lower is better) over 1,024 tokens after an 8K prompt, measured both through
  prefill and token by token through the decode path (the reference for the sparse attention).

Last updated: 2026-09-26.

## Adopted

Everything adopted is in the README's **Major improvements** table (what, where, measured effect); each ggml
patch in detail in [`B70-SYCL.md`](B70-SYCL.md).

## Tried and rejected

| Experiment | Result | Revisit when |
|---|---|---|
| `--spec-block` | slower: the duplicate block costs more GPU time than its prediction saves in I/O | graph time drops far enough that prediction dominates |
| `--mtp` draft head | 15-25% slower: 79-92% acceptance, but a draft + verify step costs 2.1-2.4x a plain step | the verify step gets one-token kernels and GPU experts |
| `--batch 32768` | the prefill borrows too much of the expert tier: decode right after drops sharply | — |
| `--batch 8192` / `4096` | prefill -19% / -24%; each halving doubles the passes over the expert set | only if VRAM is needed elsewhere |
| `--batch 24576` | +3-5% prefill with v2, decode unaffected; not adopted yet | re-measure decode after a long prefill with v3 (51% VRAM coverage) |
| Power cap below 140 W | 140 W -6%, 130 W -13-15% | — |
| `--vram` above 25 (overlay v4) | 25 leaves ~0.7 GB free at a full context with an image and nothing evicted; 26 was below that already with v2 (1.8 GB free after init, under the engine's reserve check) | a card with more VRAM |
| Rank-counting argsort (one barrier instead of bitonic stages) | +4% decode graph time | a top-k that reads each value once |
| A larger locked RAM expert tier (`--ram 11` / `13` vs 8, preferred config) | 15% fewer expert misses but decode unchanged within the between-start I/O spread; 40K prefill 3.7% slower at 11, 7% at 13 | once expert-read speed is stable between starts |
| Q8_0 matvec with 2 or 4 rows per sub-group (activation loads shared across rows) | no faster than one row (21.6 / 21.6 / 22.6 us per call): the activation re-reads hit the cache | — |
| Hyper-connection mixers at Q8_0 without patch 21 | quality unchanged, GPU time -1.8 ms/token, but host launch time +1.8 ms/token (activation quantize, unfused SiLU, the scale not foldable exactly) | adopted with patch 21 (overlay v4) |
| Split-K Q8_0 matvec for matrices with few rows | no faster than one sub-group per row | — |
| One-token Q8_0 matvec on f32 activations (no quantize kernel; SiLU/scale epilogue) | launch time -33 us per layer graph, but GPU time +42 us (f32 converts and FMAs cost more than dp4a): +0.4 ms/token on the preferred config | produce q8_1 activations in the producing op instead |
| q8_1 reuse alone (patch 19 without 20) | 108 fewer quantize launches per token but only ~-0.1 ms/token: the quantize kernel is cheap, the generic matmul routine around it is not (hence patch 20) | — |
| Top-k with a row in one sub-group (values in registers, no work-group barriers; patch 13 pays one per selected value) | 2x slower at decode shapes (predictor 8.0 -> 16.9 us, router 6.9 -> 12.2 us); correct | never in this form: patch 13's top-k is 0.70 ms/token in total |
| Grouped oneMKL `gemm_batch` for the prefill MoE | no faster than per-expert GEMMs; prefill unchanged | — |
| Grouped XMX MUL_MAT_ID kernel (joint_matrix, one launch over all experts) | correct, +3% prefill at 40K; the kernel plateaus near 20 TFLOP/s | the kernel gets well past that |
| Device-side routing for the prefill MoE | same +3%: the host round trips were not the bottleneck | — |
| Tile-sparse prefill attention (skip K/V tiles the selection does not use) | not viable: tiles a kernel can run efficiently still touch 44-92% of the dense work | — |
| Async prefill expert uploads on a side queue | correct, no gain: the graph's small host-to-device copies wait behind queued upload pieces | a queue/engine setup where copies really run concurrently |
| Pinned vs pageable prefill staging; keeping the device staging between requests | no effect | — |
| Deeper expert prefetch pool | no effect: the demand wait comes from misses, not queue depth | — |
| IQ4_NL kernels: one block per lane (8-32 lanes/row), rows staged in local memory, table in registers, aligned layout alone | all ~110 GB/s: the stock table lookup is ALU-bound (`dpct::byte_level_permute` is written with 64-bit variable shifts) | superseded by patch 17's local-memory table |
| Q4_K expert gate/up | 2.3x faster kernel, but bigger blocks cut VRAM residency: net decode worse | the expert tier can hold the whole model |
| F16 / Q8_0 experts; MXFP4, Q5_0, Q3_K | slow or too big here | MXFP4: see ideas (its lookup is the same emulated permute) |
| Level Zero knobs (counter-based events, single-thread mode, in-order lists) | no change: layer-graph time is device-side | — |

## Where decode time goes (preferred config, 2026-09-26)

Short-prompt warm decode on the reference machine, ~40 ms/token (~25 tok/s), from the server's `/stats` decode
split (`tools/perf/decab.py`):

| part | ms/token | what it is |
|---|---:|---|
| layer graphs on the GPU | ~26 | dense matvecs (Q8_0 in overlay v3: ~7.4; ~6.4 with patch 18; patches 19-20 then cut their host-side launch cost), bf16 hyper-connection matvecs (~5.5, at bandwidth), GPU experts (IQ4_NL down ~1.5, Q2_0 gate/up ~1.3), attention, DeltaNet, ~5,000 small kernels |
| expert I/O | ~7.6 (7-13 between starts) | waiting for missed experts read from the NVMe (~4 misses/token; 51% of expert blocks fit in VRAM) |
| host | ~6.5 | CPU-computed experts (~7/token), routing readback, promotions |

## Ideas -- not yet tried

Ranked by expected gain for the preferred config per effort. When one is tried, move it to a table above.
Effort: S = a kernel or a switch, M = a few days, L = a week or more.

| Idea | Expected | Effort | Helps the preferred config | First step |
|---|---|---|---|---|
| Expert residency for overlay v3 | the largest decode lever: expert I/O and most of the host time | M-L | yes | frequency-weighted VRAM tier; misses/token at 20-100K first |
| Find the start-to-start expert-read variance | up to ~13% decode on some starts; cleaner comparisons | M | yes | identical reads still vary +-9% between starts; ruled out: filesystem compression, the I/O scheduler, drive temperature, CPU clock, page-cache state. Left: background discard, the drive's own state |
| Prefill upload overlap on a dedicated copy engine | ~8-15% prefill | M | yes | route uploads to a separate copy engine; keep the graph's small copies off that queue |
| Prefill DeltaNet | a few % prefill | M | yes | per-kernel profile at 40K |
| XMX grouped MoE kernel past 20 TFLOP/s | a few seconds per 40K prefill | L | yes | tile and SLM layout profile |
| GSQ-RCO IQ3_S (3.50 bpw) as a base model | quality/speed vs v3 unknown | M | maybe | NLL through the decode path + decode speed vs v3 |
| Deterministic greedy decode on v3 | reproducibility | M | hygiene | find the op whose result varies between runs |
| One graph per token (instead of 48 per-layer graphs) | launch overhead | L | yes | needs the expert residency decided before the token |
| Patch 17's local-memory table for MXFP4 / IQ4_XS | several-x faster kernels for those types; upstreamable | S-M | no (v3 uses neither) | port the lookup, bench against the CPU with a 64-matmul graph |
