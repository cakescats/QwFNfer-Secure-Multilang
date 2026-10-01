# Strict prefix caching

**Status: implemented (P1-P3) on branch `prefix-cache`, off `cpu-opt` (c9a2527).** The design below is
kept as written, with corrections marked; "As built" at the end records where the implementation
departs from it and what was measured.

## The problem

The engine keeps one sequence. Today a request either strictly extends what the engine already
consumed -- in which case only the tail is prefilled -- or it resets and re-prefills from scratch
(`qwfn_server.cpp`, the `extend` test around line 1003). Two clients alternating, say a coding agent
and a chat assistant, therefore pay a full prefill on every switch. For a 21K-token context that is
**~86 s per switch**, which makes alternation unusable however fast decode is.

## Goal and non-goals

**Goal.** Keep a pool of whole-conversation checkpoints. When a request arrives that is a strict
prefix-extension of some checkpoint, restore that checkpoint and prefill only the tail. Pool size is a
startup parameter; when it is full, evict to make room.

**Non-goals for this work**, all deliberately:

- **Concurrency.** Still one request at a time under the existing mutex. Unified KV with parallel
  requests is known to corrupt QSA block pooling upstream, and `qwfn_state.h` says so in its header.
  Checkpointing buys fast *switching*, not parallel decoding, and that is the whole ask here.
- **Block-level sharing** (vLLM-style paged radix trees). We store and restore whole checkpoints; two
  conversations sharing a system prompt do not share its pages. Simpler, and enough for 2-3 agents.
- **Partial rewind.** A checkpoint is restored whole. Truncating a checkpoint to a shorter prefix is
  possible for KV but not for the recurrent scan, so it is out.

## Why this is cheap here, and why the hard part already exists

Two architectural facts make this much less painful than it would be for a fully recurrent model.

**1. Three quarters of the layers carry constant-size state.** Of 48 layers, only 12 are sparse
attention with a growing KV cache; the other 36 are gated DeltaNet with a `[128,128,48]` recurrent
state that does not grow with context (`qwfn_state.cpp:51`). So the "you cannot rewind a scan" problem
costs a **fixed 113 MB**, not something proportional to the prefix.

**2. The engine already rewinds recurrent state.** MTP verification does exactly this: `rollback_n()`
(`qwfn_engine.cpp:3256`) copies per-layer DeltaNet state and conv history back from snapshot slots,
decrements `n_past_`, and relies on the KV/indexer/pooled rows of undone positions simply being
overwritten by what comes next. The snapshots are written by the graph itself
(`gb.set_rollback(...)`, `qwfn_engine.cpp:1864`; on SYCL they are fused into the DeltaNet kernel), so
capture is nearly free. What is missing is scope, not mechanism: `rb_nsnap_` allocates one slot per
draft token and the snapshots describe "now minus one or two tokens", never "conversation B at token
4000".

## What a checkpoint must contain

Everything below is per sequence. Getting this list wrong is the main correctness risk, because a
missing field corrupts silently rather than failing.

| State | Tensor | Shape | Scales with prefix? |
|---|---|---|---|
| KV cache | `k_[il]`, `v_[il]` (12 attention layers) | flat `512 x n_ctx`, `--kv` type | yes, contiguous leading slice |
| Indexer keys | `idx_[il]` (12 layers) | flat `128 x n_ctx`, F16 | yes, contiguous |
| DeltaNet state | `rs_[il]` (36 layers) | `[128,128,48]` F32 | **no - 113 MB flat** |
| DeltaNet conv history | `conv_[il]` (36 layers) | `[3, 10240]` F32 | no - 4.4 MB |
| PLE conv history | `ple_conv_` | `[hist, 10240]` F32 | no - ~0.4 MB |
| Counters | `n_past_` (the MTP counters are out of scope: caching is refused under `--mtp`) | - | no |
| Server-side | `S.consumed` (tokens), `S.consumed_img` (per-position image hashes) | - | yes |

Derived structures are **rebuilt, not stored**: `qd_.bias` (reset to -inf and recomputed per token),
`qd_.blk_cells`, `qd_.cell_pos`, the decode bias window, and - corrected by the review - the **QSA pooled
block keys** `pool_cache_`, which `qsa_pool_rebuild()` regenerates from the raw KV whenever `pool_dirty_`
is set (it already is after every prefill, `qwfn_engine.cpp:1568`). Dropping them saves ~25 MB per entry
and removes a class of correctness risk. The restore path sets `pool_dirty_` and calls
`qsa_decode_prepare()`, exactly as a fresh prefill does.

**The image overrides are not checkpoint state (corrected after implementation).** The review first
listed `ov_`/`ov_pos_` as state to save. They are not: an override is read only while its position is
being evaluated (`p >= n_past && p < n_past + T`), so below `n_past` its effect already lives in the
caches, and nothing ever evaluates those positions again (MTP rollback aside, and MTP is refused). The
real hazard is the reverse: `set_embeddings()` appends and never prunes, so **another** conversation's
override at a position past the restored `n_past` would be spliced into the new tail. A restore
therefore clears them all. The test's negative control for this (leave the other prompt's override in
place) turns the replay NLL into NaN.

K, V and the indexer are flat 1-D tensors of `dim * n_ctx`, so a prefix of *n* tokens is a contiguous
leading range -- a checkpoint costs prefix length, not whole context.

## Size, and the startup parameter

Per token, at `--kv q8_0`, r = 4:

- KV: `512/32*34 = 544 B` each for K and V, x 12 layers = **13.1 KB/token**
- Indexer: `128 * 2 B` x 12 = **3.1 KB/token**

**~16.2 KB per token**, plus a flat **~118 MB** of recurrent and conv state per checkpoint. Measured:
4,095 tokens = 184.1 MB, 13,471 tokens = 335.4 MB, both matching the formula.

| Prefix | Checkpoint |
|---|---|
| 4K | ~184 MB |
| 21K (our document) | ~458 MB |
| 32K | ~636 MB |
| 128K | ~2.2 GB |

`--kv q4_0` nearly halves the KV component. Proposed flag:

```
--prefix-cache GB    host memory budget for the checkpoint pool (default 0 = off)
```

4 GB holds roughly eight 21K conversations, which covers the coding-agent-plus-chat-assistant case with room
to spare. **The budget must be declared against the same RAM ledger as `--ram`**: the expert arena is
already pinned 18 GB of 31 GB, so a large pool and a large arena cannot both be had. The flag should
refuse to start if `--ram + --prefix-cache` leaves less than a few GB of headroom.

Checkpoints live in **pinned host memory** (the same buffer type `--state-host` already uses), so the
expert tier keeps its VRAM and copies run at PCIe speed.

## Where it hooks in

The whole integration point is the `extend` test in the request path:

```
extend = P.tok starts with S.consumed  (and the image hashes at spliced positions match)
if (!extend) { eng.reset(); eng.clear_embeddings(); S.consumed.clear(); S.consumed_img.clear(); }
```

It becomes:

1. **Extend?** unchanged - the fast path stays exactly as it is.
2. **Otherwise, save** the current sequence as a checkpoint keyed by `S.consumed` (if the pool is on
   and `n_past_ > 0`), evicting as needed.
3. **Look up** the longest checkpoint whose token vector is a strict prefix of `P.tok`, with image
   hashes matching at every spliced position below its length.
4. **Hit:** restore it, set `S.consumed`/`S.consumed_img` to that checkpoint's, then prefill the tail
   as the extend path already does. **Miss:** reset as today.

Nothing about the sampling path, the streaming path or the tool-call handling changes.

## API sketch

```cpp
// qwfn_state.h -- byte-exact copies of the per-sequence tensors, prefix-limited.
size_t state::checkpoint_bytes(int32_t n_tokens) const;
bool   state::save(int32_t n_tokens, state_blob & out, std::string & err) const;
bool   state::restore(const state_blob & in, std::string & err);

// qwfn_engine.h -- the state above plus the engine's own counters and pooled keys.
bool engine::checkpoint_save(checkpoint & out, std::string & err);   // at n_past_
bool engine::checkpoint_restore(const checkpoint & in, std::string & err);

// qwfn_server.cpp -- the pool.
struct prefix_pool {
    struct entry { std::vector<int32_t> tok; std::map<int32_t, uint64_t> img; checkpoint ck; uint64_t used_at; };
    const entry * longest_prefix_of(const std::vector<int32_t> & tok, const std::map<int32_t, uint64_t> & img) const;
    void insert(entry e);      // evicts least-recently-used until it fits
    size_t bytes = 0, budget = 0;
};
```

Copies use the same idiom as `rollback_n()`: build views over the source range and
`ggml_backend_tensor_copy` them, which works device-to-host and back without touching the graph.

## Eviction

Budget in bytes; when an insert would exceed it, drop entries until it fits. You asked for oldest-out;
I propose **least-recently-used** rather than strict FIFO, since with two alternating agents LRU and
FIFO agree, while LRU behaves better if a third, rarely-used conversation appears. One entry per
conversation: when a conversation grows, its new checkpoint replaces the old one rather than
accumulating. Both choices are one line to change if you prefer otherwise.

## Correctness rules

- **Strict prefix only.** Exact token equality, no fuzzy matching. The current design's conservatism
  is what makes it correct, and a "close enough" match would silently corrupt the scan.
- **Images.** Reuse the existing per-position hash check; a prefix whose image at position p differs is
  not a match, even if the token ids (the pads) agree.
- **MTP.** `--mtp` adds `mtp_h_rows_` and the `t_hlast_` rows to the live state. v1 should either
  checkpoint them too or refuse to cache while MTP is enabled; refusing is safer and MTP is currently
  off by default (it was measured neutral).
- **Context growth.** A checkpoint is valid only for the `n_ctx`, `--kv` type and model it was taken
  with; tag entries with those and drop mismatches.
- **After restore**, mark `pool_dirty_` **true** (the pooled keys are rebuilt, not restored), clear
  `rb_valid_`/`rb_depth_` as `reset()` does, refill `qd_.bias` with -inf, then continue as the extend path.
- **Never checkpoint after a failed eval.** `n_past_ += T` runs only after the graph, so a failure partway
  through a layer loop leaves some DeltaNet scans advanced with `n_past_` unchanged. Today no `generate()`
  caller resets on that path (`:1590`, `:1734`, `:1846`), so this must be fixed first - see review issue #1.
- **Tag entries** with `n_ctx`, the `--kv` type, the model and the `--state-host` placement; refuse
  mismatches. `--state-host kv` moves the KV cache to host memory and changes the copy direction.
- **Refuse to cache while `--mtp` is on** (confirmed): the draft head adds `st_mtp_`, `t_hlast_`,
  `mtp_h_rows_` and `mtp_kv_valid_` to the live state. Assert it rather than assume it.

## Phases

1. **P1 - primitives.** `state::save/restore` + `engine::checkpoint_*`, no pool, no flag. Test: prefill
   a document, save, `reset()`, restore, then continue; the next 64 greedy tokens must match a run that
   never switched, and replay NLL must be unchanged.
2. **P2 - pool.** The `prefix_pool`, the `--prefix-cache` flag, the hook in the request path, and
   `/stats` counters (hits, misses, bytes, evictions, save and restore milliseconds).
3. **P3 - measurement.** Alternate two 21K documents for several turns and report time-to-first-token
   with the pool off and on. Expected: ~86 s -> ~0.1 s on a hit, since a 475 MB restore over pinned
   PCIe is tens of milliseconds against a full re-prefill.
4. **P4 - docs.** Parameters section, registry entry, and a note in the recipe document.

## Risks and open questions

- **A missed field corrupts silently.** This is the main risk and the reason P1's gate is token-exact
  continuation, not "it looks fine". The inventory above was read out of the code, but `engine::reset()`
  is the authority on what constitutes live state and should be re-read when implementing.
- **Host RAM contention** with the pinned expert arena, as above. The flag must be validated at startup.
- **Save cost on every switch.** A 475 MB device-to-host copy per switch (~50 ms) is paid even when the
  next request turns out to be a miss. Acceptable against 86 s, but worth measuring.
- **Does the QSA pooled cache need a rebuild anyway?** `pool_dirty_` exists because a prefill rebuilds
  pooled keys from the raw cache. If restoring them proves unreliable, the fallback is to restore
  KV/indexer only and let the rebuild run -- slower, but simpler.
- **Long contexts.** At 128K a checkpoint is 2.3 GB, so the pool holds one or two. Fine, but the flag's
  advice should say so.

## As built

**Code.** `state::checkpoint_bytes/save/restore` (`qwfn_state.*`) walk the same (tensor, leading bytes)
list in the same order: for each attention layer the first `n` rows of K, V and the indexer, for each
DeltaNet layer the whole state and conv history, then the PLE conv. `engine::checkpoint_save/restore`
(`qwfn_engine.*`) add the position, tag the state config, refuse under `--mtp`, synchronize the backend,
and on restore call `reset()` (bias, rollback flags, `pool_dirty_`) before writing, then clear the
overrides. The server (`tools/qwfn_server.cpp`) has the pool, the `--prefix-cache GB` flag and a
`prefix_cache` block in `/stats`.

**Departures from the design.**

- **Overrides are not saved** - see the correction above.
- **Checkpoints are ordinary host memory, not pinned.** Copies use `ggml_backend_tensor_get/set` into a
  `std::vector`. Measured save 80-176 ms and restore 32-71 ms for 184-337 MB, which is already negligible
  against a re-prefill, so pinning was not worth taking memory from the pinned expert arena.
- **Each entry carries its conversation's last reply** (`server::reply_state`: prompt tokens, generated
  tokens, messages, content, reasoning, tool-call key). This was missing from the design and without it
  the pool would never hit on chat traffic: the server rebuilds a returning conversation's prompt from
  the exact tokens it generated only for the *last* reply it produced, and after A -> B -> A that is B's.
  A's re-tokenized reply would differ from what the engine consumed (BPE merges across the framing seam)
  and the prefix would stop matching right there. `build_prompt` now tries the live reply, then every
  cached one, newest first; a restore makes the entry's reply the live one.
- **Images in the match.** A prompt built by continuing an entry's last request carries no splices for
  the images inside that request, so it records which entry it came from (`prompt::via`) and only that
  entry may vouch for them. Otherwise every cached image position must be matched by a splice with the
  same hash.
- **A prompt identical to the consumed tokens re-prefills.** Previously it counted as an extension with
  an empty tail and no logits to sample from; the same rule applies to pool lookups (an entry must be
  strictly shorter than the prompt).
- **Startup check:** `--ram + --prefix-cache + 4 GB` must fit in physical RAM, and `--mtp` is refused.

**Eviction and supersession.** The lookup runs first; then the live sequence is saved and the hit is
restored. Least recently used out when an insert would exceed the budget. When the live sequence is saved,
any entry whose tokens are a prefix of it is dropped (the conversation grew); an identical entry is only
touched. Neither eviction nor supersession ever removes the entry the request is about to restore (the
first version saved before looking up, and could evict its own hit - see
`adversarial-review-prefix-cache-impl.md`). An entry larger than the whole budget is not saved. Entry
size counts the checkpoint, token vectors, image hashes and the stored reply including its message list.

**Minimum length to save** - see the next section (decided: add it).

## Spec: minimum length to save (`--prefix-cache-min`)

**Status: implemented and verified.** Spec review `adversarial-review-prefix-cache-min-spec.md`, final
review `adversarial-review-prefix-cache-min-impl.md`; results at the end of this section.

**Problem.** Every switch saves the departing sequence however short it is. The recurrent state is a
constant ~118 MB, so a one-off request - a title, a summary, a raw completion - costs an entry that size
and pushes real conversations out under LRU. Measured: a 5-token `/v1/completions` left a 118.4 MB entry
for 21 tokens and filled a 0.75 GB pool to 0.74 GB.

**Rule.** `pool_save` does nothing when the live sequence holds fewer than `min` tokens (`0 < n_past < min`).
The check sits right after the existing `n_past` checks - an empty engine returns earlier and is not
counted as a skip - and before the entry size is computed (that serializes the message list), so a skipped
save costs nothing.

```
--prefix-cache-min N   tokens: shorter sequences are not checkpointed on a switch (default 1024; 0 saves all)
```

**Default 1024.** At 1,024 tokens an entry is ~135 MB (118 MB flat + 16.6 MB), and the checkpoint buys at
most the re-prefill of those tokens. The re-prefill cost of a ~1K-token prompt on this box has not been
measured directly (document prefill runs 222-270 tok/s, i.e. ~4-5 s for 1K; short prompts take the
cache-batched path and may be faster) - measure it in the verification run and revisit the default if
it is far off.

**Trade-off.** A conversation that alternates while it is still short is never cached: a client whose
system prompt and short exchanges sit around 800 tokens re-prefills on every switch until its history
passes `min`. For such clients lower `min` (or set 0); the right value is where a re-prefill of `min`
tokens stops being noticeable.

**What it does not change.**

- **Lookup and restore:** unchanged. Every entry is at least `min` tokens long, since nothing shorter is
  ever inserted.
- **The live sequence itself:** a short sequence that is not saved is simply dropped on the switch, as
  with the pool off.
- **Supersession and touch:** they run only as part of a save, so a short live sequence neither
  supersedes nor touches anything. No entry can be a prefix of it or equal to it anyway, since entries are
  at least `min` long and the live sequence is shorter.
- **`min` is constant for the process.** The claims above rely on it; a future runtime setter (e.g.
  `POST /props`) would have to drop entries below a raised `min`.
- **Compacted conversations:** not addressed. An agent that rewrites its history leaves a long entry that
  never matches again; LRU remains the only remedy. A length threshold cannot tell those apart.

**Validation at startup.**

- Parsed strictly: non-numeric or negative -> refuse (`atoi` would turn garbage into 0, "save everything").
- Checked after all arguments are parsed, since `--ctx` may come later on the command line.
- The `min >= --ctx` refusal applies only when the flag was given; the default is lowered to `--ctx / 2`
  with a note instead, so a small `--ctx` never fails over a flag nobody passed (final review #1).
- `min >= --ctx` with the pool on -> refuse: nothing could ever be saved (see the default's exception below).
- `--prefix-cache-min` without `--prefix-cache` -> accepted with a warning: the flag has no effect.

**Observability.** `/stats` `prefix_cache` gains `skipped_short` (switches whose departing sequence was
below `min`) and `min_tokens`. The skip is logged, one line per switch, like saves.

**Verification.** On the B70, pool on at the default: a long conversation A (13.4K tokens), a short one S
(~700 tokens, below `min`) and a raw completion X, in the order `A S A X A S`.

| Step | Expected |
|---|---|
| 1 A | miss |
| 2 S | miss; A saved |
| 3 A | hit; S skipped (short) |
| 4 X | miss; A saved (supersedes the older A) |
| 5 A | hit; X skipped |
| 6 S | miss (never saved); A saved; S's prompt time is the ~700-token re-prefill cost the default trades against |

Expected counters: 2 hits, 4 misses, 3 saves, 2 `skipped_short`, and a pool of one entry.

## Verification

**P1 gate** - `qwfn-gen --ckpt-test OTHER` (`p10-window.sh ckpt`). GPU decode on the B70 is not
bit-reproducible (experts move between tiers), so the gate compares against a measured noise floor
rather than token identity: *base* prefills a 4,096-token prompt, saves, and decodes 64 greedy tokens;
*fresh* re-prefills and replays base's tokens; *ckpt* prefills and decodes a different 4K prompt, restores,
and replays base's tokens. Pass needs a byte-exact round trip (save after the restore equals the saved
blob), a mean per-step logit difference within 1.5x fresh's, and replay NLL no further from base than
1.5x fresh's.

| Run | Round trip | Mean per-step max\|dlogit\| (fresh / ckpt) | NLL (base / fresh / ckpt) | Result |
|---|---|---|---|---|
| device state | exact | 1.328 / 0.980 | 0.5221 / 0.5370 / 0.5279 | PASS |
| `--state-host kv,idx` | exact | 1.176 / 0.933 | 0.4900 / 0.4957 / 0.4938 | PASS |
| control: zero 5% of the state bytes | exact | 1.134 / 2.121 | 0.4814 / 0.4927 / 0.5355 | FAIL (as required) |
| control: other prompt's image override left in place | exact | - | 0.4239 / 0.4382 / NaN | FAIL (as required) |

The noise floor is wide - two fresh prefills of the same prompt differ by ~1.2 logits per step - so the
gate cannot see a small loss. The byte-exact round trip covers the copy path, and the zeroed-state
control shows a corrupted checkpoint roughly doubles the difference. A 64-token variant was dropped: its
floor is ~2.6 logits per step and its extra control could not detect anything, which is how the
override correction above was found.

**P3 - alternating conversations** (`p10-window.sh pcache`, `scripts/box/pcache_drive.py`): two chat
conversations over different source files (13.4K and 11.3K tokens), three turns each, alternating,
each turn replaying its whole history plus a new question; recipe config `--vram 26 --ram 18 --batch
16384 --kv q8_0`, `--prefix-cache 3`.

| Turn | Pool off: prefilled, prompt time | Pool on: cached / prefilled, prompt time |
|---|---|---|
| 1 A | 13,397 in 51.5 s | 0 / 13,397 in 52.0 s |
| 1 B | 11,334 in 38.3 s | 0 / 11,334 in 38.4 s |
| 2 A | 13,493 in 43.4 s | 13,471 / 29 in 1.80 s |
| 2 B | 11,433 in 36.8 s | 11,403 / 29 in 1.47 s |
| 3 A | 13,573 in 43.7 s | 13,539 / 28 in 1.51 s |
| 3 B | 11,497 in 36.4 s | 11,467 / 28 in 1.53 s |

**Every switch after the first pair hit: 36-44 s of prompt time became 1.5-1.8 s** (wall per turn
38-48 s -> 3.7-4.5 s, including generation). Restores took 63-71 ms and saves 76-89 ms; the pool held two
entries, 0.64 GB. Of the remaining ~1.5 s the restore and save are ~0.15 s; the rest (the 28-token
tail's eval and the pooled-key rebuild that follows any prefill) was not broken down. Answers stayed on their own document in every turn. The first on
run (before the override change) gave the same picture: 1.62-2.13 s.

**Not yet exercised:** images through the pool on the server (the matching rules above are unit-level
reasoning, not a measured run), eviction under pressure, and more than two conversations.

### Results

**Window `pcache min1`** (default `min` 1024, 3 GB pool, `A S A X A S`, S ~700 tokens): **exactly the
expected counters** - 2 hits, 4 misses, 3 saves, 2 `skipped_short`, 1 entry. The log shows the two skips
("not saved: 685 tokens", "not saved: 21 tokens"); A restored in 71-74 ms with 28-29 tokens prefilled
(1.43-1.81 s), and no 118 MB entry for the completion.

**Window `pcache min2`** (final binary with the review fix, S ~400 tokens, `A S A S`): again as expected -
1 hit, 3 misses, 2 saves, 1 skip ("not saved: 418 tokens"). Timings in this run are contaminated: vLLM
had just reloaded in the reattach before the window (the first prefill took 209 s against 52 s clean, a
29-token tail 30 s against 1.8 s).

**The default's premise did not hold.** Re-prefilling a short conversation costs far more than the 4-5 s
estimate:

| Conversation | Prompt | Path | Re-prefill |
|---|---|---|---|
| S, ~700 tokens (`min1`, clean) | 624 / 715 | streamed (> 512 new tokens): a whole-expert-set sweep | **14.2 / 14.7 s** |
| S, ~400 tokens (`min2`, contaminated) | 350 / 448 | cache-batched (<= 512) | 21.8 / **8.4 s** |
| X, raw completion (`min1`) | 5 | cache-batched | 3.0 s wall incl. 16 generated |

So at `min` 1024 a conversation of 513-1023 tokens pays ~14 s on every switch to save ~125-135 MB, and
one of ~400 tokens probably ~8 s. The one-off requests the rule targets are far shorter (titles,
summaries: tens of tokens, ~3 s). **Recommendation: lower the default to 256** - it still keeps
completions and title-style requests out of the pool, and caches anything whose re-prefill is expensive.
Left at 1024 pending a decision; `--prefix-cache-min 256` applies it per server.

## Spec: prompt-boundary checkpoints (`--prefix-cache-boundary`)

**Status: implemented and verified.** Spec review `adversarial-review-boundary-spec.md`, implementation review
`adversarial-review-boundary-impl.md`; results at the end of this section.
The review found the checkpoint alone cannot work (issue #1), so this spec has three parts: a divergence log,
a token-stable rebuild, and the boundary checkpoint.

### Problem, as measured

A chat-assistant session (2026-09-16): the engine held the previous turn, 59,230 tokens; the
next request was 59,259 tokens but did not start with them, so it re-prefilled everything - **457 s**,
against 32-73 s for the turns that extended. The recurrent layers cannot be rewound to the first
differing token, so a mismatch anywhere costs the whole conversation. The likely cause (unconfirmed) is
the previous assistant reply: the server replays its exact generated tokens only when the client's text
matches its record; otherwise it re-tokenizes the reply, and BPE rarely reproduces generated tokens.

### Part 1 - divergence log (review #2)

When a chat or completion request does not extend the live sequence, log one line: the prompt length, the
consumed length, the first differing token index, and ~60 decoded characters on each side of it. The cause of a
miss in real traffic then shows up in the child log instead of being guessed.

### Part 2 - token-stable rebuild (review #1)

**The precondition the checkpoint needs.** A full rebuild renders every earlier assistant reply from text, but
the engine consumed the tokens it *generated* for those replies whenever the turn continued through the fast path.
A rebuilt 170-message prompt therefore diverges at the first assistant turn, and no checkpoint of a later
point can match it.

**Rule.** In the full rebuild, before rendering an assistant message from text, look for a reference sequence
that matches the prompt built so far - the live `S.consumed` first, then pool entries - and continues with
`<|im_start|>assistant\n`. Take the span up to and including the next `<|im_end|>` and its `\n`, decode it, and
split it the way a generated reply is split: reasoning between `<think>` and `</think>`, content after it,
tool calls parsed from the content with the request's tool schema. Substitute the span's tokens when the
message is equivalent by the rules `known_reply` already applies: content equal up to trailing whitespace,
equal tool-call keys, reasoning equal (whitespace-trimmed) when the client sends it. Otherwise render from text
as today. Messages with images are never substituted.

### Part 3 - boundary checkpoint

### Idea

A chat prompt is *history* followed by the assistant opener (`<|im_start|>assistant\n` plus the think
prefix, plus a forced tool-call opening). During prefill, **stop at the end of the history, save a checkpoint
into the prefix pool, then feed the opener.** A later request whose history matches up to that point - and
differs only in how the reply and the new turn are rendered - restores it and prefills only those tokens.

**Why before the opener, not at the end of the prompt:** the opener is rendered differently when it is the
open end of a prompt (`open_assistant`) and when it is a past turn in the history
(`<|im_start|>assistant\n<think>\n` + reasoning + `\n</think>\n\n`); for a non-thinking turn
`<think>\n\n</think>\n\n` and `<think>\n` + `\n</think>\n\n` are different token sequences. A checkpoint that
includes the opener would not be a prefix of the next request; one taken before it is.

### Behaviour

- **Where the boundary comes from.** `server::prompt` gains `boundary` (token index, -1 = none), set by
  `build_prompt` to `P.tok.size()` immediately before `open_assistant()`, on both the continuation and the full
  rebuild path. `/v1/completions` prompts have none.
- **When a boundary checkpoint is taken.** In `generate()`, with the pool on, if
  `boundary >= max(--prefix-cache-boundary, --prefix-cache-min)`, the boundary lies inside the tail being fed
  (`fed < boundary < hist.size()`), and the entry fits the budget: prefill stops at `boundary`, the checkpoint
  is saved, prefill continues. Otherwise nothing changes.
- **Entry.** `tok = hist[0, boundary)`, image hashes filtered to positions below `boundary`, no reply
  (`last` empty), `boundary = true`.
- **Supersession.** A boundary save drops entries of either kind that are strict prefixes of it (the same
  conversation's older checkpoints) and touches an identical one. A full save (the departing sequence on a
  switch) never drops a boundary entry.
- **Full saves next to a boundary.** If a boundary entry that is a prefix of the live sequence exists, the full
  save is opportunistic: made only if it fits without evicting anything. A full entry wins a lookup when the
  history is echoed exactly (longest prefix), and the boundary is the fallback when it is not.
- **Lookup, restore, eviction:** unchanged. Longest matching prefix; LRU eviction; the entry being restored is
  never evicted by the save that precedes it.
- **Buffer reuse (review #4):** when a boundary save supersedes exactly one entry, it reuses that entry's
  checkpoint buffer instead of allocating a second one, so the host peak stays one checkpoint.
- **Failure:** a failed or skipped boundary save is logged and the request proceeds. The engine state at the save
  is consistent (`n_past == boundary`, the eval returned).

### Flag and defaults

```
--prefix-cache-boundary N   tokens: checkpoint chat prompts at the end of their history when it is at
                            least N tokens long (default 4096; 0 disables)
```

4096 because a re-prefill of that much costs ~25-40 s here and the checkpoint ~180 MB; below it, the per-turn
cost (next section) buys little.

### Cost, to be measured

Every qualifying turn pays **one device-to-host copy** of the prompt's state (474 ms for 59K tokens, 1.07 GB) and
**one extra eval**, since the opener (5-15 tokens) is fed separately. In a single-conversation session the pool
holds the newest boundary entry (1.07 GB at 59K); at a 2 GB pool, prompts above ~116K tokens do not fit and are
not checkpointed.

### Observability

`/stats` `prefix_cache` gains `boundary_saves`, `boundary_hits` (restores of a boundary entry) and
`t_boundary_save`; restores log whether the entry was a boundary.

### Verification

A GPU window with the server run directly (not through its supervisor), `--prefix-cache 2`: one conversation over a
13K-token document.

| Turn | Client sends | Expected, boundary on | Expected, boundary off |
|---|---|---|---|
| 1 | question | full prefill; boundary saved | full prefill |
| 2 | turn 1's reply verbatim + question | extends; boundary saved; time vs off = the per-turn cost | extends |
| 3 | turn 2's reply **altered** (a word changed) + question | restores turn 2's boundary; prefills reply + turn (hundreds of tokens) | full re-prefill |
| 4 | turn 1's **user message JSON-changed but not in text** (an extra `"name": null` key) + turn 3's reply verbatim + question | continuation fails, the rebuild reproduces every earlier reply's generated tokens, the prompt **extends** the live sequence | same (Part 2 is not behind the flag) - compare with the pre-change release, which re-prefills |

Plus a switch `A B A` to confirm full entries still hit when histories are echoed exactly, and the divergence
log line on every miss.

### Results

Window `boundary b2` (2026-09-16), server run directly with the production launcher's settings (`--ctx 163840
--vram 24 --ram 16 --prefix-cache 2 --prefix-cache-min 256`), one conversation over a 13.4K-token document, five
turns (`scripts/box/boundary_test.py`); **new** = this change (boundary 4096), **old** = release `qwfn-4abd10e`.

| Turn | Old: new tokens / total | New: new tokens / total |
|---|---|---|
| T1 question | 13,397 / 66.3 s | 13,397 / 68.1 s |
| T2 reply verbatim | 24 / 4.9 s | 24 / 5.2 s |
| T3 reply verbatim | 23 / 4.6 s | 23 / 6.0 s |
| **T4 message 1 JSON-changed** (full rebuild) | **13,628 / 47.9 s** | **26 / 6.2 s** (extended: token-stable rebuild) |
| **T5 T4's reply altered** | **13,698 / 59.5 s** | **74 / 6.8 s** (restored T4's boundary in 72 ms) |

- **Both parts work, and the test discriminates.** A first run (`b1`) had T4 after an altered-reply turn, whose
  full rebuild had already re-tokenized the earlier replies, so the old server extended too; `b2` makes the
  earlier replies go in as generated tokens first.
- **The divergence log names the cause.** T5: `first difference at 13612` - four tokens into the re-rendered
  reply, at the `<think>\n\n</think>\n\n` seam - with the held and new snippets side by side.
- **Per-turn cost:** boundary saves 81-202 ms at 13.4K tokens (334-338 MB); prefill of a verbatim turn +0.5-0.8 s
  for the split opener eval; total +0.3-1.4 s per turn. At 59K tokens the copy grows to ~0.5 s.
- **Pool:** single conversation holds one boundary entry (superseded every turn); on the altered-reply turn the
  departing full sequence was saved alongside it into free room, then superseded by the next boundary.
- The earlier `A B A` switch (`b1`) still restored normally with boundary entries in the pool.
