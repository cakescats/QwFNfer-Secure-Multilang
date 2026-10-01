# The B70 configuration

Details of `scripts/b70/qwfn-b70.sh` that the README does not carry: the engine defaults, why the flags have their
values, and the benchmark tools. The configurations (preferred: overlay v4; validated: overlay v2), their switches
and measured effects are in the README; the overlays
are built by `scripts/b70/build-overlay.sh` ([`dense-overlay.md`](dense-overlay.md)); what was tried and rejected
is [`B70-registry.md`](B70-registry.md).

## Switches

Every switch in `scripts/b70/qwfn-b70.sh` is a row of the README's **Major improvements** table (what it changes and
its measured effect); the README's run configuration lists them by group.

Engine defaults in this fork (no switch): the late fold sized to the promotion budget (`QWFN_LATE_FULL=1`
restores), deferred speculative prefetch (`QWFN_PREFETCH_EARLY=1` restores).

## Flags

The flag set is in the README's run configuration. Why these values:

- `--ctx 131072` is the per-session cap; the engine runs one sequence at a time.
- `--vram 25` with overlay v4 (0.6 GB less dense weight than v3): 53% of the expert blocks in VRAM (51% at 24).
  Measured at the limit -- an image plus a 126K-token document at `--ctx 131072`, then the 89K needle -- with no
  GPU memory evicted to system memory and ~0.7 GB still free at the lowest point (1.6 GB at `--vram 24`), on a host
  where nothing else uses the card. On a desktop sharing the card, or with a bigger head, use 24.
  `--ram 8` plus the 3 GB prefix cache plus ~4 GB for the system is what a 31 GB machine can give.
- `--reserve 2048` is not memory held back: the engine only checks, while it sizes the expert tier, that tier plus
  reserve could be allocated, and shrinks the tier until it can. With `--vram` capping the tier first it changes
  nothing (2048, 1536 and 1024 give a byte-identical tier); the headroom that matters is what `--vram` leaves, above.
  (Without the flag the engine's default is 768 MB plus 8 MB per 1K tokens of context beyond 48K, ~1.4 GB at 131K.)
- `--prefix-cache 3`: host-memory checkpoints of conversations switched away from (~2 at 64K, one at 130K).

## Benchmarks

`tools/perf`: `decab.py` (warm decode with the `/stats` decode split), `ctxdec.py` (prefill and decode after
a long prompt from a fixture: `{"messages": [...]}`), `mtpprobe.py` (tok/s, draft acceptance and step cost),
`pcswitch.py` (two alternating conversations: the prefix cache at work), and the `mv_bench` / `moe_bench`
kernel benches (CMake targets: `mv_bench BACKEND_DIR`).
