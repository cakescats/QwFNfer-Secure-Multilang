# Locked host memory for the RAM expert tier and prefill staging

**Status: implemented and reviewed (`adversarial-review-locked-host-spec.md`, `-impl.md`), measured in GPU windows
l1/l2/l3 on 2026-09-17: the mechanism works, no speed benefit shown, ~10-15% cost. Opt-in (`QWFN_LOCK_HOST=1`),
off by default; not deployed.** Branch `prefix-cache`, on top of 9ea36fb (deployed release `qwfn-9ea36fb`).

## Problem, as measured

The RAM expert tier (`expert_cache`, `--ram`) and the prefill streamer's host staging are allocated through
ggml-sycl's host buffer type (`ggml_backend_dev_host_buffer_type`: SYCL `malloc_host`, Level Zero host memory). The
engine treats them as pinned. On the B70's xe driver they are not locked pages:

- `/proc/<pid>/fdinfo` for qwfn-server during a stall (2026-09-17): `drm-resident-system 15238 MiB`,
  **`drm-purgeable-system 15238 MiB`**; host `Shmem` 13.7 GB, `Unevictable` 73 MB, `Mlocked` 0.8 MB.
- Under host memory pressure the kernel reclaims those pages (zswap 2 GB, swap 9 GB at the time) and every expert read
  from the tier faults them back in. Memory PSI `full` reached 18-28%.
- Effect in a live agent session at `--ram 16`: prefill 40-105 tok/s instead of ~300, decode 6-12 tok/s instead of ~15.
  `--ram 12` removed the pressure for now (PSI ~0%, prefill 294 tok/s), at the cost of 4 GB less tier, and the tier can
  still be reclaimed if something else on the host grows.

## Change (amended by the spec review)

**The first version - `mlock` the existing host buffers - would not work:** they are mappings of the DRM render node
with `VM_PFNMAP | VM_IO` (`/proc/<pid>/smaps`: `pf io`, `VmLck` 0 kB), which `mlock` skips silently; the pages behind
them are the driver's backing store. (`adversarial-review-locked-host-spec.md`, issue #1.)

Instead, with `QWFN_LOCK_HOST=1` (opt-in until validated):

1. **Allocate** the expert arena and the prefill staging buffers as anonymous memory (`mmap`, page-aligned, so
   O_DIRECT reads still land in them), `madvise(MADV_HUGEPAGE)` for the arena.
2. **Lock** them with `mlock` (effective on anonymous memory). Measure it: read `VmLck` before and after and log the
   delta. A lock failure (`RLIMIT_MEMLOCK`) logs and continues unlocked.
3. **Register** them with the GPU driver: `zeDriverGet` + `zeDriverGetExtensionFunctionAddress(h,
   "zexDriverImportExternalPointer")` from the already-loaded Level Zero loader (`dlopen`/`dlsym`, no build
   dependency), so SYCL copies from them are direct. Log success or the result code.
4. `arena_pinned_` (which gates async promotions) and `host_pinned_` are true only when the import succeeded.
5. **Release** the import (`zexDriverReleaseImportedPointer`) before `munmap`, after the last copy (the existing
   `settle_promotions` ordering in `shutdown`).

Locking commits the arena at startup (~12 GB page-in) instead of during the first long prefill.

## Budget

Locked memory is really gone from the host: at `--ram 12` ~11.9 GB of tier + 2 x 1.42 GB staging = **~14.8 GB locked**.
Everything else on the host (other workloads, the prefix pool) then competes for the remaining ~17 GB. A service that
launches the server needs `LimitMEMLOCK=infinity`; an
interactive shell or the user manager (`user@1000`) is limited to 8 MiB, so tests run the server in a system transient
unit with `LimitMEMLOCK=infinity`.

## Verification (GPU window)

Same release and launcher settings as deployed (`--vram 25 --ram 12`), three phases per arm, arms `lock` and `nolock`:

1. **Startup:** `VmLck` of the process ~ tier + staging; `Mlocked`/`Unevictable` in `/proc/meminfo` up by the same;
   import logged as successful; fdinfo `drm-purgeable-system` no longer counts the tier; startup time.
2. **No pressure:** short-chat decode and 13.4K / 74.8K prefills - must match `nolock` (no speed cost; this is where
   an unregistered or slowly-copied buffer would show, as the pageable arena once did: 220 -> 80 tok/s).
3. **Under pressure:** a helper allocates and touches memory until `MemAvailable` is ~1 GB, holds it for the length of
   a 13.4K prefill, then frees it. Record prefill tok/s, memory PSI, and the server's `VmSwap` before/after.
   **Pass:** `lock` keeps prefill within ~15% of its no-pressure speed and `VmSwap` does not grow; `nolock` is expected
   to stall.

The pressure phase deliberately squeezes the rest of the host (other workloads will swap for about a minute); it runs only in a
GPU window when there is no traffic.

## Risks

- **Startup can fail to lock** on a host with less free memory than tier + staging; the log says so and the engine
  runs unlocked (today's behaviour).
- **Locked memory cannot be reclaimed** by anything, including when the host really needs it; the budget above is the
  guard, and `--ram` sizes it.
- **The driver import may fail or copies may be slower** from registered anonymous memory than from the driver's own
  host allocation (a pageable arena once cost 220 -> 80 tok/s prefill); the no-pressure phase is designed to show it.

## Measured (GPU windows l1, l2, l3, 2026-09-17, release `qwfn-lockhost-test` = 9ea36fb + this change)

**Mechanism - works as designed.** Startup 10 s either way; with `QWFN_LOCK_HOST=1` the process shows `VmLck`
13.8 GB (11.93 GB tier + 2 x 1.42 GB staging), `/proc/meminfo` `Mlocked`/`Unevictable` the same, the driver reports
0 GB purgeable, and all three blocks log "registered with the GPU driver". Without it the same 13.8 GB sits in
`Shmem`, which the kernel reclaims. No errors, no xe faults in any window.

**Speed, no pressure (l1).** lock vs nolock: short-chat decode 15.7 vs 16.8 tok/s; 13.4K prefill 205 vs 238 tok/s;
74.8K prefill 221 vs 264 tok/s. Locked is 6-16% slower - the cost the review's issue #7 predicted for copies out of
registered anonymous memory (one sample per cell; the same arm also swung 205 -> 288 between runs).

**Under pressure - the benefit did not appear.** A helper held host memory at 0.6-1.1 GB available:

| Window | Arm | decode free | decode squeezed | swap-ins during the run |
|---|---|---|---|---|
| l2 (`--vram 25`) | lock | 12.3 | 14.9 tok/s | 26,925 pages |
| l2 | nolock | 11.8 | 16.9 tok/s | 35,619 pages |
| l3 (`--vram 12`, tier-heavy) | lock | 10.2 | 8.4 tok/s | 45,778 pages |
| l3 | nolock | 9.6 | **9.8 tok/s** | **1,044,304 pages (~4 GB)** |

In l3 the unlocked tier was fully evicted (`VmRSS` 0.0 GB) and faulted ~4 GB back during a single 400-token
generation, yet throughput held. l1's pressure phase likewise showed no collapse (13.4K prefill 288 lock / 237 nolock).

**Why the production failure is not reproduced.** The live stalls (2026-09-17, `--ram 16` and `--ram 10`) had a 70K
context, minutes of sustained pressure, swap 11 GB and zswap 4.9 GB saturated, and `xe_page_fault_work_queue` at 95%
CPU with decode at 0.64 tok/s. A one-minute squeeze with free swap does not recreate that state.

**Decision (2026-09-17):** keep opt-in and off. The cheap decisive test is live: next time an agent session degrades,
restart with `QWFN_LOCK_HOST=1` and compare against the same session. (Later, with the prefill work of 2026-09-22,
it measured +13% prefill at 89K and became part of the B70 configuration.)
