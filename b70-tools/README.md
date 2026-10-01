# b70-tools

Tooling around the Arc Pro B70 work that is not part of the engine: how the measurements were made and how the
machine is set up. Nothing here is built or used by the engine itself.

| Directory | What |
|---|---|
| `fixtures/` | `make_fixtures.py`: the long-prompt request files `tools/perf` expects (`ctx20k.json` ... `ctx100k.json`) and the text for the NLL token file |
| `nll/` | `nll_replay.sh`: natural-text replay NLL, the quality measurement behind the dense overlays |
| `profiling/` | `build-unitrace.sh` and how to trace a warm server with Intel's unitrace |
| `host/` | host setup: oneDNN with SYCL, locked memory, the power-cap timer, memory headroom |

The benchmark scripts themselves are in `tools/perf`; the build and configuration are in `scripts/b70` and
`docs/B70-SYCL.md`, `docs/B70-config.md`, `docs/dense-overlay.md`.
