# Contributing (Windows notes from the Rafiekuntest/QwFNfer port)

Upstream is [Apolog1ze-Dev/QwFNfer](https://github.com/Apolog1ze-Dev/QwFNfer) (Linux). This fork exists for one reason: **Windows support**. Model behavior, tiers, and measured numbers come from upstream — please don't file those here.

## What belongs here

- Windows build breaks (MSVC/MinGW, CMake, CUDA packaging)
- Windows runtime bugs: overlapped I/O, file mapping, `install.ps1` / `package.ps1` / `qwfnfer.bat`, console Windows paths (`%LOCALAPPDATA%`, tasklist, `nvidia-smi` parsing, drive probe)
- Docs: [SETUP.md](SETUP.md), [BUILD_WINDOWS.md](BUILD_WINDOWS.md), README Windows sections
- Measured Windows numbers (drive probe vs decode, tier sizing on non-reference machines)

## What belongs upstream

Model quality, Linux engine behavior, the cost model constants, new architectures. If you're unsure, open it here first and we'll redirect you rather than lose it.

## How to contribute

1. Fork this repo, branch from `windows-port` (`git checkout -b fix/thing windows-port`).
2. Keep the Linux code compiling: every platform change must be `#ifdef _WIN32` (or CMake `if(WIN32)`), never a behavior change on Linux. If you touch shared logic, say how you regression-checked it.
3. Verify before you push:
   - `cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release && cmake --build build -j` against the pinned llama.cpp mix (`b10798-mix-659e406`) — zero errors, and note new warnings.
   - `python -m py_compile tools/qwfn_console.py`.
   - If you touched the I/O path, run the engine's own checks you can: `qwfn-server` / `qwfn-inspect` usage + error paths, and the console's Serve flow.
4. Write the PR like a lab note: what you changed, what you measured, machine (GPU/VRAM, RAM, drive, Windows build), before/after numbers where speed is involved. One change per PR.

## Reporting issues

Use the templates (bug report / feature request). A good bug report has:

- Windows version, GPU + driver (`nvidia-smi` output), RAM, drive type + where the model sits
- Quant served (Q4/Q3), tier + flags (the Serve banner line names them all)
- What you expected, what happened, and the Log tab tail (`%LOCALAPPDATA%\qwfn-console\server.log`)
- `/stats` output while it runs (hit rate, VRAM-served share)
- Whether the same setup works on the Linux upstream (isolates port bugs from engine bugs)

No response in a week? Ping the issue — things slip, malice is rare.

## Release process (maintainers)

1. `scripts\package.ps1` → `dist\qwfnfer-windows-x86_64-cuda.zip` (portable `/arch:AVX2` engine, ggml + CUDA DLLs beside the binaries).
2. Smoke-test the zip on a clean machine: `install.ps1`, `hf download` (small `--include` first), `qwfnfer` → Auto-tune → chat.
3. Tag `windows-vN`, GitHub release with the zip, notes = measured deltas + known issues.
4. Never force-push `windows-port`; rebase the work, merge the PR.

## License

Apache-2.0, same as upstream (see [LICENSE](LICENSE)). By contributing you agree your work ships under it.
