# What

One change, one reason. What breaks today, what this makes true instead.

# How verified

- [ ] `cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release && cmake --build build -j` clean against pinned llama.cpp `b10798-mix-659e406` (paste new warnings, if any)
- [ ] `python -m py_compile tools/qwfn_console.py` (if touched)
- [ ] Runtime check (usage/error paths, console flow, I/O test — say which)
- [ ] Linux path untouched or regression-checked (`#ifdef _WIN32` / `if(WIN32)` only)

# Numbers (if speed-related)

Machine (GPU/VRAM, RAM, drive), before/after tok/s, how measured (`/stats`, replay, or console Self-test).

# Docs

- [ ] SETUP.md / BUILD_WINDOWS.md / README updated if user-visible behavior changed
