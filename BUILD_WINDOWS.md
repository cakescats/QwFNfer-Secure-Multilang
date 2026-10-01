# Building qwfnfer on Windows

Linux is the reference platform. This is the Windows port: same engine, same
console, same tiers. Everything below the engine that touches the OS goes
through one platform layer (`src/qwfn_plat.h`, `qwfn_plat_win32.cpp`):
unbuffered `CreateFileW` (`FILE_FLAG_NO_BUFFERING` when the volume allows it),
positional `ReadFile` + `OVERLAPPED`, `MapViewOfFile` for `mmap`,
`GlobalMemoryStatusEx` for `/proc/meminfo`; `liburing` is not needed. The NVMe
layer has two read engines, as on Linux: a thread pool (the default) and an
I/O completion port (`--io-uring` selects it -- the Linux name of the async
engine). `qwfn-iobench` runs both on your drive; keep the faster one.

> This file comes from the Windows port by Rafiekuntest
> ([Rafiekuntest/QwFNfer](https://github.com/Rafiekuntest/QwFNfer)), merged into
> this fork; the engine's platform layer is this fork's own.

## Requirements

- 64-bit Windows 10/11, NVIDIA GPU, driver 580 or newer (CUDA 13 runtime)
- Visual Studio 2022 (MSVC + Windows SDK) **or** MinGW-w64; CMake 3.20+; Ninja
- CUDA toolkit 13 (`nvcc` on PATH)
- Python 3.10+ (console + launcher)
- A built [llama.cpp](https://github.com/ggerganov/llama.cpp) tree at
  `%USERPROFILE%\.unsloth\llama.cpp`, commit `e85e15cf6d810cd1268498c2e5b657bb3ece47bc`
  (2026-09-25) or newer — it must know the `qwen4exp` architecture. Older pins
  (e.g. the Unsloth `b10798-mix-659e406`) fail every Flash-Next start with
  `unknown model architecture: 'qwen4exp'` / `failed to load vocab`, because
  the engine loads its tokenizer through llama's model loader. This tree is
  also what the forward pass is validated against.

No admin rights and no toolkit installer? The whole CUDA side works from
NVIDIA's pip wheels (no admin needed) — this is how the first release bundle
was built:

```powershell
python -m venv E:\cuda-env; E:\cuda-env\Scripts\python -m pip install --extra-index-url https://pypi.ngc.nvidia.com nvidia-cuda-nvcc nvidia-cuda-runtime nvidia-cublas nvidia-cuda-cccl
```

That gives `E:\cuda-env\Lib\site-packages\nvidia\cu13` with `bin\nvcc.exe`,
`bin\x86_64\*.dll`, `include\` and `lib\x64\`. Two gaps to close manually:

1. The cublas wheel ships no import `.lib`. Generate them from the DLL exports
   (MSVC `dumpbin` + `lib`; ~2 minutes), straight into `lib\x64` next to
   `cudart.lib`:

```powershell
foreach ($d in 'cublas64_13.dll','cublasLt64_13.dll') {
  $base = $d -replace '\.dll$',''; $lib = ($base -replace '64_13$','') + '.lib'
  $names = dumpbin /EXPORTS "E:\cuda-env\Lib\site-packages\nvidia\cu13\bin\x86_64\$d" |
    Select-String '^\s+\d+\s+[0-9A-F]+\s+[0-9A-F]+\s+(\S+)' | ForEach-Object { $_.Matches.Groups[1].Value } |
    Where-Object { $_ -notmatch '^(ordinal|hint|RVA|name|Summary)' } | Sort-Object -Unique
  "LIBRARY $base`r`nEXPORTS`r`n" + ($names -join "`r`n") | Out-File -Encoding ascii "$env:TEMP\$base.def"
  lib /DEF:"$env:TEMP\$base.def" /OUT:"E:\cuda-env\Lib\site-packages\nvidia\cu13\lib\x64\$lib" /MACHINE:X64
}
```

2. Point CMake at it: `-DCUDAToolkit_ROOT=E:/cuda-env/Lib/site-packages/nvidia/cu13`
   plus `-DCMAKE_CUDA_COMPILER=.../bin/nvcc.exe` (the pip layout is found as a
   toolkit, but `enable_language(CUDA)` still wants the compiler path).

```powershell
git clone https://github.com/ggerganov/llama.cpp $env:USERPROFILE\.unsloth\llama.cpp
cd $env:USERPROFILE\.unsloth\llama.cpp; git checkout e85e15cf6d810cd1268498c2e5b657bb3ece47bc   # qwen4exp-aware (see above)
$cu = 'E:/cuda-env/Lib/site-packages/nvidia/cu13'   # or your toolkit root
cmake -S $env:USERPROFILE\.unsloth\llama.cpp -B $env:USERPROFILE\.unsloth\llama.cpp\build -G Ninja -DCMAKE_BUILD_TYPE=Release `
  -DBUILD_SHARED_LIBS=ON -DGGML_BACKEND_DL=ON -DGGML_NATIVE=OFF -DGGML_CUDA=ON `
  "-DCMAKE_CUDA_ARCHITECTURES=75;80;86;89;90;120" "-DCMAKE_CUDA_COMPILER=$cu/bin/nvcc.exe" "-DCUDAToolkit_ROOT=$cu" `
  -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF `
  -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_SERVER=OFF
cmake --build $env:USERPROFILE\.unsloth\llama.cpp\build --target ggml-base ggml ggml-cpu ggml-cuda llama
```

Notes: `ggml-cuda` builds as a plugin (no import lib needed — the engine loads
it at runtime, the bundle just carries the DLL). The CUDA compile is the long
pole (fattn template instances × 6 archs); give it time. `-DLLAMA_CPP_BUILD`
points at the `build\bin` dir if yours differs.

## Build the engine

```powershell
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release  # -DLLAMA_CPP_ROOT=<path> if the tree is elsewhere
cmake --build build
```

Binaries land in `build\Release\` (Ninja: `build\`): `qwfn-server.exe`,
`qwfn-tok.exe`, plus the dev tools (`qwfn-chat.exe`, `qwfn-iobench.exe`, …).

## Run it

```powershell
hf download unsloth/Qwen3.8-Flash-Next-GGUF --include "UD-Q4_K_XL/*" "mmproj-F16.gguf"
scripts\qwfnfer.bat
```

It opens `http://127.0.0.1:8090`. Pick a tier and press **Auto-tune & start** —
the console measures the drive (buffered random 2 MiB reads on Windows;
`FILE_FLAG_NO_BUFFERING` in the engine itself), sweeps the CPU thread count,
sizes the RAM tier with `GlobalMemoryStatusEx`, and verifies on a short chat
plus a 16K–32K-token document. Endpoint: `http://127.0.0.1:8080/v1`.

`scripts\qwfnfer.bat --no-browser` runs the console in the terminal;
`--port 8091` picks another console port.

## Install / bundle

```powershell
powershell -ExecutionPolicy Bypass -File scripts\install.ps1        # from a release zip
powershell -ExecutionPolicy Bypass -File scripts\package.ps1        # build dist\qwfnfer-windows-x86_64-cuda.zip
```

`package.ps1` writes the zip that `install.ps1` consumes (`$env:QWFN_ZIP`
for an offline path). Uninstall: delete `%LOCALAPPDATA%\qwfnfer` and
`%USERPROFILE%\.local\bin\qwfnfer.bat`.

## Notes and limits

- `--io-uring` selects the completion-port engine (Windows has no io_uring).
- `FILE_FLAG_NO_BUFFERING` needs sector-aligned offset/length/buffer; the
  engine's `dio_align()`/`dio_alloc()` layout already guarantees it, and falls
  back to buffered reads per file when the volume refuses it.
- The console stores state in `%LOCALAPPDATA%\qwfn-console` (`config.json`,
  `server.log`); the Hugging Face cache resolves via `HF_HUB_CACHE` /
  `HF_HOME` / `%USERPROFILE%\.cache\huggingface\hub`.
- One engine at a time (same as Linux); `taskkill /F /IM qwfn-server.exe`
  stops an orphan.
- The MX230-class GPUs (2 GB) cannot hold the ~5 GB dense core: the engine
  will refuse to size a VRAM tier there. A 16 GB GPU is the reference target.
