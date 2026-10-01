#Requires -Version 5.1
<#
.SYNOPSIS
  Build the relocatable Windows release bundle: dist\qwfnfer-windows-x86_64-cuda.zip

  The bundle carries everything but the NVIDIA driver: the engine and the
  tokenizer tool (built with -DQWFN_PORTABLE=ON: /arch:AVX2 code, DLLs next to
  the binaries), ggml/llama.cpp's DLLs from a portable build
  (GGML_NATIVE=OFF, CUDA architectures 75;80;86;89;90;120), the CUDA runtime
  DLLs NVIDIA redistributes (cudart, cublas, cublasLt), the console, the
  launcher and the README.

  Prerequisites: Visual Studio 2022 (MSVC + Windows SDK), CMake 3.20+,
  CUDA toolkit 13, and a built llama.cpp tree (upstream
  e85e15cf6d810cd1268498c2e5b657bb3ece47bc or newer -- it must know the
  qwen4exp architecture, older pins fail every Flash-Next start in the
  vocab loader; also the tree the engine is validated against):

    git clone https://github.com/ggerganov/llama.cpp $env:USERPROFILE\.unsloth\llama.cpp
    cd $env:USERPROFILE\.unsloth\llama.cpp; git checkout e85e15cf6d810cd1268498c2e5b657bb3ece47bc
    cmake -S . -B build -DGGML_CUDA=ON "-DCMAKE_CUDA_ARCHITECTURES=75;80;86;89;90;120" `
      -DBUILD_SHARED_LIBS=ON -DGGML_BACKEND_DL=ON -DGGML_NATIVE=OFF `
      -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF `
      -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_SERVER=OFF
    cmake --build build --config Release

  Then:  powershell -ExecutionPolicy Bypass -File scripts\package.ps1
#>
[CmdletBinding()]
param(
  [string]$LlamaCppRoot = $(if ($env:LLAMA_CPP_ROOT) { $env:LLAMA_CPP_ROOT } else { Join-Path $env:USERPROFILE '.unsloth\llama.cpp' }),
  [string]$GgmlLibs     = '',
  [string]$CudaLibs     = '',
  [string]$Version      = ''
)
$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
if (-not $LlamaCppRoot -or -not (Test-Path (Join-Path $LlamaCppRoot 'ggml\include\ggml.h'))) { throw "ggml headers not found under $LlamaCppRoot (pass -LlamaCppRoot)" }
if (-not $GgmlLibs) { $GgmlLibs = Join-Path $LlamaCppRoot 'build\bin\Release' }
if (-not (Test-Path $GgmlLibs)) { $GgmlLibs = Join-Path $LlamaCppRoot 'build\bin' }
$cudaCands = @($CudaLibs, 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.*\bin',
  'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.*\bin') | Where-Object { $_ }
$CudaDir = $null
foreach ($c in $cudaCands) {
  $hit = Get-ChildItem -Path $c -Filter 'cudart64_*.dll' -ErrorAction SilentlyContinue | Select-Object -First 1
  if ($hit) { $CudaDir = $hit.DirectoryName; break }
}
if (-not $CudaDir) { throw 'no CUDA runtime DLLs found (cudart64_*.dll); pass -CudaLibs' }
if (-not $Version) {
  try { $Version = (git describe --tags --always --dirty 2>$null).Trim() } catch { $Version = Get-Date -Format 'yyyyMMdd' }
  if (-not $Version) { $Version = Get-Date -Format 'yyyyMMdd' }
}
$Name = 'qwfnfer-windows-x86_64-cuda'
$Out  = "dist\$Name"
Write-Host "== engine (portable build against $GgmlLibs)"
cmake -S . -B build-portable -G Ninja -DCMAKE_BUILD_TYPE=Release "-DQWFN_PORTABLE=ON" `
  "-DLLAMA_CPP_ROOT=$LlamaCppRoot" "-DLLAMA_CPP_BUILD=$GgmlLibs" 2>&1 | Tee-Object build-portable.cmake.log | Select-Object -Last 5
# -k 0: every failing file in one run; on failure the compiler errors, not just the last lines.
cmake --build build-portable --target qwfn-server qwfn-tok -- -k 0 2>&1 | Tee-Object build-portable.build.log | Select-Object -Last 2
if ($LASTEXITCODE -ne 0) {
  Select-String -Path build-portable.build.log -Pattern ': (fatal )?error |FAILED:|LNK[0-9]+' | Select-Object -First 80 | ForEach-Object { $_.Line }
  throw 'engine build failed (errors above, full log: build-portable.build.log)'
}
Write-Host "== bundle $Out"
Remove-Item -Recurse -Force $Out -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Path "$Out\bin", "$Out\tools\console", "$Out\scripts" -Force | Out-Null
Copy-Item 'build-portable\qwfn-server.exe', 'build-portable\qwfn-tok.exe' "$Out\bin\"
foreach ($dll in @('ggml-base.dll', 'ggml.dll', 'llama.dll', 'ggml-cuda.dll')) {
  $f = Get-ChildItem -Path $GgmlLibs -Filter $dll -ErrorAction SilentlyContinue | Select-Object -First 1
  if (-not $f) { throw "required $dll not found in $GgmlLibs (a CUDA-less llama.cpp build would ship a CPU-only bundle labeled CUDA)" }
  Copy-Item $f.FullName "$Out\bin\"
}
Get-ChildItem -Path $GgmlLibs -Filter 'ggml-cpu*.dll' -ErrorAction SilentlyContinue | ForEach-Object { Copy-Item $_.FullName "$Out\bin\" }
foreach ($pat in @('cudart64_*.dll', 'cublas64_*.dll', 'cublasLt64_*.dll')) {
  Get-ChildItem -Path $CudaDir -Filter $pat | ForEach-Object { Copy-Item $_.FullName "$Out\bin\" }
}
# The Visual C++ runtime, app-local next to the binaries (Microsoft allows this for
# vcruntime/msvcp): a clean Windows has none, and the server then dies with
# 0xC0000135 before printing a line. Taken from the MSVC that built the engine.
$crt = Get-ChildItem -Path "$env:VCToolsRedistDir\x64" -Directory -Filter 'Microsoft.VC*.CRT' -ErrorAction SilentlyContinue | Select-Object -First 1
if (-not $crt) { throw 'the Visual C++ runtime (VCToolsRedistDir\x64\Microsoft.VC*.CRT) was not found: run from a Developer PowerShell' }
foreach ($dll in 'vcruntime140.dll', 'vcruntime140_1.dll', 'msvcp140.dll') { Copy-Item (Join-Path $crt.FullName $dll) "$Out\bin\" }
# The console with its sign-in, translations and images, as scripts/package.sh ships it.
Copy-Item 'tools\qwfn_console.py', 'tools\qwfn_auth.py', 'tools\qwfn_i18n.py', 'tools\qwfn_router.py' "$Out\tools\"
Copy-Item 'tools\console\index.html', 'tools\console\login.html', 'tools\console\console.css' "$Out\tools\console\"
New-Item -ItemType Directory -Force "$Out\tools\console\i18n", "$Out\tools\console\img" | Out-Null
Copy-Item 'tools\console\i18n\*.json' "$Out\tools\console\i18n\"
Copy-Item 'tools\console\img\*.png' "$Out\tools\console\img\"
Copy-Item 'scripts\qwfnfer.bat' "$Out\qwfnfer.bat"
Copy-Item 'README.md', 'README.ru.md', 'LICENSE', 'SETUP.md', 'BUILD_WINDOWS.md' $Out
$Version | Out-File "$Out\VERSION" -Encoding ascii
@"
qwfnfer $Version -- Qwen3.8-Flash-Next on one 16 GB GPU (Windows x86_64, NVIDIA)

1. Unzip anywhere and run:   qwfnfer.bat
   (opens the console at http://127.0.0.1:8090; needs python and an NVIDIA driver 580 or newer)
2. Get the model once:       hf download unsloth/Qwen3.8-Flash-Next-GGUF --include "UD-Q4_K_XL/*" "mmproj-F16.gguf"
   (pip install -U huggingface_hub for the hf command; the console finds the files in the Hugging Face cache)
3. Pick a tier (Chat, Agentic coding, Agentic coding+ or your own Custom one) and press Auto-tune & start:
   the console measures your drive, threads and memory, picks every flag, starts the server and verifies it.
   The endpoint is http://127.0.0.1:8080/v1. Model locations: add any folder that holds the shards.
Everything the engine needs is in bin\ except the NVIDIA driver. See README.md and BUILD_WINDOWS.md.
"@ | Out-File "$Out\INSTALL.txt" -Encoding ascii
Write-Host '== checks'
# usage with no args exits nonzero by design; don't let $ErrorActionPreference turn that into a failure.
$oldEap = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
$probe = & "$Out\bin\qwfn-server.exe" 2>&1 | Out-String
$ErrorActionPreference = $oldEap
if ($probe -notmatch 'usage: qwfn-server') { throw 'qwfn-server.exe does not start' }
Write-Host 'qwfn-server runs (DLLs from bin\ via PATH)'
$zipPath = "dist\$Name.zip"
if (Test-Path $zipPath) { Remove-Item $zipPath }
Compress-Archive -Path $Out -DestinationPath $zipPath
Write-Host ("wrote {0} {1:N0} MB" -f $zipPath, ((Get-Item $zipPath).Length / 1e6))
