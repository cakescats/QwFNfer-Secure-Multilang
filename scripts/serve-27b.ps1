#Requires -Version 5.1
<#
.SYNOPSIS
  Serve a Qwen3.8-27B (dense qwen35) safetensors checkout in llama.cpp on Windows.

  The qwfnfer engine only serves MoE qwen4exp checkpoints, so the dense 27B
  goes through llama.cpp instead: this script converts safetensors -> GGUF
  (once) and starts `llama.exe serve` with GPU offload, vision and a ctx that
  fits the card.

  powershell -ExecutionPolicy Bypass -File scripts\serve-27b.ps1 -ModelDir D:\models\Qwen3.8-27B -Quant Q4_K_M
  powershell -ExecutionPolicy Bypass -File scripts\serve-27b.ps1 -ModelDir D:\models\Qwen3.8-27B -NoServe   # convert only

.PARAMETER ModelDir
  Folder with the safetensors checkpoint (model.*.safetensors + config.json +
  tokenizer files), e.g. an `hf download ... --local-dir` target.
.PARAMETER OutDir
  Where the GGUF goes. Default: <ModelDir>\gguf.
.PARAMETER Quant
  Conversion target. Q4_K_M (default, ~16.5 GB, the all-round pick),
  Q3_K_M (~13 GB) or IQ3_XXS (~11.6 GB) for smaller VRAM.
.PARAMETER Ctx
  Server context. Default 131072 (27B KV is cheap: 16 attn layers only).
.PARAMETER Port
  Server port. Default 8081 (qwfnfer's own server takes 8080).
.PARAMETER LlamaMain
  ggml-org/llama.cpp checkout with a qwen3_5-aware converter (b10502+).
  Cloned shallow/sparse automatically when missing.
.PARAMETER LlamaBin
  Folder with llama.exe (with CUDA backend). Defaults to this fork's own
  build (%USERPROFILE%\.unsloth\llama.cpp\build-cuda86\bin) when present,
  else <LlamaMain>\build\bin.
.PARAMETER Python
  Python for the converter venv. Default: the `python` on PATH.
.PARAMETER NoServe
  Convert only, do not start the server.
.PARAMETER KeepIntermediate
  Keep the F16 intermediate (~50 GB for 27B) instead of deleting it after
  quantizing. Needs ~70 GB free during the run either way.
#>
[CmdletBinding()]
param(
  [Parameter(Mandatory)] [string]$ModelDir,
  [string]$OutDir = '',
  [ValidateSet('Q4_K_M', 'Q3_K_M', 'IQ3_XXS')] [string]$Quant = 'Q4_K_M',
  [int]$Ctx = 131072,
  [int]$Port = 8081,
  [string]$LlamaMain = (Join-Path ([Environment]::GetFolderPath('UserProfile')) '.llama-main'),
  [string]$LlamaBin = '',
  [string]$Python = 'python',
  [switch]$NoServe,
  [switch]$KeepIntermediate
)
$ErrorActionPreference = 'Stop'
function Die([string]$m) { Write-Error $m; exit 1 }

if (-not (Test-Path (Join-Path $ModelDir 'config.json'))) { Die "no config.json in $ModelDir (pass the safetensors folder, e.g. an hf download --local-dir target)" }
if (-not (Get-ChildItem -Path $ModelDir -Filter *.safetensors | Select-Object -First 1)) { Die "no *.safetensors in $ModelDir" }
if (-not $OutDir) { $OutDir = Join-Path $ModelDir 'gguf' }
New-Item -ItemType Directory -Path $OutDir -Force | Out-Null
$stem = (Get-ChildItem -Path $ModelDir -Filter *.safetensors | Select-Object -First 1).BaseName -replace '\.safetensors.*$', '' -replace '-00001-of-.*$', ''
$gguf = Join-Path $OutDir ("$stem-" + $Quant + '.gguf')

# 1. Converter: mainline llama.cpp (qwen3_5 support), shallow + sparse.
# NOTE: no `||`/`&&` below -- Windows PowerShell 5.1 does not have them.
if (-not (Test-Path (Join-Path $LlamaMain 'convert_hf_to_gguf.py'))) {
  Write-Host "Cloning llama.cpp converter into $LlamaMain ..."
  git clone --depth 1 --filter=blob:none --sparse https://github.com/ggml-org/llama.cpp $LlamaMain
  if ($LASTEXITCODE) { Die 'git clone failed' }
  git -C $LlamaMain sparse-checkout set --skip-checks convert_hf_to_gguf.py conversion ggml-py gguf-py
  if ($LASTEXITCODE) { Die 'sparse checkout failed' }
}
$conv = Join-Path $LlamaMain 'convert_hf_to_gguf.py'

# 2. Converter venv (torch CPU + safetensors + transformers), reused across runs.
$venv = Join-Path $LlamaMain '.conv-env'
$vpy = Join-Path $venv 'Scripts\python.exe'
if (-not (Test-Path $vpy)) {
  Write-Host 'Creating converter venv (torch CPU, one time, ~500 MB) ...'
  & $Python -m venv $venv
  if ($LASTEXITCODE) { Die 'venv failed' }
  & $vpy -m pip install -q --timeout 300 --retries 10 torch --index-url https://download.pytorch.org/whl/cpu
  if ($LASTEXITCODE) { Die 'torch install failed' }
  & $vpy -m pip install -q --timeout 300 --retries 10 safetensors transformers numpy
  if ($LASTEXITCODE) { Die 'converter deps failed' }
}

# 3. Convert (skipped when the output is already there). The converter writes an
# F16 intermediate; K-quants need llama-quantize as a second step.
$inter = Join-Path $OutDir ("$stem-f16.gguf")
if (Test-Path $gguf) {
  Write-Host "Reusing $gguf"
} else {
  if (-not (Test-Path $inter)) {
    Write-Host "Converting $ModelDir -> $inter (F16 intermediate; one time, several minutes) ..."
    & $vpy $conv $ModelDir --outtype f16 --outfile $inter
    if ($LASTEXITCODE) { Die 'conversion failed (is this a qwen3_5 safetensors checkout?)' }
  }
  $qb = if ($LlamaBin) { Join-Path $LlamaBin 'llama-quantize.exe' } else { '' }
  if (-not $qb -or -not (Test-Path $qb)) {
    $qb = Get-ChildItem -Path @((Join-Path ([Environment]::GetFolderPath('UserProfile')) '.unsloth\llama.cpp\build-cuda86\bin'), (Join-Path $LlamaMain 'build\bin')) `
      -Filter 'llama-quantize.exe' -ErrorAction SilentlyContinue | Select-Object -First 1 -ExpandProperty FullName
  }
  if (-not $qb) { Die 'llama-quantize.exe not found (build mainline llama.cpp tools, or point -LlamaBin at a build that has it)' }
  Write-Host "Quantizing -> $gguf ($Quant; one time, several minutes) ..."
  & $qb $inter $gguf $Quant
  if ($LASTEXITCODE) { Die 'quantize failed' }
  if (-not $KeepIntermediate) { Remove-Item $inter -Force; Write-Host '(intermediate removed; -KeepIntermediate to keep it)' }
}
$toolsDir = Join-Path $PSScriptRoot '..\tools'
$arch = & $Python -c "import sys; sys.path.insert(0, sys.argv[1]); import qwfn_console as C; print(C.gguf_arch(sys.argv[2]))" $toolsDir $gguf 2>$null
if ($arch -and $arch -ne 'qwen35') { Write-Warning "GGUF arch is '$arch', expected 'qwen35' (Qwen3.8-27B dense)" }
if ($NoServe) { Write-Host "Wrote $gguf"; return }

# 4. Serve binary: this fork's CUDA build first, else the mainline build dir.
if (-not $LlamaBin) {
  $cands = @((Join-Path ([Environment]::GetFolderPath('UserProfile')) '.unsloth\llama.cpp\build-cuda86\bin'),
             (Join-Path $LlamaMain 'build\bin'))
  $LlamaBin = $cands | Where-Object { Test-Path (Join-Path $_ 'llama.exe') } | Select-Object -First 1
}
$exe = if ($LlamaBin) { Join-Path $LlamaBin 'llama.exe' } else { '' }
if (-not $exe -or -not (Test-Path $exe)) {
  Die ("no llama.exe found. Build mainline llama.cpp with CUDA first (see BUILD_WINDOWS.md, llama.cpp section), then pass -LlamaBin <build\bin>. " +
       "Checked: " + (($cands | ForEach-Object { $_ }) -join ', '))
}
$env:PATH = "$LlamaBin;" + $env:PATH

# 5. Vision projector next to the GGUF or the safetensors (optional).
$mm = Get-ChildItem -Path $OutDir, $ModelDir -Filter 'mmproj-*.gguf' -ErrorAction SilentlyContinue | Select-Object -First 1 -ExpandProperty FullName
$mmArgs = @(); if ($mm) { $mmArgs = @('--mmproj', $mm); Write-Host "Vision: $mm" }

Write-Host "Serving $gguf on http://127.0.0.1:$Port (ctx $Ctx; Ctrl+C to stop) ..."
& $exe serve -m $gguf @mmArgs -c $Ctx --port $Port
