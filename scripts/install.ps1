#Requires -Version 5.1
<#
.SYNOPSIS
  qwfnfer installer for Windows: installs the release bundle, checks the GPU
  driver and starts the console.

  powershell -ExecutionPolicy Bypass -File scripts\install.ps1
  $env:QWFN_ZIP = "C:\path\to\qwfnfer-windows-x86_64-cuda.zip"  # offline install
#>
[CmdletBinding()]
param()
$ErrorActionPreference = 'Stop'
$Repo  = if ($env:QWFN_REPO) { $env:QWFN_REPO } else { 'Apolog1ze-Dev/QwFNfer' }
$Name  = 'qwfnfer-windows-x86_64-cuda'
$Dest  = if ($env:QWFN_HOME) { $env:QWFN_HOME } else { Join-Path $env:LOCALAPPDATA 'qwfnfer' }
$Bin   = if ($env:QWFN_BIN)  { $env:QWFN_BIN }  else { Join-Path $env:USERPROFILE '.local\bin' }

function Say([string]$m) { Write-Host $m -ForegroundColor White }
function Die([string]$m) { Write-Error "qwfnfer install: $m"; exit 1 }

if ([Environment]::OSVersion.Platform -ne 'Win32NT') { Die 'this installer is for Windows (install.sh covers Linux)' }
if ([Environment]::Is64BitOperatingSystem -ne $true) { Die '64-bit Windows only' }
if (-not (Get-Command python -ErrorAction SilentlyContinue)) { Die 'python is needed for the console (install Python 3.10+ and retry)' }
if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
  Write-Host 'note: nvidia-smi not found; the engine needs an NVIDIA GPU with driver 580 or newer'
} else {
  try {
    $drv = (nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>$null | Select-Object -First 1).Trim()
    $gpu = (nvidia-smi --query-gpu=name --format=csv,noheader 2>$null | Select-Object -First 1).Trim()
    Say "GPU: $gpu, driver $drv"
    if (($drv -split '\.')[0] -as [int] -lt 580) { Write-Host "note: the bundled CUDA 13 runtime needs driver 580+ (yours: $drv)" }
  } catch { Write-Host 'note: could not query nvidia-smi' }
}

$tmp = Join-Path ([IO.Path]::GetTempPath()) ('qwfnfer-' + [Guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $tmp | Out-Null
try {
  $zip = Join-Path $tmp "$Name.zip"
  if ($env:QWFN_ZIP) {
    Copy-Item $env:QWFN_ZIP $zip
  } else {
    $url = "https://github.com/$Repo/releases/latest/download/$Name.zip"
    Say "Downloading $url"
    try { Invoke-WebRequest -Uri $url -OutFile $zip -UseBasicParsing } catch { Die "download failed. No release yet? Build from source instead (BUILD_WINDOWS.md), or set QWFN_ZIP" }
  }
  Say "Installing into $Dest"
  if (Test-Path $Dest) { Remove-Item -Recurse -Force $Dest }
  New-Item -ItemType Directory -Path $Dest, $Bin -Force | Out-Null
  Expand-Archive -Path $zip -DestinationPath (Join-Path $tmp 'x') -Force
  $src = Join-Path $tmp "x\$Name"
  if (-not (Test-Path $src)) { $src = Join-Path $tmp 'x' }
  Copy-Item (Join-Path $src '*') $Dest -Recurse -Force
  $link = Join-Path $Bin 'qwfnfer.bat'
  Copy-Item (Join-Path $Dest 'qwfnfer.bat') $link -Force
  # The engine must load: every DLL it needs is in bin\ except the driver's nvcuda.dll.
  # (usage with no args exits nonzero by design; keep that from tripping Stop.)
  $env:PATH = (Join-Path $Dest 'bin') + ';' + $env:PATH
  $oldEap = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
  $probe = & (Join-Path $Dest 'bin\qwfn-server.exe') 2>&1 | Out-String
  $ErrorActionPreference = $oldEap
  if ($probe -notmatch 'usage: qwfn-server') { Die 'the engine did not start' }
  $hub = if ($env:HF_HUB_CACHE) { $env:HF_HUB_CACHE } elseif ($env:HF_HOME) { Join-Path $env:HF_HOME 'hub' } else { Join-Path $env:USERPROFILE '.cache\huggingface\hub' }
  $found = Get-ChildItem -Path $hub -Filter '*.gguf' -Recurse -ErrorAction SilentlyContinue | Where-Object { $_.FullName -like '*Qwen3.8-Flash-Next*' } | Select-Object -First 1
  if ($found) { Say "Model found in the Hugging Face cache ($hub)." }
  else {
    Say 'Get the model once (111 GB; UD-Q3_K_XL is the 90 GB faster choice):'
    Write-Host '    pip install -U huggingface_hub'
    Write-Host '    hf download unsloth/Qwen3.8-Flash-Next-GGUF --include "UD-Q4_K_XL/*" "mmproj-F16.gguf"'
  }
  Say 'Installed. Start it with:'
  Write-Host '    qwfnfer'
} finally {
  Remove-Item -Recurse -Force $tmp -ErrorAction SilentlyContinue
}
