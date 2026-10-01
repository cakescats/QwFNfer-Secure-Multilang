# Setup — qwfnfer on Windows (simple guide)

Run Qwen3.8-Flash-Next (125B MoE) on your own gaming PC: 15–25 tok/s on a 16 GB NVIDIA GPU. This page is the short path. Details live in [README.md](README.md) (what it is, measured numbers) and [BUILD_WINDOWS.md](BUILD_WINDOWS.md) (compiling it yourself).

## 1. What you need

- 64-bit Windows 10/11
- NVIDIA GPU with **16 GB VRAM** (reference target; the dense core alone needs ~5 GB) and **driver 580 or newer** (`nvidia-smi` to check)
- ~30 GB RAM (it works with less — a GB of RAM tier costs ~3% decode — but 30 GB is the measured config)
- An NVMe SSD with the model on it (spinning disks are refused by the console for a reason)
- Python 3.10+ (for the console page)
- Disk: 111 GB for the quality quant (UD-Q4_K_XL) or 90 GB for the faster one (UD-Q3_K_XL)

## 2. Install the app

Download `qwfnfer-windows-x86_64-cuda.zip` from the [Releases](../../releases) page, unzip it anywhere (e.g. `C:\qwfnfer`), and run the installer from that folder:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\install.ps1
```

The bundle ships its own ggml DLLs and CUDA 13 runtime beside the binary — no PATH manipulation, nothing added to your system. The installer puts it under `%LOCALAPPDATA%\qwfnfer`, checks your GPU driver (needs 580+), and proves the engine binary loads. If it complains about a missing DLL, the archive wasn't fully unzipped — every DLL lives in `bin\` beside the engine. (No release yet, or offline? `set QWFN_ZIP=C:\path\to\qwfnfer-windows-x86_64-cuda.zip` first — same command, no download.)

Uninstall: delete `%LOCALAPPDATA%\qwfnfer` and `%USERPROFILE%\.local\bin\qwfnfer.bat`. The model files stay where you put them.

## 3. Download the model (once)

```powershell
pip install -U huggingface_hub
hf download unsloth/Qwen3.8-Flash-Next-GGUF --include "UD-Q4_K_XL/*" "mmproj-F16.gguf"
```

- `UD-Q4_K_XL` — the quality choice (13–16 tok/s chat).
- `UD-Q3_K_XL` — the faster choice (20–21 tok/s chat). Swap it into the `--include` above.
- `mmproj-F16.gguf` — adds vision (image input). Optional.
- `MTP/*` — adds the draft head (verified speculative decoding, exact output). Optional: `--include "MTP/*"`.

The console finds downloads in your Hugging Face cache automatically. Put the file somewhere else with `--local-dir`? Add that folder under *Model locations* on the Serve tab.

## 4. Run it

```powershell
qwfnfer
```

If your GGUFs sit outside the Hugging Face cache (you downloaded with
`--local-dir`, like the Flash-Next command most people use), skip the picking:
point the console straight at the first shard and it serves right away (the
folder is saved into *Model locations* automatically):

```powershell
qwfnfer --start --model D:\models\flash-next\UD-Q4_K_XL\Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf --no-browser
```

It opens `http://127.0.0.1:8090`. Pick your quant and a tier:

| Tier | Context | For |
|---|---|---|
| Chat | 32K | short conversations, fastest decode |
| Agentic coding | 128K | coding harnesses, tool calls |
| Agentic coding+ | 256K | longest sessions, full trained context |
| Custom | yours | anything you save under *Advanced settings* |

Press **Auto-tune & start** (about five minutes, once per model): it measures your drive, sweeps the CPU thread count on the running server, sizes the RAM tier with headroom to spare, and verifies on a short chat plus a 16K–32K-token document. After that the tier card shows your measured speed. **Start server** skips re-measuring; **Self-test** measures a running server. The OpenAI endpoint is `http://127.0.0.1:8080/v1`.

> **One hard rule:** run Flash-Next only in `qwfnfer` / `qwfn-server`.
> Stock `llama.cpp` builds (and any bundle or checkout pinned before `qwen4exp`
> support) fail on these files with `unknown model architecture: 'qwen4exp'` /
> `failed to load vocab`. That error names an outdated *loader*, not a broken
> model — the engine reads its tokenizer through llama's loader, so the loader
> must know the architecture. Bundles `windows-beta4` and up ship a
> `qwen4exp`-aware build; from source, check out the commit in
> [BUILD_WINDOWS.md](BUILD_WINDOWS.md).

## Smaller GPUs: 8 GB and 12 GB (RTX 5060 / 5070 and friends)

It runs — slower, not broken. The console sizes everything from the VRAM it
finds, and the engine shrinks its tier to what fits (or runs tierless):

| GPU | What to expect (UD-Q4_K_XL, predictions) |
|---|---|
| 16 GB (4080-class) | 6–8 GB VRAM tier, 13–16 tok/s chat |
| 12 GB (5070) | ~4 GB tier on Chat (~13 tok/s); Agentic tiers step down to 64K to keep a tier |
| 8 GB (5060) | No VRAM tier — experts stream from RAM/NVMe, ~7–8 tok/s. Use the Chat tier |

Rules of thumb: the ~4.8 GB dense core must fit with room to spare — if it
can't, starting the server refuses up front with the numbers instead of dying
in `cudaMalloc`. 50-series (Blackwell, sm_120a) is covered by the bundled
CUDA 13 runtime and ggml build; driver 580+ still required. 16 GB RAM pairings
work but keep the headroom at 3 GB: the RAM tier is what carries an 8 GB GPU,
and every GB is ~3% of decode.

## Other Qwen3.8 checkpoints (27B-class and other sizes)

Any `qwen4exp`-architecture GGUF loads: layer counts, experts, context length
and KV geometry are read from the file's own metadata, not hardcoded. Two
things adapt automatically:

- The console reads each download's GGUF headers (expert block size, dense
  core, trained context, attention dims) and plans from those — a 128K-trained
  checkpoint gets its presets capped at 128K with a note, instead of a 256K
  tier that would fail.
- Speed predictions stay 125B-fitted until Auto-tune measures your machine;
  treat pre-tune tok/s on other sizes as a starting point, not a promise.

## What about Qwen3.8-27B (non-flash)?

Not here - and that's structural, not a missing flag. The 27B is a **dense**
`qwen35` model (64 plain layers, no routed experts); this engine *is* an MoE
runtime (expert tiers, speculative prefetch, in-graph MoE, PLE table). There is
nothing in it for a dense checkpoint to use, so it refuses those files up front
- in the console's scan list and on the command line - instead of failing
halfway. Run the 27B where dense models run best: any **llama.cpp that knows
the `qwen35` architecture (build b10502 or newer, current master, or the
pinned Unsloth mix in BUILD_WINDOWS.md** - verified working, 27 tok/s on CPU
for the 0.8B sibling, and the safetensors-to-serving loop below proven
end to end on Windows).

Easiest: one script does convert + serve from your safetensors folder:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\serve-27b.ps1 -ModelDir D:\models\Qwen3.8-27B -Quant Q4_K_M
```

It makes a converter venv once (torch CPU), converts to F16, quantizes to your
tier, and serves on `http://127.0.0.1:8081` (ctx 131072 — the 27B's KV is
cheap: only 16 of its 64 layers keep one). Needs ~70 GB free during the run
for the 50 GB F16 intermediate (deleted after, unless `-KeepIntermediate`).
Tiers: `Q4_K_M` (~16.5 GB, default), `Q3_K_M` (~13 GB), `IQ3_XXS` (~11.6 GB).
By hand instead:

```powershell
hf download qtum/Qwen3.8-27B-GGUF --include "Qwen3.8-27B-Q4_K_M.gguf"
llama.exe cli -m Qwen3.8-27B-Q4_K_M.gguf -c 8192       # chat in the terminal
llama.exe serve -m Qwen3.8-27B-Q4_K_M.gguf -c 131072   # OpenAI endpoint for your clients
```

(Modern llama.cpp ships one `llama.exe` with `cli`/`serve` subcommands, not
separate `llama-cli`/`llama-server` binaries. Q4_K_M is about 16.5 GB, the
default pick; smaller VRAM? take `Q3_K_M` at about 13 GB or `IQ3_XXS` at about
11.6 GB. Add `--mmproj mmproj-...gguf` for vision.) `llama.exe serve` speaks
the same OpenAI endpoint shape, so your clients transfer over.

One rule for both engines: **qwfnfer and llama.cpp read GGUF, not safetensors.**
Got `.safetensors` files? Convert once (that's what `serve-27b.ps1` automates
above, and what `convert_hf_to_gguf.py` + `llama-quantize` do by hand), then
serve the GGUF. Never rename a `.safetensors` to `.gguf` — the formats are
unrelated and the loader will just report a bad magic.

## Sending logs (so it can be fixed)

Something broken? Collect these three things and open an issue at
[github.com/cakescats/QwFNfer-Secure-Multilang/issues](https://github.com/cakescats/QwFNfer-Secure-Multilang/issues)
(bug-report template walks you through it):

1. **What you did + what happened**: quant and tier (the Serve banner line
   names every flag — paste it), the model folder layout, what you expected.
2. **The server log**: `%LOCALAPPDATA%\qwfn-console\server.log` — tail 60
   lines is usually enough (the console's Log tab shows the same). For the
   27B path: the `serve-27b.ps1` console output instead.
3. **Live numbers**: while it runs, save `http://127.0.0.1:8080/stats`
   (qwfnfer) or `http://127.0.0.1:8081/health` + the request that failed
   (27B/llama.cpp), plus your GPU + driver (`nvidia-smi`), RAM, and where the
   model sits (NVMe? network drive? that matters more than people think).

No logs, no fix — "it doesn't work" can't be debugged. With the three above,
most issues get a cause within one reply. What the template asks for beyond
this (versions, repro steps) is in [CONTRIBUTING.md](CONTRIBUTING.md).

(Modern llama.cpp ships one `llama.exe` with `cli`/`serve` subcommands, not
separate `llama-cli`/`llama-server` binaries. Q4_K_M is about 16.5 GB, the
default pick; smaller VRAM? take `Q3_K_M` at about 13 GB or `IQ3_XXS` at about
11.6 GB. Add `--mmproj mmproj-...gguf` for vision.) `llama.exe serve` speaks
the same OpenAI endpoint shape, so your clients transfer over.

Anything that is not `qwen4exp` is refused at scan time with the reason named
(the engine implements that graph, not a general one).

## 5. Point your tools at it

Endpoint: `http://127.0.0.1:8080/v1` (any API key works — none is checked). Model id: `qwen3.8-flash-next`.

- **Any OpenAI client**: Unsloth Studio (custom provider), Open WebUI, your own scripts. Streaming, thinking (`reasoning_effort`: `xhigh` | `medium` | `low` | `off`), tool calling, vision, `timings` — see README §4.
- **Claude Code** (native, no plugin): point it at the server root, not `/v1`:

```powershell
$env:ANTHROPIC_BASE_URL='http://127.0.0.1:8080'; $env:ANTHROPIC_AUTH_TOKEN='local'; $env:ANTHROPIC_MODEL='qwen3.8-flash-next'; claude
```

Use the Agentic coding tier or larger (its first request is ~17K tokens of system prompt, prefilled once, then continued from cache).

## 6. How it works (30-second version)

Each token activates ~1.1 GB of expert weights — more than any SSD streams. So the engine keeps a **VRAM tier** (~2,600–3,900 hot experts), a **pinned-RAM tier** (13–15 GB), and the rest on NVMe, reading whole expert slices (0.6–1.2 MB) with parallel positional reads instead of 4 KiB page faults (23× the bandwidth). The next layer's experts are predicted a layer early (95–96% right) and prefetched while the current layer computes. Sparse attention keeps decode flat to 160K+ context. Full story: README *"How it works"*.

Windows specifics: the Linux build reads via `io_uring`; this port reads with overlapped positional `ReadFile` (`FILE_FLAG_NO_BUFFERING` where the volume allows it), through a thread pool by default or an I/O completion port with `--io-uring`. Same tiers, same graphs, same console — the numbers above are the Linux reference machine; your auto-tune measures yours.

## 7. Troubleshooting

| Symptom | Check |
|---|---|
| Console finds no model | *Model locations* lists every scanned folder and why each GGUF was rejected; re-run the same `hf download` to resume interrupted downloads |
| Slow decode | Log tab: VRAM tier size, RAM tier clamps, VRAM audit; `/stats`: hit rate + VRAM-served share; another app may hold VRAM |
| Server won't start | `server.log` in `%LOCALAPPDATA%\qwfn-console`; one engine at a time — `taskkill /F /IM qwfn-server.exe` kills an orphan |
| Engine missing-DLL error | Bundle users: everything is in `bin\`; source users: the console adds the llama.cpp `build\bin` to `PATH` automatically |
| Release zip won't install | `install.ps1` refuses only on <64-bit, missing Python, or an engine that won't load — read its message, it names the cause |

Still stuck? Open an issue (see [CONTRIBUTING.md](CONTRIBUTING.md)): paste the tier card, the Log tab tail, your GPU/RAM/drive, and `/stats` output. Full checklist of what to collect: [Sending logs](#sending-logs-so-it-can-be-fixed) above.
