#!/usr/bin/env python3
"""MTP probe against a running qwfn-server: mtpprobe.py PORT LABEL WORKLOAD [N_WARM] [N_MEAS] [N_TOK]

WORKLOAD: essay (decab's greedy essay prompt), code (a coding-agent style request, greedy),
code-t07 (the same at temperature 0.7, top_p 0.95: what pi sends by default).
Per measured request, from /stats `last`: tok/s, the draft head's steps / drafted / accepted, and the
step cost: a step emits 1 + its accepted drafts, so steps = generated - accepted and
ms/step = generation_ms / steps. Against a plain server (no drafts) ms/step is the T=1 token cost, so
(MTP ms/step) / (plain ms/step) is the verify-step multiplier the revival hinges on."""
import json, sys, urllib.request

port, label, wl = sys.argv[1], sys.argv[2], sys.argv[3]
n_warm = int(sys.argv[4]) if len(sys.argv) > 4 else 1
n_meas = int(sys.argv[5]) if len(sys.argv) > 5 else 3
n_tok = int(sys.argv[6]) if len(sys.argv) > 6 else 256
base = f"http://127.0.0.1:{port}"
ESSAY = ("Write a detailed technical essay about memory hierarchies, swapping, GPUs and mixture-of-experts "
         "inference. Be specific and long.")
CODE = ("Write a Python module implementing an LRU cache with TTL expiry: a class with get, put, delete and "
        "a background-free lazy expiry, full type hints, docstrings, and a pytest test file covering "
        "eviction order, expiry and capacity edge cases. Output only the code.")
body = {"model": "x", "max_tokens": n_tok, "reasoning_effort": "off", "ignore_eos": True,
        "messages": [{"role": "user", "content": ESSAY if wl == "essay" else CODE}]}
if wl == "code-t07":
    body.update({"temperature": 0.7, "top_p": 0.95})
else:
    body["temperature"] = 0


def get(path):
    with urllib.request.urlopen(base + path, timeout=10) as r:
        return json.load(r)


def one():
    req = urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        json.load(r)
    return get("/stats")["last"]


for _ in range(n_warm):
    one()
tot = {"gen": 0, "ms": 0.0, "acc": 0, "dr": 0, "pairs": 0}
rows = []
for _ in range(n_meas):
    L = one()
    sp = L.get("speculative") or {}
    g, ms = L["generated_tokens"], L["generation_ms"]
    acc, dr, pairs = sp.get("accepted", 0), sp.get("drafted", 0), sp.get("pairs", 0)
    steps = max(1, g - acc)
    rows.append(f"{g / ms * 1e3:5.2f}")
    for k, v in (("gen", g), ("ms", ms), ("acc", acc), ("dr", dr), ("pairs", pairs)):
        tot[k] += v
steps = max(1, tot["gen"] - tot["acc"])
accr = tot["acc"] / tot["dr"] if tot["dr"] else 0.0
print(f"  {label:<18} {wl:<8} tok/s {' '.join(rows)} | {tot['gen'] / tot['ms'] * 1e3:5.2f} overall | "
      f"ms/step {tot['ms'] / steps:6.2f} | tok/step {tot['gen'] / steps:4.2f} | "
      f"drafts {tot['acc']}/{tot['dr']} accepted ({accr:.1%}) over {tot['pairs']} verify steps", flush=True)
