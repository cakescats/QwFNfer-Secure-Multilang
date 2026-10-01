#!/usr/bin/env python3
"""Warm decode measurement against a running qwfn-server.

usage: decab.py PORT LABEL [N_WARM] [N_MEAS] [N_TOK]
Sends N_WARM + N_MEAS identical greedy requests (so every arm decodes the same tokens and routes to the same
experts), then prints per-token numbers from the /stats `decode` block over the measured requests only:
wall ms/token, graph_a, host launch and device wait per replayed layer graph, expert io, and the rest.
"""
import json, sys, time, urllib.request

port, label = sys.argv[1], sys.argv[2]
n_warm = int(sys.argv[3]) if len(sys.argv) > 3 else 2
n_meas = int(sys.argv[4]) if len(sys.argv) > 4 else 3
n_tok = int(sys.argv[5]) if len(sys.argv) > 5 else 256
base = f"http://127.0.0.1:{port}"
body = {"model": "x", "max_tokens": n_tok, "temperature": 0, "reasoning_effort": "off", "ignore_eos": True,
        "messages": [{"role": "user", "content": "Write a detailed technical essay about memory hierarchies, "
                                                 "swapping, GPUs and mixture-of-experts inference. Be specific and long."}]}


def stats():
    with urllib.request.urlopen(base + "/stats", timeout=10) as r:
        return json.load(r)


def one():
    req = urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        d = json.load(r)
    t = d.get("timings", {})
    return t.get("predicted_n", 0), t.get("predicted_per_second", 0.0), d["choices"][0]["message"].get("content") or ""


for _ in range(n_warm):
    one()
s0 = stats()
rates, texts = [], []
for _ in range(n_meas):
    n, tps, txt = one()
    rates.append(tps)
    texts.append(txt)
s1 = stats()
a, b = s0["decode"], s1["decode"]
d = {k: b[k] - a[k] for k in b if isinstance(b[k], (int, float))}
steps = d["n_replay"] / 48.0 if d["n_replay"] else 1
ms = lambda k: 1e3 * d[k] / steps
per = lambda k: 1e6 * d[k] / d["n_replay"] if d["n_replay"] else 0
rest = ms("t_decode") - ms("t_graph_a") - ms("t_io")
print(f"  {label:<14} tok/s {' '.join(f'{r:5.2f}' for r in rates)} | decode {ms('t_decode'):5.2f} ms/tok = graphA {ms('t_graph_a'):5.2f}"
      f" (launch {per('t_replay_launch'):4.0f} + wait {per('t_replay_wait'):4.0f} us/graph) + io {ms('t_io'):4.2f} + rest {rest:4.2f}"
      f" | cpu-exp/tok {d['n_exp_cpu'] / steps:4.2f} misses/tok {d['cache_misses'] / steps:4.2f}")
print(f"  {label:<14} text[0:80]: {texts[-1][:80]!r} | same-across-runs: {len(set(texts)) == 1}")
