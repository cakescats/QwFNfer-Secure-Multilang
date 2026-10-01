#!/usr/bin/env python3
"""Decode rate after a long prompt: ctxdec.py PORT LABEL FIXTURE [N_TOK]. Posts the fixture request with
max_tokens N_TOK, greedy, thinking off; prints prompt tokens, prefill tok/s and decode tok/s."""
import json, sys, urllib.request
port, label, fx = sys.argv[1], sys.argv[2], sys.argv[3]
n = int(sys.argv[4]) if len(sys.argv) > 4 else 256
body = json.load(open(fx))
body.update({"max_tokens": n, "temperature": 0, "reasoning_effort": "off", "stream": False})
req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=3600) as r:
    d = json.load(r)
t = d.get("timings", {})
print(f"  {label:<10} prompt {d.get('usage', {}).get('prompt_tokens')} tok @ {t.get('prompt_per_second', 0):6.1f} tok/s | decode {t.get('predicted_n')} tok @ {t.get('predicted_per_second', 0):5.2f} tok/s")
