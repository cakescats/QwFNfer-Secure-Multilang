#!/usr/bin/env python3
"""Two alternating conversations against one server: pcswitch.py PORT LABEL FIXTURE
A and B share the fixture's long document but open differently, so they diverge at the first tokens.
Turns: A1, B1, A2 (A1 + its reply + a new question), B2, A3. With --prefix-cache, A2/B2/A3 should restore
their conversation and prefill only the new turn; without it, each switch re-prefills the whole history.
Prints per turn: prompt tokens, cached tokens, prompt seconds, and the server's prefix_cache counters."""
import json, sys, time, urllib.request

port, label, fx = sys.argv[1], sys.argv[2], sys.argv[3]
base = f"http://127.0.0.1:{port}"
doc = json.load(open(fx))["messages"][0]["content"]
conv = {"A": [{"role": "user", "content": "Conversation A. Read this and answer briefly.\n" + doc + "\n\nWho is speaking in the first paragraph?"}],
        "B": [{"role": "user", "content": "Conversation B, a different reader. Read this.\n" + doc + "\n\nName one place mentioned."}]}
followups = {"A": ["And in the last paragraph?", "Summarize it in one line."], "B": ["What happens next?"]}


def post(msgs):
    body = {"model": "x", "messages": msgs, "max_tokens": 48, "temperature": 0, "reasoning_effort": "off"}
    req = urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        d = json.load(r)
    return d, time.time() - t0


for turn in ["A", "B", "A", "B", "A"]:
    msgs = conv[turn]
    d, wall = post(msgs)
    msg = d["choices"][0]["message"]
    msgs.append({"role": "assistant", "content": msg.get("content") or ""})
    if followups[turn]:
        msgs.append({"role": "user", "content": followups[turn].pop(0)})
    t = d.get("timings", {})
    with urllib.request.urlopen(base + "/stats", timeout=10) as r:
        pc = json.load(r).get("prefix_cache", {})
    print(f"  {label} {turn}: prompt {d['usage']['prompt_tokens']:6d} tok, cached {t.get('prompt_cached_n', 0):6d}, "
          f"prompt {t.get('prompt_ms', 0) / 1e3:6.1f} s (wall {wall:5.1f} s) | pool hits {pc.get('hits', '-')} "
          f"misses {pc.get('misses', '-')} saves {pc.get('saves', '-')} restored {pc.get('tokens_restored', '-')}", flush=True)
