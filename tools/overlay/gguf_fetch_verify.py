#!/usr/bin/env python3
"""Spot-check a gguf_fetch.py output against its source: random 1 MiB ranges of random tensors.

    gguf_fetch_verify.py VARIANT FILE.gguf [SAMPLES]
"""
import json, random, sys, urllib.request
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import gguf_fetch as gf

variant, path = sys.argv[1], sys.argv[2]
samples = int(sys.argv[3]) if len(sys.argv) > 3 else 12
with open(path, "rb") as f:
    local_ts, local_data = gf.header(f.read(64 << 20))
local = {name: (local_data + off, gf.tensor_bytes(dims, ty)) for name, dims, ty, off in local_ts}
tree = json.load(urllib.request.urlopen("https://huggingface.co/api/models/%s/tree/main?recursive=true" % gf.REPO, timeout=60))
remote = {}
for p in sorted(e["path"] for e in tree if e.get("type") == "file" and e["path"].startswith(variant + "/")):
    url = gf.BASE + p
    n = 1 << 20
    for _ in range(5):
        try:
            ts, data_off = gf.header(gf.get(url, 0, n - 1)); break
        except EOFError:
            n *= 8
    for name, dims, ty, off in ts:
        if name in local:
            remote[name] = (url, data_off + off, gf.tensor_bytes(dims, ty))
rng = random.Random(7)
names = sorted(local)
bad = 0
with open(path, "rb") as f:
    for _ in range(samples):
        name = rng.choice(names)
        loff, nb = local[name]
        url, roff, rnb = remote[name]
        assert nb == rnb, name
        k = rng.randrange(0, max(1, nb - (1 << 20)))
        n = min(1 << 20, nb - k)
        f.seek(loff + k); a = f.read(n)
        b = gf.get(url, roff + k, roff + k + n - 1)
        ok = a == b
        bad += not ok
        print("%-32s +%11d  %s" % (name, k, "match" if ok else "MISMATCH"))
    f.seek(0, 2)
    print("tensors %d (remote %d), file %.2f GB, mismatches %d of %d" % (len(local), len(remote), f.tell() / 1e9, bad, samples))
sys.exit(1 if bad else 0)
