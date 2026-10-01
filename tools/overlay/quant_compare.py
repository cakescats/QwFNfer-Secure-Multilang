#!/usr/bin/env python3
"""Per-tensor-role quant comparison: our effective model (overlay over GSQ-RCO, first tensor of a name wins,
in the head shard's split order) vs Unsloth's published tiers (headers read by HTTP range, nothing downloaded).
    quant_compare.py GGML_H OUR_HEAD_SHARD [TIER ...]      -> prints a role table + JSON to stdout
"""
import glob, json, os, re, sys, urllib.request
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gguf_fetch as gf

def ggml_types(ggml_h):
    src = open(ggml_h).read()
    body = src[src.index("enum ggml_type {"):]
    body = body[:body.index("};")]
    out = {}
    for name, val in re.findall(r"GGML_TYPE_(\w+)\s*=\s*(\d+)", body):
        out[int(val)] = name.lower()
    return out

def type_bytes(ggml_h):
    # bytes-per-weight from ggml-common block sizes is overkill here; use gf.TYPE where known, else None
    return None

def local_header(path):
    n = 1 << 20
    with open(path, "rb") as f:
        while True:
            f.seek(0); buf = f.read(n)
            try:
                return gf.header(buf)[0]
            except EOFError:
                n *= 8

def remote_tier(tier):
    tree = json.load(urllib.request.urlopen("https://huggingface.co/api/models/%s/tree/main?recursive=true" % gf.REPO, timeout=60))
    shards = sorted(e["path"] for e in tree if e.get("type") == "file" and e.get("path", "").startswith(tier + "/"))
    out = {}
    for path in shards:
        url = gf.BASE + path
        n = 1 << 20
        while True:
            try:
                ts, _ = gf.header(gf.get(url, 0, n - 1)); break
            except EOFError:
                n *= 8
        for name, dims, ty, off in ts:
            out.setdefault(name, (dims, ty))
    return out

def role(name):
    m = re.match(r"blk\.(\d+)\.(.+)", name)
    return (m.group(2), int(m.group(1))) if m else (name, -1)

def main():
    ggml_h, head = sys.argv[1], sys.argv[2]
    tiers = sys.argv[3:] or ["UD-Q2_K_XL", "UD-Q3_K_XL", "UD-Q4_K_XL"]
    tn = ggml_types(ggml_h)
    # our model: the head shard names its siblings by split count; qwfn loads them in file order
    d = os.path.dirname(os.path.realpath(head)) if False else os.path.dirname(head)
    base = re.sub(r"-00001-of-(\d+)\.gguf$", "", os.path.basename(head))
    count = int(re.search(r"-of-(\d+)\.gguf$", head).group(1))
    ours = {}
    for i in range(1, count + 1):
        p = os.path.join(d, "%s-%05d-of-%05d.gguf" % (base, i, count))
        for name, dims, ty, off in local_header(p):
            ours.setdefault(name, (dims, ty, i))
    remote = {t: remote_tier(t) for t in tiers}
    roles = {}
    for name, (dims, ty, shard) in ours.items():
        r, layer = role(name)
        n = 1
        for x in dims: n *= x
        roles.setdefault(r, []).append((layer, name, tn.get(ty, ty), n, shard,
                                        [tn.get(remote[t][name][1], "?") if name in remote[t] else "-" for t in tiers]))
    print("%-28s %6s  %-28s | %s" % ("role", "Mw", "ours (overlay shard#)", " | ".join("%-22s" % t for t in tiers)))
    for r in sorted(roles, key=lambda k: -sum(x[3] for x in roles[k])):
        rows = roles[r]
        def summ(vals):
            c = {}
            for v in vals: c[v] = c.get(v, 0) + 1
            return " ".join("%s x%d" % (k, v) if len(c) > 1 else str(k) for k, v in sorted(c.items(), key=lambda kv: -kv[1]))
        ours_s = summ(["%s@%d" % (x[2], x[4]) for x in rows])
        tier_s = [summ([x[5][i] for x in rows]) for i in range(len(tiers))]
        print("%-28s %6.0f  %-28s | %s" % (r, sum(x[3] for x in rows) / 1e6, ours_s[:28], " | ".join("%-22s" % t[:22] for t in tier_s)))
    json.dump({r: [[x[0], x[2], x[5]] for x in sorted(roles[r])] for r in roles}, open("/tmp/quant_compare.json", "w"))

main()
