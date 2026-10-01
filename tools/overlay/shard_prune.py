#!/usr/bin/env python3
"""Rewrite a GGUF shard keeping only the tensors an overlay head resolves to it (qwfn: the first tensor of a name
across the head's parts wins). KV metadata is copied byte for byte except
split.tensors.count, which drops by the number of tensors removed; kept tensors keep their order and alignment.
  tools/overlay/shard_prune.py OVERLAY_DIR SHARD OUT   writes OUT, then checks every kept tensor's bytes against SHARD.
With the v3 overlay, 14 GB of the stock first shard is shadowed; replacing the shard with OUT frees it (v1/v2 heads
then stop working: they read tensors OUT no longer has). With v4, 15.3 GB: the stock BF16 hyper-connection mixers too,
so v1-v3 heads stop working; prune against v3 to keep a v3 rollback (v4 runs on either)."""
import hashlib, os, struct, sys

def rd_str(f): n, = struct.unpack("<Q", f.read(8)); return f.read(n)
def skip_val(f, t):
    fixed = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}
    if t in fixed: f.read(fixed[t])
    elif t == 8: rd_str(f)
    elif t == 9:
        at, n = struct.unpack("<IQ", f.read(12))
        for _ in range(n): skip_val(f, at)
    else: raise ValueError(f"kv type {t}")

def parse(p):
    with open(p, "rb") as f:
        magic, ver = f.read(4), f.read(4); assert magic == b"GGUF"
        nt, nkv = struct.unpack("<QQ", f.read(16)); align = 32
        kv0 = f.tell(); cnt = None
        for _ in range(nkv):
            k = rd_str(f); t, = struct.unpack("<I", f.read(4))
            if k == b"split.tensors.count": cnt = (f.tell() - kv0, t)
            if k == b"general.alignment": align, = struct.unpack("<I", f.read(4))
            else: skip_val(f, t)
        kv1 = f.tell(); f.seek(kv0); kvraw = f.read(kv1 - kv0)
        ti = []
        for _ in range(nt):
            name = rd_str(f); nd, = struct.unpack("<I", f.read(4)); dims = f.read(8 * nd)
            t, off = struct.unpack("<IQ", f.read(12)); ti.append([name, nd, dims, t, off])
        start = (f.tell() + align - 1) // align * align
    size = os.path.getsize(p)
    offs = sorted(x[4] for x in ti) + [size - start]
    nxt = {o: offs[i + 1] for i, o in enumerate(offs[:-1])}
    for x in ti: x.append(nxt[x[4]] - x[4])          # span in the file, padding included
    return dict(ver=ver, nkv=nkv, kvraw=kvraw, cnt=cnt, align=align, start=start, ti=ti)

def main():
    odir, shard, out = sys.argv[1:4]
    shard = os.path.realpath(shard)
    seen, keep = set(), set()
    for fn in sorted(os.listdir(odir)):
        p = os.path.realpath(os.path.join(odir, fn))
        for x in parse(p)["ti"]:
            if x[0] in seen: continue
            seen.add(x[0])
            if p == shard: keep.add(x[0])
    s = parse(shard); A = s["align"]
    kept = sorted((x for x in s["ti"] if x[0] in keep), key=lambda x: x[4])
    # new offsets: kept tensors packed in their original order, each at an aligned offset
    # split.tensors.count is the total across all shards; llama.cpp's loader (the tokenizer load) checks it
    kvraw = bytearray(s["kvraw"])
    if s["cnt"]:
        at, t = s["cnt"]; fmt = {2: "<H", 3: "<h", 4: "<I", 5: "<i", 10: "<Q", 11: "<q"}[t]
        old, = struct.unpack_from(fmt, kvraw, at)
        struct.pack_into(fmt, kvraw, at, old - (len(s["ti"]) - len(kept)))
    hdr = bytearray(b"GGUF" + s["ver"] + struct.pack("<QQ", len(kept), s["nkv"]) + kvraw)
    off, new = 0, []
    for name, nd, dims, t, o, span in kept:
        new.append((name, o, off, span))
        hdr += struct.pack("<Q", len(name)) + name + struct.pack("<I", nd) + dims + struct.pack("<IQ", t, off)
        off = (off + span + A - 1) // A * A
    start = (len(hdr) + A - 1) // A * A
    hdr += b"\0" * (start - len(hdr))
    with open(shard, "rb") as fi, open(out, "wb") as fo:
        fo.write(hdr)
        for name, o, no, span in new:
            fo.seek(start + no); fi.seek(s["start"] + o); left = span
            while left:
                b = fi.read(min(left, 64 << 20)); fo.write(b); left -= len(b)
        fo.truncate(start + off)
    # verify: the new file parses to the kept tensors, and each tensor's bytes match the original
    n = parse(out); assert [x[0] for x in n["ti"]] == [x[0] for x in kept], "tensor list differs"
    with open(shard, "rb") as fa, open(out, "rb") as fb:
        for (name, o, no, span), y in zip(new, n["ti"]):
            ha, hb = hashlib.blake2b(), hashlib.blake2b()
            fa.seek(s["start"] + o); fb.seek(n["start"] + y[4]); left = span
            while left:
                k = min(left, 64 << 20); ha.update(fa.read(k)); hb.update(fb.read(k)); left -= k
            assert ha.digest() == hb.digest(), f"bytes differ: {name}"
    dropped = [x for x in s["ti"] if x[0] not in keep]
    print(f"kept {len(kept)} tensors ({sum(x[5] for x in kept)/1e9:.2f} GB), dropped {len(dropped)} "
          f"({sum(x[5] for x in dropped)/1e9:.2f} GB); {out}: {os.path.getsize(out)/1e9:.2f} GB, all kept tensors byte-identical")
    kinds = {}
    for x in dropped:
        k = x[0].decode().split(".", 2)[-1] if x[0].startswith(b"blk.") else x[0].decode()
        kinds[k] = kinds.get(k, 0) + 1
    print("dropped by kind:", dict(sorted(kinds.items())))

main()
