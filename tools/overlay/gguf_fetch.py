#!/usr/bin/env python3
"""Assemble a GGUF from selected tensors of a published quant, downloading only those tensors' bytes.

    gguf_fetch.py [--jobs N] VARIANT OUT.gguf PATTERN [PATTERN ...]

VARIANT is a directory in the HF repo (e.g. UD-Q2_K_XL); PATTERNs are regexes matched against tensor names.
The output carries no metadata keys: qwfn reads those from the head shard and takes the first tensor of a
name across shards, so this file is an overlay that wins over the originals. A megabyte of zero slack is
appended because qwfn's prefill rounds expert ranges up to the I/O alignment.
"""
import argparse, concurrent.futures, json, os, re, struct, sys, threading, time, urllib.request

REPO = "unsloth/Qwen3.8-Flash-Next-GGUF"
BASE = "https://huggingface.co/%s/resolve/main/" % REPO
ALIGN = 32
CHUNK = 8 << 20

# type id -> (block size in weights, bytes per block)
TYPE = {0: (1, 4), 1: (1, 2), 2: (32, 18), 3: (32, 20), 6: (32, 22), 7: (32, 24), 8: (32, 34),
        10: (256, 84), 11: (256, 110), 12: (256, 144), 13: (256, 176), 14: (256, 210),
        16: (256, 66), 17: (256, 74), 18: (256, 98), 19: (256, 50), 20: (32, 18), 21: (256, 110),
        22: (256, 82), 23: (256, 136), 29: (256, 56), 30: (1, 2), 39: (32, 17)}
NAME = {8: "Q8_0", 7: "Q5_1", 13: "Q5_K", 14: "Q6_K", 12: "Q4_K", 20: "IQ4_NL", 18: "IQ3_XXS", 17: "IQ2_XS", 3: "Q4_1", 22: "IQ2_S", 21: "IQ3_S"}


def get(url, start, end, retries=6):
    """Bytes [start, end] inclusive, with retries."""
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"Range": "bytes=%d-%d" % (start, end), "User-Agent": "qwfn-xpu"})
            with urllib.request.urlopen(req, timeout=120) as r:
                if r.status != 206:
                    raise RuntimeError("expected 206, got %s" % r.status)
                return r.read()
        except Exception as e:
            if attempt == retries - 1:
                raise
            time.sleep(2 * (attempt + 1))


class Reader:
    def __init__(self, buf):
        self.b, self.i = buf, 0

    def take(self, n):
        if self.i + n > len(self.b):
            raise EOFError
        v = self.b[self.i:self.i + n]
        self.i += n
        return v

    def u(self, fmt):
        return struct.unpack("<" + fmt, self.take(struct.calcsize("<" + fmt)))[0]

    def s(self):
        return self.take(self.u("Q")).decode("utf-8", "replace")


KV = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}


def header(buf):
    """(tensors, data_offset) where tensors is [(name, dims, type_id, offset)]."""
    r = Reader(buf)
    if r.take(4) != b"GGUF":
        raise RuntimeError("not a GGUF")
    r.u("I")
    nt, nkv = r.u("Q"), r.u("Q")
    align = ALIGN

    def val(t):
        if t == 8:
            return r.s()
        if t == 9:
            et, n = r.u("I"), r.u("Q")
            return [val(et) for _ in range(n)]
        return r.u(KV[t])

    for _ in range(nkv):
        k = r.s()
        v = val(r.u("I"))
        if k == "general.alignment":
            align = v
    ts = []
    for _ in range(nt):
        name = r.s()
        nd = r.u("I")
        dims = [r.u("Q") for _ in range(nd)]
        ty = r.u("I")
        off = r.u("Q")
        ts.append((name, dims, ty, off))
    return ts, (r.i + align - 1) // align * align


def tensor_bytes(dims, ty):
    n = 1
    for d in dims:
        n *= d
    blck, size = TYPE[ty]
    if n % blck:
        raise RuntimeError("shape %s not a multiple of the block size for type %d" % (dims, ty))
    return n // blck * size


def gguf_string(s):
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("variant")
    ap.add_argument("out")
    ap.add_argument("patterns", nargs="+")
    ap.add_argument("--jobs", type=int, default=1, help="parallel range requests (per-connection rates are throttled)")
    args = ap.parse_args()
    pats = [re.compile(p) for p in args.patterns]

    tree = json.load(urllib.request.urlopen("https://huggingface.co/api/models/%s/tree/main?recursive=true" % REPO, timeout=60))
    shards = sorted(e["path"] for e in tree if e.get("type") == "file" and e.get("path", "").startswith(args.variant + "/"))
    if not shards:
        sys.exit("no files under %s" % args.variant)

    picked = []   # (url, abs_offset, nbytes, name, dims, ty)
    for path in shards:
        url = BASE + path
        n = 1 << 20
        for _ in range(5):
            try:
                ts, data_off = header(get(url, 0, n - 1))
                break
            except EOFError:
                n *= 8
        else:
            sys.exit("header of %s is larger than %d bytes" % (path, n))
        for name, dims, ty, off in ts:
            if any(p.search(name) for p in pats):
                picked.append((url, data_off + off, tensor_bytes(dims, ty), name, dims, ty))
    if not picked:
        sys.exit("no tensors matched")
    picked.sort(key=lambda t: t[3])

    total = sum(p[2] for p in picked)
    print("%s: %d tensors, %.2f GB to download" % (args.variant, len(picked), total / 1e9))
    for _, _, nb, name, dims, ty in picked[:4]:
        print("   %-40s %-8s %s %.1f MB" % (name, NAME.get(ty, ty), dims, nb / 1e6))

    # header: no kv, offsets padded to ALIGN
    infos, off = b"", 0
    for _, _, nb, name, dims, ty in picked:
        infos += gguf_string(name) + struct.pack("<I", len(dims)) + b"".join(struct.pack("<Q", d) for d in dims)
        infos += struct.pack("<I", ty) + struct.pack("<Q", off)
        off += (nb + ALIGN - 1) // ALIGN * ALIGN
    head = b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", len(picked)) + struct.pack("<Q", 0) + infos
    pad = (-len(head)) % ALIGN

    data0 = len(head) + pad
    if args.jobs <= 1:
        done = 0
        t0 = time.time()
        with open(args.out, "wb") as f:
            f.write(head + b"\0" * pad)
            for url, start, nb, name, dims, ty in picked:
                got = 0
                while got < nb:
                    n = min(CHUNK, nb - got)
                    buf = get(url, start + got, start + got + n - 1)
                    if len(buf) != n:
                        sys.exit("short read on %s at %d" % (name, got))
                    f.write(buf)
                    got += n
                f.write(b"\0" * ((-nb) % ALIGN))
                done += nb
                print("   %-40s %6.1f MB   %5.1f%% at %.0f MB/s" % (name, nb / 1e6, 100.0 * done / total,
                      done / 1e6 / max(time.time() - t0, 1e-9)), flush=True)
            f.write(b"\0" * (1 << 20))   # tail slack for the prefill reader
    else:
        # Same layout, written by offset: header, each tensor padded to ALIGN, 1 MiB of tail slack.
        jobs, off = [], data0
        for url, start, nb, name, dims, ty in picked:
            for got in range(0, nb, CHUNK):
                jobs.append((url, start + got, min(CHUNK, nb - got), off + got, name))
            off += (nb + ALIGN - 1) // ALIGN * ALIGN
        with open(args.out, "wb") as f:
            f.write(head + b"\0" * pad)
            f.truncate(off + (1 << 20))
        fd = os.open(args.out, os.O_WRONLY)
        lock, state = threading.Lock(), {"done": 0, "next": 0.0}
        t0 = time.time()
        def one(j):
            url, src, n, dst, name = j
            buf = get(url, src, src + n - 1)
            if len(buf) != n:
                raise RuntimeError("short read on %s at %d" % (name, src))
            os.pwrite(fd, buf, dst)
            with lock:
                state["done"] += n
                pct = 100.0 * state["done"] / total
                if pct >= state["next"]:
                    state["next"] = pct + 2.0
                    print("   %5.1f%% at %.1f MB/s" % (pct, state["done"] / 1e6 / max(time.time() - t0, 1e-9)), flush=True)
        with concurrent.futures.ThreadPoolExecutor(args.jobs) as ex:
            for fut in [ex.submit(one, j) for j in jobs]:
                fut.result()
        os.fsync(fd); os.close(fd)
    print("wrote %s: %.2f GB + 1 MiB slack" % (args.out, os.path.getsize(args.out) / 1e9))


if __name__ == "__main__":
    main()
