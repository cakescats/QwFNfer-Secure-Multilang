#!/usr/bin/env python3
"""Long-prompt fixtures for tools/perf (ctxdec.py, pcswitch.py) and token files for the NLL replay.

  make_fixtures.py json OUT_DIR [SIZES_K ...]          ctx{N}k.json: a chat request whose user turn is ~N thousand
                                                        tokens of public-domain text plus a question (default 20 40 60 80 100)
  make_fixtures.py text OUT.txt [CHARS]                the raw text (default ~150K chars, ~40K tokens) for
                                                        `qwfn-tok <head shard> --raw OUT.txt > p40k.ids` (the NLL replay)

The text is Project Gutenberg eBook #8300 (The Bible, Douay-Rheims), fetched once and cached next to the output;
--source FILE_OR_URL uses another. Sizes are set by characters at ~3.55 characters per token (this tokenizer on this
text); the server reports the exact count. The research fixtures were cut from the Old Testament part of the same
translation, so counts differ slightly from the numbers in the docs.
"""
import argparse, json, os, urllib.request

SRC = "https://www.gutenberg.org/cache/epub/8300/pg8300.txt"
CHARS_PER_TOKEN = 3.55
QUESTION = "\n\nSummarise the main themes above in about 150 words."


def load(source, cache_dir):
    if os.path.exists(source):
        return open(source, encoding="utf-8").read()
    path = os.path.join(cache_dir, os.path.basename(source))
    if not os.path.exists(path):
        with urllib.request.urlopen(source, timeout=120) as r, open(path, "wb") as f:
            f.write(r.read())
    return open(path, encoding="utf-8").read()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["json", "text"])
    ap.add_argument("out")
    ap.add_argument("sizes", nargs="*", type=int)
    ap.add_argument("--source", default=SRC)
    a = ap.parse_args()
    out_dir = a.out if a.mode == "json" else (os.path.dirname(os.path.abspath(a.out)) or ".")
    os.makedirs(out_dir, exist_ok=True)
    text = load(a.source, out_dir)
    if a.mode == "text":
        n = a.sizes[0] if a.sizes else 150000
        open(a.out, "w", encoding="utf-8").write(text[:n])
        print(f"{a.out}: {min(n, len(text))} chars")
        return
    for k in a.sizes or [20, 40, 60, 80, 100]:
        n = int(k * 1000 * CHARS_PER_TOKEN)
        if n > len(text):
            raise SystemExit(f"{k}K needs {n} chars; the source has {len(text)}")
        body = {"model": "x", "messages": [{"role": "user", "content": text[:n] + QUESTION}],
                "max_tokens": 256, "temperature": 0, "reasoning_effort": "off"}
        path = os.path.join(a.out, f"ctx{k}k.json")
        json.dump(body, open(path, "w"))
        print(f"{path}: {n} chars (~{k}K tokens)")


if __name__ == "__main__":
    main()
