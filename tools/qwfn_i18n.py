"""Translations for the qwfn console.

The console's code and page are written in English; a language is a catalog file,
tools/console/i18n/<lang>.json:

    {"lang": "ru", "name": "Русский",
     "html":     [["English fragment of index.html / login.html", "translation"], ...],
     "messages": [["printf-style English message, e.g. %d GB: %.1f GB available", "translation with the same placeholders"], ...]}

"html" pairs are applied to the page as it is served. "messages" are applied to every
string the console's API returns, after formatting: each pattern becomes a regular
expression whose placeholders capture the values, so "12 GB: 26.0 GB available" is
translated without the code that built it knowing about languages, and a log written
before the language was switched reads in the new one. Placeholders must appear in
the translation in the same order. Adding a language is adding a file; en.json is
empty because English is the source.
"""
import json, os, re

_PH = re.compile(r"%%|%[-+ #0]*\d*(?:\.\d+)?[sdif]")


def _compile(en, tr):
    """(regex, template) for one printf-style pair, or a plain literal when it has no placeholders."""
    parts, n, pos = [], 0, 0
    phs = [m for m in _PH.finditer(en) if m.group() != "%%"]
    for m in _PH.finditer(en):
        parts.append(re.escape(en[pos:m.start()]))
        tok = m.group()
        if tok == "%%": parts.append("%")
        else:
            n += 1
            if tok.endswith("s"):
                if m.start() == 0: parts.append(r"(\S+)")          # a leading %s is a token: a kv type, a model key, a path
                elif m.end() == len(en): parts.append(r"(.*)")      # a trailing %s runs to the end of the string
                else: parts.append(r"(.*?)")
            elif tok.endswith(("d", "i")): parts.append(r"(-?\d+)")
            else: parts.append(r"(-?\d+(?:\.\d+)?)")
        pos = m.end()
    parts.append(re.escape(en[pos:]))
    tr_phs = [m for m in _PH.finditer(tr) if m.group() != "%%"]
    if len(tr_phs) != len(phs):
        raise ValueError("placeholder count differs: %r -> %r" % (en, tr))
    # the template: every placeholder of the translation becomes the next captured group
    tpl, pos, k = [], 0, 0
    for m in _PH.finditer(tr):
        tpl.append(tr[pos:m.start()].replace("\\", "\\\\"))
        if m.group() == "%%": tpl.append("%")
        else: k += 1; tpl.append("\\g<%d>" % k)
        pos = m.end()
    tpl.append(tr[pos:].replace("\\", "\\\\"))
    pattern = "".join(parts)
    if n == 0 and len(en) < 25:
        pattern = "^" + pattern + "$"        # short literals ("Custom") translate only as a whole string
    anchor = max(re.split(r"%[-+ #0]*\d*(?:\.\d+)?[sdif%]", en), key=len)   # cheap pre-check before the regex
    return re.compile(pattern, re.S), "".join(tpl), anchor


class Catalogs:
    def __init__(self, directory):
        self.dir = directory
        self.cats = {}
        for f in sorted(os.listdir(directory)) if os.path.isdir(directory) else []:
            if not f.endswith(".json"): continue
            c = json.load(open(os.path.join(directory, f), encoding="utf-8"))
            code = c.get("lang") or f[:-5]
            msgs = sorted(c.get("messages", []), key=lambda p: -len(p[0]))   # outer messages before the ones nested in them
            c["_msgs"] = [_compile(en, tr) for en, tr in msgs]
            self.cats[code] = c
        if "en" not in self.cats:
            self.cats["en"] = {"lang": "en", "name": "English", "html": [], "_msgs": []}

    def languages(self):
        return [{"code": k, "name": v.get("name", k)} for k, v in self.cats.items()]

    def has(self, code):
        return code in self.cats

    def page(self, html, lang):
        """Translate an HTML page (the fragments are applied in catalog order)."""
        c = self.cats.get(lang)
        if not c: return html
        for en, tr in c.get("html", []):
            html = html.replace(en, tr)
        return html

    def text(self, s, lang):
        c = self.cats.get(lang)
        if not c or not c["_msgs"] or not s: return s
        for rx, tpl, anchor in c["_msgs"]:
            if anchor in s:
                s = rx.sub(tpl, s)
        return s

    def obj(self, o, lang, skip=()):
        """Translate every string inside a JSON-able object, except under the keys in `skip`."""
        if lang not in self.cats or not self.cats[lang]["_msgs"]: return o
        if isinstance(o, str): return self.text(o, lang)
        if isinstance(o, list): return [self.obj(x, lang, skip) for x in o]
        if isinstance(o, dict): return {k: (v if k in skip else self.obj(v, lang, skip)) for k, v in o.items()}
        return o

    def check(self, lang):
        """Round-trip every message pattern with sample values: returns the pairs that fail."""
        bad = []
        c = self.cats.get(lang, {})
        def sample(tok, i, first, last):
            if tok.endswith("s"): return "tok%d" % i if first else ("tail %d text" % i if last else "val %d x" % i)
            if tok.endswith(("d", "i")): return 10 + i
            return 1.5 + i
        for en, tr in c.get("messages", []):
            toks = [m for m in _PH.finditer(en) if m.group() != "%%"]
            vals = tuple(sample(m.group(), i, m.start() == 0, m.end() == len(en)) for i, m in enumerate(toks))
            try:
                src, want = (en % vals, tr % vals) if toks else (en, tr)
            except Exception as e:
                bad.append((en, "format: %s" % e)); continue
            got = self.text(src, lang)
            if got != want: bad.append((en, got))
        return bad
