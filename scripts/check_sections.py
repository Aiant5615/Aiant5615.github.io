#!/usr/bin/env python3
"""Validate code/<slug>/sections.json: every review heading id exists in the post and every code name exists in the file.
sections.json maps a review heading id (the ## text as a kramdown anchor) to top-level names in the implementation:
a function or class name, or "[banner text]" for a whole '# ─── banner ───' region. Run: python3 scripts/check_sections.py"""
import glob, json, os, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def heading_id(t):
    t = re.sub(r"[*_`$\\]", "", t); t = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", t)
    t = re.sub(r"[^\w\s-]", "", t.lower()); t = re.sub(r"^[^a-z]+", "", t.strip())
    return re.sub(r"\s+", "-", t.strip())


bad = 0
for f in sorted(glob.glob(os.path.join(ROOT, "code", "*", "sections.json"))):
    slug = os.path.basename(os.path.dirname(f))
    src = open(glob.glob(os.path.join(ROOT, "code", slug, "*.py"))[0], encoding="utf-8").read()
    md = open(glob.glob(os.path.join(ROOT, "_posts", f"*-{slug}.md"))[0], encoding="utf-8").read()
    names = set(re.findall(r"^(?:def|class) ([A-Za-z_0-9]+)", src, re.M)) | {"[" + l.strip("# ─\n ").strip() + "]" for l in src.split("\n") if l.startswith("# ───")}
    heads = {heading_id(l[3:]) for l in md.split("\n") if l.startswith("## ")}
    for h, ns in json.load(open(f, encoding="utf-8")).items():
        if h not in heads: print(f"{slug}: heading '{h}' not in the review"); bad += 1
        for n in ns:
            if n not in names: print(f"{slug}: '{n}' not in the code"); bad += 1
print(f"checked {len(glob.glob(os.path.join(ROOT, 'code', '*', 'sections.json')))} maps, problems: {bad}")
sys.exit(1 if bad else 0)
