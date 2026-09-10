#!/usr/bin/env python3
"""Read a LeetHub-style solutions repo and write _data/leetcode.json.
(The tracker page derives the `coding` habit from this file; nothing else is written.)

Env: LEETCODE_DIR (path to a full clone of the solutions repo). Prints a summary; exit 0 always.
"""
import json, os, re, subprocess, sys
from datetime import datetime, timedelta, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "_data", "leetcode.json")
KST = timezone(timedelta(hours=9))
EXT = {".c": "c", ".cpp": "cpp", ".cc": "cpp", ".cxx": "cpp", ".py": "py"}
SKIP_DIRS = {".git", ".github", "node_modules"}


def _log(repo, path, *extra):
    try:
        return subprocess.run(["git", "-C", repo, "log", *extra, "--format=%aI", "--", path],
                              capture_output=True, text=True, check=True).stdout.strip().splitlines()
    except subprocess.CalledProcessError:
        return []


def _kst(iso):
    return datetime.fromisoformat(iso.replace("Z", "+00:00")).astimezone(KST).date().isoformat()


def commit_dates(repo, path):
    """(first, all): KST date of the most recent commit that added `path` (a file deleted and re-added counts from
    the re-add), and every commit date touching it since then — each accepted re-submission is one more commit."""
    added = _log(repo, path, "--diff-filter=A", "--follow")
    if not added: return None, []
    first_iso = added[0]; first = _kst(first_iso)
    every = sorted({_kst(d) for d in _log(repo, path) if d >= first_iso})
    return first, every


def parse_readme(path):
    """LeetHub README: <h2><a href="https://leetcode.com/problems/two-sum/">1. Two Sum</a></h2><h3>Easy</h3>"""
    meta = {}
    try:
        s = open(path, encoding="utf-8", errors="replace").read(4000)
    except OSError:
        return meta
    m = re.search(r'href="(https?://leetcode\.com/problems/([^/"]+)/?)"[^>]*>\s*(?:(\d+)\.\s*)?([^<]+)<', s)
    if m:
        meta.update(url=m.group(1), slug=m.group(2), title=m.group(4).strip())
        if m.group(3): meta["id"] = int(m.group(3))
    m = re.search(r"<h3>\s*(Easy|Medium|Hard)\s*</h3>", s, re.I)
    if m: meta["difficulty"] = m.group(1).capitalize()
    return meta


def scan(repo):
    problems = []
    for name in sorted(os.listdir(repo)):
        d = os.path.join(repo, name)
        if not os.path.isdir(d) or name in SKIP_DIRS or name.startswith("."): continue
        langs = {}
        for f in sorted(os.listdir(d)):
            lang = EXT.get(os.path.splitext(f)[1].lower())
            if not lang or lang in langs: continue
            date, every = commit_dates(repo, os.path.join(name, f))
            langs[lang] = {"file": f, "date": date, "days": every}   # days: first solve plus every later re-submission
        if not langs: continue
        m = re.match(r"^(\d+)[-_.]?(.*)$", name)
        slug = (m.group(2) if m else name).strip("-_ ") or name
        p = {"id": int(m.group(1)) if m and m.group(1) else None, "slug": slug,
             "title": slug.replace("-", " ").replace("_", " ").title(), "difficulty": None,
             "url": f"https://leetcode.com/problems/{slug}/", "path": name, "langs": langs}
        p.update({k: v for k, v in parse_readme(os.path.join(d, "README.md")).items() if v})
        dates = [l["date"] for l in langs.values() if l["date"]]
        alld = sorted({d for l in langs.values() for d in l["days"]})
        p["first"] = min(dates) if dates else None
        p["last"] = alld[-1] if alld else None
        problems.append(p)
    problems.sort(key=lambda p: (p["last"] or "", p["id"] or 0), reverse=True)
    return problems


def main():
    repo = os.environ.get("LEETCODE_DIR")
    if not repo or not os.path.isdir(repo):
        print("LEETCODE_DIR missing; nothing to do"); return
    problems = scan(repo)
    data = {"updated": datetime.now(KST).isoformat(timespec="minutes"), "problems": problems}
    old = open(OUT, encoding="utf-8").read() if os.path.exists(OUT) else ""
    new = json.dumps(data, ensure_ascii=False, indent=1) + "\n"
    changed_json = re.sub(r'"updated": "[^"]*"', "", old) != re.sub(r'"updated": "[^"]*"', "", new)
    if changed_json:
        open(OUT, "w", encoding="utf-8").write(new)
    print(f"problems: {len(problems)} · json {'updated' if changed_json else 'unchanged'}")


if __name__ == "__main__":
    main()
