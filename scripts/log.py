#!/usr/bin/env python3
"""Log a day from the terminal into the private tracker repo (days.json) via `gh api`.

Examples:
  python scripts/log.py --arrive 9:10 english coding
  python scripts/log.py --arrive 8:50 --leave 18:30 --mood 4 --sleep 7 english coding paper -n "Finished the DQN review"
  python scripts/log.py --date 2026-09-06 --remove exercise
  python scripts/log.py --replace --arrive 9:00 english     # overwrite the whole day

Fields are merged into the existing day (habits added, notes appended, times replaced). Needs `gh auth login`.
"""
import argparse, base64, json, os, re, subprocess, sys
from datetime import date

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def cfg(key, default=None):
    m = re.search(rf"^{key}:\s*\"?([^\"#\n]+?)\"?\s*(#.*)?$", open(os.path.join(ROOT, "_config.yml"), encoding="utf-8").read(), re.M)
    return m.group(1).strip() if m else default


REPO, BRANCH, FILE = cfg("tracker_repo"), cfg("tracker_branch", "main"), cfg("tracker_file", "days.json")


def hhmm(s):
    """'9:10', '09:10', '0910', '9:10 AM', '2pm' → 'HH:MM'. Empty → None."""
    s = (s or "").strip()
    if not s: return None
    m = re.fullmatch(r"(\d{1,2})\s*(?::\s*(\d{2})|(\d{2}))?\s*([ap])\.?m?\.?", s, re.I) or re.fullmatch(r"(\d{1,2})\s*:?\s*(\d{2})()()", s)
    if not m: raise ValueError(f"Time must be like 9:10, 0910, or 9:10 AM — got: {s}")
    h, mi, ap = int(m.group(1)), int(m.group(2) or m.group(3) or 0), (m.group(4) or "").lower()
    if ap:
        if h < 1 or h > 12: raise ValueError(f"Time out of range: {s}")
        h = h % 12 + (12 if ap == "p" else 0)
    if h > 23 or mi > 59: raise ValueError(f"Time out of range: {s}")
    return f"{h:02d}:{mi:02d}"


def habits():
    out, cur, in_habits = [], None, False
    for line in open(os.path.join(ROOT, "_config.yml"), encoding="utf-8"):
        if re.match(r"\s*habits:", line): in_habits = True; continue
        if not in_habits: continue
        if line.strip() and not line.startswith(" "): break
        m = re.match(r"\s*-\s*key:\s*(\S+)", line)
        if m: cur = {"key": m.group(1), "label": "", "emoji": ""}; out.append(cur); continue
        m = re.match(r"\s*(label|emoji):\s*\"?([^\"#\n]*?)\"?\s*(#.*)?$", line)
        if m and cur: cur[m.group(1)] = m.group(2).strip()
    return out


def parse_done(tokens):
    hs, keys = habits(), []
    for t in tokens:
        t = t.strip().strip(",")
        if not t: continue
        hit = next((h for h in hs if t.lower() == h["key"] or t == h["label"] or t == h["emoji"] or (h["label"] and h["label"].lower().startswith(t.lower()))), None)
        if not hit: raise ValueError(f"Unknown habit: {t} (valid: {', '.join(h['key'] for h in hs)})")
        if hit["key"] not in keys: keys.append(hit["key"])
    return keys


def gh_api(path, method="GET", body=None):
    cmd = ["gh", "api", "-X", method, "-H", "Accept: application/vnd.github+json", path]
    if body is not None: cmd += ["--input", "-"]
    r = subprocess.run(cmd, input=json.dumps(body) if body is not None else None, capture_output=True, text=True)
    if r.returncode != 0:
        if "404" in r.stderr or "Not Found" in r.stdout: return None
        sys.exit(f"gh api failed: {r.stderr.strip() or r.stdout.strip()}")
    return json.loads(r.stdout) if r.stdout.strip() else {}


def merge(cur, f, replace):
    cur = {} if replace else dict(cur or {})
    for k in ("arrive", "leave", "wake"):
        if f.get(k): cur[k] = f[k]
    for k in ("sleep", "mood", "focus"):
        if f.get(k) is not None: cur[k] = f[k]
    done = list(cur.get("done", []))
    for d in f.get("done", []):
        if d not in done: done.append(d)
    for d in f.get("remove", []):
        done = [x for x in done if x != d]
    cur["done"] = done
    n = (f.get("note") or "").strip()
    if n and n not in (cur.get("note") or ""): cur["note"] = f"{cur['note']} · {n}" if cur.get("note") else n
    if not cur.get("note"): cur.pop("note", None)
    return cur


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("done", nargs="*", help=f"habits done (valid: {', '.join(h['key'] for h in habits())})")
    p.add_argument("--date", default=date.today().isoformat())
    p.add_argument("--arrive"); p.add_argument("--leave"); p.add_argument("--wake")
    p.add_argument("--sleep", type=float); p.add_argument("--mood", type=int, choices=range(1, 6)); p.add_argument("--focus", type=float)
    p.add_argument("-n", "--note", default="")
    p.add_argument("--remove", action="append", default=[], help="un-check a habit")
    p.add_argument("--replace", action="store_true", help="overwrite the whole day instead of merging")
    p.add_argument("--show", action="store_true", help="print the day and exit")
    a = p.parse_args()
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", a.date): sys.exit(f"Bad date: {a.date}")
    cur = gh_api(f"repos/{REPO}/contents/{FILE}?ref={BRANCH}")
    days = json.loads(base64.b64decode(cur["content"]).decode("utf-8")) if cur else {}
    if a.show:
        print(json.dumps(days.get(a.date, {}), ensure_ascii=False, indent=1)); return
    try:
        f = dict(arrive=hhmm(a.arrive), leave=hhmm(a.leave), wake=hhmm(a.wake), sleep=a.sleep, mood=a.mood, focus=a.focus,
                 done=parse_done(a.done), remove=parse_done(a.remove), note=a.note)
    except ValueError as e:
        sys.exit(f"Error: {e}")
    days[a.date] = merge(days.get(a.date), f, a.replace)
    days = dict(sorted(days.items()))
    body = {"message": f"log: {a.date}", "branch": BRANCH,
            "content": base64.b64encode((json.dumps(days, ensure_ascii=False, indent=1) + "\n").encode("utf-8")).decode("ascii")}
    if cur: body["sha"] = cur["sha"]
    gh_api(f"repos/{REPO}/contents/{FILE}", "PUT", body)
    print(f"Saved {a.date} → {REPO}/{FILE}\n{json.dumps(days[a.date], ensure_ascii=False, indent=1)}")


if __name__ == "__main__":
    main()
