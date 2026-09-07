#!/usr/bin/env python3
"""Log a day from the terminal and push it.

Examples:
  python scripts/log.py --arrive 9:10 english coding
  python scripts/log.py --arrive 8:50 --leave 18:30 --mood 4 --sleep 7 english coding paper -n "Finished the DQN review"
  python scripts/log.py --date 2026-09-06 english --no-push

Fields are merged into the existing day (habits added, notes appended, times replaced). Use --replace to overwrite.
"""
import argparse, json, os, re, subprocess, sys
from datetime import date

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DAYS = os.path.join(ROOT, "_data", "days")


def hhmm(s):
    """'9:10', '09:10', '0910' → '09:10'. Empty → None."""
    s = (s or "").strip()
    if not s: return None
    m = re.fullmatch(r"(\d{1,2})\s*[:시]?\s*(\d{2})\s*분?", s)
    if not m: raise ValueError(f"Time must be H:MM or HHMM, got: {s}")
    h, mi = int(m.group(1)), int(m.group(2))
    if h > 23 or mi > 59: raise ValueError(f"Time out of range: {s}")
    return f"{h:02d}:{mi:02d}"


def habits():
    """Read tracker.habits from _config.yml as [{key,label,emoji}] (no PyYAML needed)."""
    out, cur, in_habits = [], None, False
    with open(os.path.join(ROOT, "_config.yml"), encoding="utf-8") as f:
        for line in f:
            if re.match(r"\s*habits:", line): in_habits = True; continue
            if not in_habits: continue
            if line.strip() and not line.startswith(" "): break
            m = re.match(r"\s*-\s*key:\s*(\S+)", line)
            if m: cur = {"key": m.group(1), "label": "", "emoji": ""}; out.append(cur); continue
            m = re.match(r"\s*(label|emoji):\s*\"?([^\"#\n]*?)\"?\s*(#.*)?$", line)
            if m and cur: cur[m.group(1)] = m.group(2).strip()
    return out


def parse_done(tokens):
    """Map tokens (keys, labels, or emoji) to habit keys. Unknown tokens raise ValueError."""
    hs = habits(); keys = []
    for t in tokens:
        t = t.strip().strip(",")
        if not t: continue
        hit = next((h for h in hs if t.lower() == h["key"] or t == h["label"] or t == h["emoji"]
                    or (h["label"] and (t in h["label"] or h["label"].startswith(t)))), None)
        if not hit: raise ValueError(f"Unknown habit: {t} (valid: {', '.join(h['key'] for h in hs)})")
        if hit["key"] not in keys: keys.append(hit["key"])
    return keys


def build_yaml(arrive=None, leave=None, wake=None, sleep=None, mood=None, focus=None, done=(), note=""):
    lines = []
    for k, v in (("arrive", arrive), ("leave", leave), ("wake", wake)):
        if v: lines.append(f'{k}: "{v}"')
    for k, v in (("sleep", sleep), ("mood", mood), ("focus", focus)):
        if v is not None and v != "":
            v = float(v)
            lines.append(f"{k}: {int(v) if v.is_integer() else v}")
    lines.append(f"done: [{', '.join(done)}]")
    if note: lines.append(f"note: {json.dumps(note.strip(), ensure_ascii=False)}")
    return "\n".join(lines) + "\n"


def read_entry(path):
    """Minimal parser for the YAML we write: scalar fields, done: [a, b], note: "json string"."""
    e = {}
    if not os.path.exists(path): return e
    for line in open(path, encoding="utf-8"):
        m = re.match(r"^(\w+):\s*(.*?)\s*$", line)
        if not m: continue
        k, v = m.group(1), m.group(2)
        if k == "done":
            e["done"] = [x.strip() for x in v.strip("[]").split(",") if x.strip()]
        elif k == "note":
            try: e["note"] = json.loads(v) if v.startswith('"') else v
            except ValueError: e["note"] = v.strip('"')
        else:
            e[k] = v.strip('"')
    return e


def merge_entry(day, arrive=None, leave=None, wake=None, sleep=None, mood=None, focus=None, done=(), note=""):
    """Overlay the given fields on the existing day file. Habits are unioned, notes appended, other fields replaced when given."""
    old = read_entry(os.path.join(DAYS, f"{day}.yml"))
    f = {k: old.get(k) for k in ("arrive", "leave", "wake", "sleep", "mood", "focus")}
    for k, v in (("arrive", arrive), ("leave", leave), ("wake", wake), ("sleep", sleep), ("mood", mood), ("focus", focus)):
        if v not in (None, ""): f[k] = v
    dn = list(old.get("done", []))
    for d in done:
        if d not in dn: dn.append(d)
    nt = (old.get("note") or "").strip()
    note = (note or "").strip()
    if note and note not in nt: nt = f"{nt} · {note}" if nt else note
    return build_yaml(f["arrive"], f["leave"], f["wake"], f["sleep"], f["mood"], f["focus"], dn, nt)


def write_entry(day, content):
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", day): raise ValueError(f"Bad date format: {day}")
    os.makedirs(DAYS, exist_ok=True)
    path = os.path.join(DAYS, f"{day}.yml")
    existed = os.path.exists(path)
    with open(path, "w", encoding="utf-8") as f: f.write(content)
    return path, existed


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("done", nargs="*", help=f"habits done (valid: {', '.join(h['key'] for h in habits())})")
    p.add_argument("--date", default=date.today().isoformat())
    p.add_argument("--arrive"); p.add_argument("--leave"); p.add_argument("--wake")
    p.add_argument("--sleep", type=float); p.add_argument("--mood", type=int, choices=range(1, 6)); p.add_argument("--focus", type=float)
    p.add_argument("-n", "--note", default="")
    p.add_argument("--no-push", action="store_true", help="write the file only, no commit/push")
    p.add_argument("--replace", action="store_true", help="overwrite the day instead of merging into it")
    a = p.parse_args()
    try:
        args = (hhmm(a.arrive), hhmm(a.leave), hhmm(a.wake), a.sleep, a.mood, a.focus, parse_done(a.done), a.note)
        content = build_yaml(*args) if a.replace else merge_entry(a.date, *args)
        path, existed = write_entry(a.date, content)
    except ValueError as e:
        sys.exit(f"Error: {e}")
    print(f"{'Updated' if existed else 'Created'}: {os.path.relpath(path, ROOT)}\n{content}")
    if not a.no_push:
        subprocess.run(["git", "-C", ROOT, "add", path], check=True)
        subprocess.run(["git", "-C", ROOT, "commit", "-q", "-m", f"log: {a.date}"], check=True)
        subprocess.run(["git", "-C", ROOT, "push", "-q"], check=True)
        print("Pushed. The site updates in a minute or two.")


if __name__ == "__main__":
    main()
