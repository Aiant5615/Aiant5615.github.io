#!/usr/bin/env python3
"""Reduce the private days.json to a public, low-detail summary for the home-page heatmap.
Per day only: n = number of habits checked, c = whether the coding habit was checked. No times, notes, mood, sleep.
"""
import json, re, sys
from datetime import datetime, timedelta, timezone

src, dst = sys.argv[1], sys.argv[2]
lc_src = sys.argv[3] if len(sys.argv) > 3 else None          # optional leetcode.json: a solution day counts as the coding habit
days = json.load(open(src, encoding="utf-8")) or {}
lc_days = set()
if lc_src:
    try:
        for p in (json.load(open(lc_src, encoding="utf-8")) or {}).get("problems", []):
            for l in (p.get("langs") or {}).values():
                lc_days.update(l.get("days") or ([l["date"]] if l.get("date") else []))
    except Exception as e:
        print(f"leetcode.json not used: {e}")
out = {}
for k in set(days) | lc_days:
    v = days.get(k) if isinstance(days.get(k), dict) else {}
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", k): continue
    done = set(str(x) for x in (v.get("done") or []))
    if k in lc_days: done.add("coding")
    out[k] = {"n": len(done), "c": "coding" in done}
try:
    old = json.load(open(dst, encoding="utf-8"))
except Exception:
    old = {}
if old.get("days") == out:
    print("unchanged"); sys.exit(0)
json.dump({"updated": datetime.now(timezone(timedelta(hours=9))).isoformat(timespec="minutes"), "days": dict(sorted(out.items()))},
          open(dst, "w", encoding="utf-8"), indent=1)
print(f"wrote {len(out)} days")
