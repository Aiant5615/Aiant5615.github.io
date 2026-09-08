#!/usr/bin/env python3
"""Reduce the private days.json to a public, low-detail summary for the home-page heatmap.
Per day only: n = number of habits checked, c = whether the coding habit was checked. No times, notes, mood, sleep.
"""
import json, re, sys
from datetime import datetime, timedelta, timezone

src, dst = sys.argv[1], sys.argv[2]
days = json.load(open(src, encoding="utf-8")) or {}
out = {}
for k, v in days.items():
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", k) or not isinstance(v, dict): continue
    done = [str(x) for x in (v.get("done") or [])]
    out[k] = {"n": len(set(done)), "c": "coding" in done}
try:
    old = json.load(open(dst, encoding="utf-8"))
except Exception:
    old = {}
if old.get("days") == out:
    print("unchanged"); sys.exit(0)
json.dump({"updated": datetime.now(timezone(timedelta(hours=9))).isoformat(timespec="minutes"), "days": dict(sorted(out.items()))},
          open(dst, "w", encoding="utf-8"), indent=1)
print(f"wrote {len(out)} days")
