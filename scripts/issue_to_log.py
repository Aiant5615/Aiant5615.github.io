#!/usr/bin/env python3
"""Convert a GitHub Issue form body (.github/ISSUE_TEMPLATE/log.yml) into _data/days/YYYY-MM-DD.yml.
Runs in GitHub Actions: reads ISSUE_BODY / ISSUE_TITLE and writes ok/date/error to GITHUB_OUTPUT.
"""
import os, re, sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from log import build_yaml, hhmm, merge_entry, parse_done, write_entry  # noqa: E402

KST = timezone(timedelta(hours=9))
# Issue-form label → field key (English labels first; Korean kept so older issues still parse)
LABELS = {"date": "date", "arrived": "arrive", "left": "leave", "done": "done", "note": "note",
          "woke": "wake", "sleep": "sleep", "mood": "mood", "focus": "focus", "replace": "replace",
          "날짜": "date", "출근": "arrive", "퇴근": "leave", "오늘 한 것": "done", "메모": "note",
          "기상": "wake", "수면": "sleep", "컨디션": "mood", "집중": "focus"}


def parse_body(body):
    """Turn '### Label\\n\\nvalue' blocks into {field: value}. '_No response_' means empty."""
    fields = {}
    for m in re.finditer(r"^###\s+(.+?)\s*\n(.*?)(?=^###\s|\Z)", body.replace("\r\n", "\n"), re.S | re.M):
        label, val = m.group(1).strip(), m.group(2).strip()
        if val == "_No response_": val = ""
        key = next((k for lbl, k in LABELS.items() if label.lower().startswith(lbl)), None)
        if key: fields[key] = val
    return fields


def out(**kv):
    path = os.environ.get("GITHUB_OUTPUT")
    with (open(path, "a", encoding="utf-8") if path else sys.stdout) as f:
        for k, v in kv.items():
            v = str(v).replace("\n", " ")
            f.write(f"{k}={v}\n")


def main():
    body = os.environ.get("ISSUE_BODY", "")
    title = os.environ.get("ISSUE_TITLE", "")
    f = parse_body(body)
    try:
        day = f.get("date", "").strip()
        if not day:
            m = re.search(r"\d{4}-\d{2}-\d{2}", title)
            day = m.group(0) if m else datetime.now(KST).date().isoformat()
        day = day.replace(".", "-").replace("/", "-")
        m = re.fullmatch(r"(\d{4})-(\d{1,2})-(\d{1,2})", day)
        if not m: raise ValueError(f"Bad date format: {day}")
        day = f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
        num = lambda k: (f.get(k) or "").strip() or None
        kw = dict(arrive=hhmm(f.get("arrive")), leave=hhmm(f.get("leave")), wake=hhmm(f.get("wake")),
                  sleep=num("sleep"), mood=num("mood"), focus=num("focus"),
                  done=parse_done(re.split(r"[,\s/]+", f.get("done", ""))), note=f.get("note", ""))
        replace = "[x]" in (f.get("replace") or "").lower()
        content = build_yaml(**kw) if replace else merge_entry(day, **kw)
        path, existed = write_entry(day, content)
    except ValueError as e:
        print(f"::error::{e}"); out(ok="false", error=str(e)); return
    print(f"{'Updated' if existed else 'Created'}: {path}\n{content}")
    out(ok="true", date=day, error="")


if __name__ == "__main__":
    main()
