#!/usr/bin/env python3
"""GitHub Issue 폼(.github/ISSUE_TEMPLATE/log.yml) 본문을 _data/days/YYYY-MM-DD.yml 로 변환.
GitHub Actions 에서 실행되며 ISSUE_BODY / ISSUE_TITLE 환경변수를 읽고 GITHUB_OUTPUT 에 ok/date/error 를 씁니다.
"""
import os, re, sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from log import build_yaml, hhmm, parse_done, write_entry  # noqa: E402

KST = timezone(timedelta(hours=9))
# 이슈 폼의 label → 필드 키
LABELS = {"날짜": "date", "출근": "arrive", "퇴근": "leave", "오늘 한 것": "done", "메모": "note",
          "기상": "wake", "수면": "sleep", "컨디션": "mood", "집중": "focus"}


def parse_body(body):
    """'### 라벨\\n\\n값' 블록들을 {필드: 값} 으로. '_No response_' 는 빈 값."""
    fields = {}
    for m in re.finditer(r"^###\s+(.+?)\s*\n(.*?)(?=^###\s|\Z)", body.replace("\r\n", "\n"), re.S | re.M):
        label, val = m.group(1).strip(), m.group(2).strip()
        if val == "_No response_": val = ""
        key = next((k for lbl, k in LABELS.items() if label.startswith(lbl)), None)
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
        if not m: raise ValueError(f"날짜 형식 오류: {day}")
        day = f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
        num = lambda k: (f.get(k) or "").strip() or None
        content = build_yaml(
            arrive=hhmm(f.get("arrive")), leave=hhmm(f.get("leave")), wake=hhmm(f.get("wake")),
            sleep=num("sleep"), mood=num("mood"), focus=num("focus"),
            done=parse_done(re.split(r"[,\s/]+", f.get("done", ""))), note=f.get("note", ""))
        path, existed = write_entry(day, content)
    except ValueError as e:
        print(f"::error::{e}"); out(ok="false", error=str(e)); return
    print(f"{'수정' if existed else '생성'}: {path}\n{content}")
    out(ok="true", date=day, error="")


if __name__ == "__main__":
    main()
