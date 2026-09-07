#!/usr/bin/env python3
"""터미널에서 하루 루틴을 기록하고 바로 push 합니다.

예시:
  python scripts/log.py --arrive 9:10 english coding
  python scripts/log.py --arrive 8:50 --leave 18:30 --mood 4 --sleep 7 english coding paper -n "DQN 리뷰 마무리"
  python scripts/log.py --date 2026-09-06 english --no-push
"""
import argparse, json, os, re, subprocess, sys
from datetime import date

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DAYS = os.path.join(ROOT, "_data", "days")


def hhmm(s):
    """'9:10', '09:10', '0910', '9시 10분' → '09:10'. 빈 값은 None."""
    s = (s or "").strip()
    if not s: return None
    m = re.fullmatch(r"(\d{1,2})\s*[:시]?\s*(\d{2})\s*분?", s)
    if not m: raise ValueError(f"시간 형식은 H:MM 또는 HHMM — 받은 값: {s}")
    h, mi = int(m.group(1)), int(m.group(2))
    if h > 23 or mi > 59: raise ValueError(f"시간 범위 오류: {s}")
    return f"{h:02d}:{mi:02d}"


def habits():
    """_config.yml 의 tracker.habits 를 [{key,label,emoji}] 로 (PyYAML 없이)."""
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
    """['english', '코딩', '📄'] 처럼 key/한글 라벨/이모지가 섞여 와도 key 리스트로. 모르는 값은 ValueError."""
    hs = habits(); keys = []
    for t in tokens:
        t = t.strip().strip(",")
        if not t: continue
        hit = next((h for h in hs if t.lower() == h["key"] or t == h["label"] or t == h["emoji"]
                    or (h["label"] and (t in h["label"] or h["label"].startswith(t)))), None)
        if not hit: raise ValueError(f"알 수 없는 항목: {t} (가능: {', '.join(h['key'] for h in hs)})")
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


def write_entry(day, content):
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", day): raise ValueError(f"날짜 형식 오류: {day}")
    os.makedirs(DAYS, exist_ok=True)
    path = os.path.join(DAYS, f"{day}.yml")
    existed = os.path.exists(path)
    with open(path, "w", encoding="utf-8") as f: f.write(content)
    return path, existed


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("done", nargs="*", help=f"한 것들 (가능: {', '.join(h['key'] for h in habits())})")
    p.add_argument("--date", default=date.today().isoformat())
    p.add_argument("--arrive"); p.add_argument("--leave"); p.add_argument("--wake")
    p.add_argument("--sleep", type=float); p.add_argument("--mood", type=int, choices=range(1, 6)); p.add_argument("--focus", type=float)
    p.add_argument("-n", "--note", default="")
    p.add_argument("--no-push", action="store_true", help="파일만 쓰고 commit/push 안 함")
    a = p.parse_args()
    try:
        content = build_yaml(hhmm(a.arrive), hhmm(a.leave), hhmm(a.wake), a.sleep, a.mood, a.focus, parse_done(a.done), a.note)
        path, existed = write_entry(a.date, content)
    except ValueError as e:
        sys.exit(f"오류: {e}")
    print(f"{'수정' if existed else '생성'}: {os.path.relpath(path, ROOT)}\n{content}")
    if not a.no_push:
        subprocess.run(["git", "-C", ROOT, "add", path], check=True)
        subprocess.run(["git", "-C", ROOT, "commit", "-q", "-m", f"log: {a.date}"], check=True)
        subprocess.run(["git", "-C", ROOT, "push", "-q"], check=True)
        print("push 완료 — 1~2분 뒤 사이트에 반영됩니다.")


if __name__ == "__main__":
    main()
