#!/usr/bin/env python3
"""터미널에서 오늘 루틴을 기록하고 바로 push 합니다.

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
    m = re.fullmatch(r"(\d{1,2}):?(\d{2})", s.strip())
    if not m: raise argparse.ArgumentTypeError(f"시간 형식은 H:MM 또는 HHMM — 받은 값: {s}")
    return f"{int(m.group(1)):02d}:{m.group(2)}"

def habit_keys():
    cfg = os.path.join(ROOT, "_config.yml")
    keys, in_habits = [], False
    with open(cfg, encoding="utf-8") as f:
        for line in f:
            if re.match(r"\s*habits:", line): in_habits = True; continue
            if in_habits:
                m = re.match(r"\s*-\s*key:\s*(\S+)", line)
                if m: keys.append(m.group(1))
                elif line.strip() and not line.startswith(" "): break
    return keys

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("done", nargs="*", help=f"한 것들 (가능: {', '.join(habit_keys())})")
p.add_argument("--date", default=date.today().isoformat())
p.add_argument("--arrive", type=hhmm); p.add_argument("--leave", type=hhmm); p.add_argument("--wake", type=hhmm)
p.add_argument("--sleep", type=float); p.add_argument("--mood", type=int, choices=range(1, 6)); p.add_argument("--focus", type=float)
p.add_argument("-n", "--note", default="")
p.add_argument("--no-push", action="store_true", help="파일만 쓰고 commit/push 안 함")
a = p.parse_args()

valid = habit_keys()
bad = [d for d in a.done if d not in valid]
if bad: sys.exit(f"알 수 없는 항목: {bad} (가능: {valid})")

lines = [f'date: "{a.date}"']
for k in ("arrive", "leave", "wake"):
    if getattr(a, k): lines.append(f'{k}: "{getattr(a, k)}"')
for k in ("sleep", "mood", "focus"):
    v = getattr(a, k)
    if v is not None: lines.append(f"{k}: {v:g}" if isinstance(v, float) else f"{k}: {v}")
lines.append(f"done: [{', '.join(a.done)}]")
if a.note: lines.append(f"note: {json.dumps(a.note, ensure_ascii=False)}")
content = "\n".join(lines) + "\n"

os.makedirs(DAYS, exist_ok=True)
path = os.path.join(DAYS, f"{a.date}.yml")
existed = os.path.exists(path)
with open(path, "w", encoding="utf-8") as f: f.write(content)
print(f"{'수정' if existed else '생성'}: {os.path.relpath(path, ROOT)}\n{content}")

if not a.no_push:
    subprocess.run(["git", "-C", ROOT, "add", path], check=True)
    subprocess.run(["git", "-C", ROOT, "commit", "-q", "-m", f"log: {a.date}"], check=True)
    subprocess.run(["git", "-C", ROOT, "push", "-q"], check=True)
    print("push 완료 — 1~2분 뒤 사이트에 반영됩니다.")
