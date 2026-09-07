# 승원의 연구 일지

대학원 생활 기록용 GitHub Pages 사이트. https://aiant5615.github.io

> ⚠️ 공개 사이트입니다. `_data/days/`의 출근 시간·메모·컨디션 등은 저장소와 페이지에 그대로 노출되니, 남에게 보여도 괜찮은 내용만 적으세요.

## 기록하는 법

| 방법 | 언제 |
|---|---|
| **휴대폰** — [📝 오늘 기록 Issue 폼](https://github.com/Aiant5615/Aiant5615.github.io/issues/new?template=log.yml)을 홈 화면에 추가해 두고 Submit | 가장 빠름. Action이 파일을 만들고 이슈를 닫습니다 |
| **사이트** — `/tracker/` 폼 → "GitHub에 저장" | 내용이 채워진 Issue 폼이 열리고 Submit만 누르면 됨 |
| **터미널** — `python scripts/log.py --arrive 9:10 english coding -n "메모"` | 노트북 앞에 있을 때 |
| **직접** — `_data/days/2026-09-07.yml` 작성 후 push | 여러 날 한꺼번에 고칠 때 |

같은 날짜를 다시 보내면 덮어씁니다. Issue 폼은 저장소 소유자가 만든 이슈만 처리합니다.

```yaml
# _data/days/2026-09-07.yml
arrive: "09:10"      # 반드시 따옴표 (YAML 이 09:10 을 숫자로 읽는 것 방지)
leave: "18:30"       # 자정 넘기면 다음날로 계산
wake: "07:30"
sleep: 7
mood: 4              # 1~5
focus: 3             # 집중 시간(h)
done: [english, coding, paper]
note: "한 줄 회고"
```

## 구조

| 경로 | 역할 |
|---|---|
| `_data/days/YYYY-MM-DD.yml` | 하루 기록 |
| `_data/goals.yml` | 주간 목표. `week: 2026-W37`처럼 ISO 주차가 현재 주와 같은 항목이 홈에 표시 |
| `_data/reading_list.yml` | 읽을 논문 큐. `link`가 리뷰 글의 `paper.link`와 같으면 자동 연결 |
| `_posts/` | 블로그 글. `categories: [paper]`면 논문 리뷰, `tags: [RL]`로 분야 필터 |
| `_drafts/` | 빌드되지 않는 초안. 글 템플릿 2개가 있음 |
| `_config.yml` | 사이트 정보, `tracker.habits`(습관 목록), `arrive_goal`, `skip_weekends` |
| `.github/ISSUE_TEMPLATE/log.yml` | 휴대폰용 기록 폼 |
| `.github/workflows/log-from-issue.yml` | 이슈 → YAML 파일 → 커밋 → Pages 빌드 → 이슈 닫기 |
| `scripts/log.py`, `scripts/issue_to_log.py` | 터미널 기록 / Action 변환 스크립트 |

## 트래커 설정

- **습관 바꾸기**: `_config.yml`의 `tracker.habits`. `weekends: true`를 붙이면 그 습관은 주말도 스트릭에 포함.
- **주말 처리**: `skip_weekends: true`면 토·일은 스트릭을 끊지도 세지도 않음.
- **목표 출근 시간**: `arrive_goal`.

## 글 쓰기

`_posts/YYYY-MM-DD-slug.md`. 수식은 front matter에 `math: true`. 논문 리뷰는 `_posts/2026-09-07-dqn-atari.md` 형식 참고.

## 로컬 미리보기 (선택)

시스템 Ruby 4.0은 `github-pages` gem과 호환되지 않습니다. rbenv로 Ruby 3.3을 설치한 뒤:

```bash
bundle install && bundle exec jekyll serve --drafts
```
