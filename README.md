# 승원의 연구 일지

대학원 생활 기록용 GitHub Pages 사이트. https://aiant5615.github.io

## 구조

| 경로 | 역할 |
|---|---|
| `_data/days/YYYY-MM-DD.yml` | 하루 기록 (출근/퇴근, 습관 체크, 수면, 컨디션, 메모) |
| `_data/goals.yml` | 주간 목표 (홈에 표시) |
| `_data/reading_list.yml` | 읽을 논문 큐 (논문 리뷰 페이지에 표시) |
| `_posts/` | 블로그 글. `categories: [paper]` 면 논문 리뷰로 분류 |
| `_config.yml` | 사이트 정보 + 트래커 습관 목록/목표 출근 시간 |
| `scripts/log.py` | 터미널에서 하루 기록 + push |

## 기록하는 법

1. **사이트에서**: `/tracker/` 의 폼을 채우고 "GitHub에 저장" → GitHub 에서 Commit.
2. **터미널에서**: `python scripts/log.py --arrive 9:10 english coding -n "메모"`
3. **직접**: `_data/days/2026-09-07.yml` 파일을 만들어 push.

```yaml
date: "2026-09-07"
arrive: "09:10"      # 반드시 따옴표 (YAML 이 09:10 을 숫자로 읽는 것 방지)
leave: "18:30"
wake: "07:30"
sleep: 7
mood: 4              # 1~5
focus: 3             # 집중 시간(h)
done: [english, coding, paper]
note: "한 줄 회고"
```

## 습관 항목 바꾸기

`_config.yml` 의 `tracker.habits` 를 수정하면 폼·히트맵·통계에 바로 반영됩니다.

## 로컬 미리보기 (선택)

```bash
bundle install
bundle exec jekyll serve
```
