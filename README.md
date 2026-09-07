# Seungwon's Research Log

A GitHub Pages site for tracking grad school life. https://aiant5615.github.io

> ⚠️ This site is public. Everything in `_data/days/` (arrival times, notes, mood) is visible in the repo and on the site. Only log what you're comfortable sharing.

## Logging a day

| Method | When |
|---|---|
| **Phone** — add the [📝 Log today issue form](https://github.com/Aiant5615/Aiant5615.github.io/issues/new?template=log.yml) to your home screen and Submit | Fastest. An Action writes the file and closes the issue |
| **Site** — the form at `/tracker/` → "Save to GitHub" | Opens the same issue form, prefilled |
| **Terminal** — `python scripts/log.py --arrive 9:10 english coding -n "note"` | When you're at the laptop |
| **By hand** — write `_data/days/2026-09-07.yml` and push | Fixing several days at once |

Submitting the same date again overwrites it. The issue form only accepts issues opened by the repository owner; anyone else's are closed automatically.

```yaml
# _data/days/2026-09-07.yml
arrive: "09:10"      # quote times, or YAML reads 09:10 as an integer
leave: "18:30"       # past midnight is handled
wake: "07:30"
sleep: 7
mood: 4              # 1–5
focus: 3             # hours of deep work
done: [english, coding, paper]
note: "one-line retro"
```

## LeetCode

Solutions live in a separate repo, [Aiant5615/leetcode](https://github.com/Aiant5615/leetcode), one folder per problem
(`0001-two-sum/0001-two-sum.{c,cpp,py}`). The [LeetHub](https://github.com/raphaelheinz/LeetHub-3.0) extension pushes every
accepted submission there. `.github/workflows/sync-leetcode.yml` pulls that repo hourly (or on demand from the Actions tab),
writes `_data/leetcode.json`, and checks the `coding` habit for each day a solution was first committed. The `/leetcode/` page
shows stats, a daily heatmap, and the code for each language.

## Layout

| Path | Purpose |
|---|---|
| `_data/days/YYYY-MM-DD.yml` | One file per day |
| `_data/goals.yml` | Weekly goals. The entry whose `week` matches the current ISO week (`2026-W37`) shows on the home page |
| `_data/reading_list.yml` | Reading queue. Links to a review automatically when `link` matches the review's `paper.link` |
| `_posts/` | Blog posts. `categories: [paper]` marks a paper review; `tags: [RL]` powers the field filter |
| `_drafts/` | Not built. Contains two post templates |
| `_config.yml` | Site info, `tracker.habits`, `arrive_goal`, `skip_weekends` |
| `.github/ISSUE_TEMPLATE/log.yml` | The phone logging form |
| `.github/workflows/log-from-issue.yml` | Issue → YAML → commit → close issue |
| `.github/workflows/sync-leetcode.yml` | Hourly: leetcode repo → `_data/leetcode.json` + `coding` habit |
| `scripts/sync_leetcode.py` | The converter used by that workflow |
| `scripts/log.py`, `scripts/issue_to_log.py` | Terminal logger / Action converter |

## Tracker settings

- **Habits**: edit `tracker.habits` in `_config.yml`. Add `weekends: true` to a habit to count weekends in its streak.
- **Weekends**: with `skip_weekends: true`, Saturday and Sunday neither break nor count toward streaks.
- **Arrival goal**: `arrive_goal`.

## Writing

Posts live in `_posts/YYYY-MM-DD-slug.md`. Add `math: true` to the front matter for KaTeX. See `_posts/2026-09-07-dqn-atari.md` for the paper review format.

## Local preview (optional)

The system Ruby 4.0 is incompatible with the `github-pages` gem. Install Ruby 3.3 via rbenv, then:

```bash
bundle install && bundle exec jekyll serve --drafts
```
