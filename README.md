# Seungwon's Research Log

A GitHub Pages site for tracking grad school life. https://aiant5615.github.io

The blog, papers, LeetCode, and about pages are public. **The tracker is private**: its data lives in the private repo
[Aiant5615/tracker](https://github.com/Aiant5615/tracker) (`days.json`) and the tracker page only shows anything in a browser where
the owner has connected a GitHub token.

## Logging a day

1. **Connect once per device**: Tracker page → "Private · not connected" → paste a fine-grained personal access token with
   *Repository access: only `Aiant5615/tracker`* and *Permissions: Contents → Read and write*. It is kept in that browser's
   localStorage only.
2. **Quick buttons** on the tracker (arrival/departure open a clock; habit buttons toggle) save one item at a time and merge it
   into the day. The **full form** saves several fields at once or another day; tick *Replace the whole entry* to start a day over.
3. **Terminal**: `python scripts/log.py --arrive 9:10 english coding -n "note"` (uses `gh` auth; `--remove`, `--replace`, `--show`).

Day format (`days.json`, keyed by date):

```json
{ "2026-09-07": { "arrive": "09:10", "leave": "18:30", "wake": "07:30", "sleep": 7, "mood": 4, "focus": 3,
                  "done": ["english", "coding", "paper"], "note": "one-line retro" } }
```

Times are 24-hour `HH:MM` in the data and shown as 12-hour AM/PM on the page. The `coding` habit is also checked automatically
on days with a new LeetCode solution (derived from the public `_data/leetcode.json`, nothing is written).

## LeetCode

Solutions live in a separate repo, [Aiant5615/leetcode](https://github.com/Aiant5615/leetcode), one folder per problem
(`0001-two-sum/0001-two-sum.{c,cpp,py}`). The [LeetHub](https://github.com/raphaelheinz/LeetHub-3.0) extension pushes every
accepted submission there. `.github/workflows/sync-leetcode.yml` pulls that repo hourly (or on demand from the Actions tab),
writes `_data/leetcode.json`. The `/leetcode/` page
shows stats, a daily heatmap, and the code for each language.

## Layout

| Path | Purpose |
|---|---|
| `_data/goals.yml` | Weekly goals. The entry whose `week` matches the current ISO week (`2026-W37`) shows on the home page |
| `_posts/` | Blog posts. `categories: [paper]` marks a paper review; `tags: [RL]` powers the field filter |
| `_drafts/` | Not built. Contains two post templates |
| `_config.yml` | Site info, `tracker_repo` (private data), `tracker.habits`, `arrive_goal`, `skip_weekends` |
| `.github/workflows/sync-leetcode.yml` | Hourly: leetcode repo → `_data/leetcode.json` + `coding` habit |
| `scripts/sync_leetcode.py` | The converter used by that workflow |
| `scripts/log.py` | Terminal logger (edits the private `days.json` via `gh api`) |

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
