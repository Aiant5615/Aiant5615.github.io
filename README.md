# Seungwon Kook — research site

A GitHub Pages site for tracking grad school life. https://aiant5615.github.io

The blog, papers, LeetCode, and about pages are public. **The tracker is private**: its data lives in the private repo
[Aiant5615/tracker](https://github.com/Aiant5615/tracker) (`days.json`) and the tracker page only shows anything in a browser where
the owner has connected a GitHub token.

## Public heatmap on the home page

`.github/workflows/sync-heatmap.yml` runs hourly, reads the private `days.json` with the repository secret **TRACKER_TOKEN**
(any token with Contents read on `Aiant5615/tracker`), and writes `_data/heatmap.json` holding only, per day, how many habits
were checked. The home page draws its heatmap from that file plus the public LeetCode data, so visitors see activity levels
but never times, notes or moods. Set the secret under Settings → Secrets and variables → Actions.

## Logging a day

1. **Set up each device once**: Tracker page → "Private · set up this device" → paste a fine-grained personal access token
   (*Repository access: only `Aiant5615/tracker`*, *Permissions: Contents → Read and write*) and choose a password. The token is
   stored in that browser only, encrypted with the password (PBKDF2 + AES-GCM in `localStorage`); the decrypted token lives in
   `sessionStorage` for the open tab. Each new tab or browser start asks for the password; "Lock" clears the session,
   "Forget this device" deletes the encrypted token.
2. **One tap** on the tracker's today bar: the arrival and leave chips open a clock, habit chips toggle, the mood chip picks
   1–5; each saves that one item into the day at once. The **log form** below shows the whole stored day (any date up to
   today) and saves it exactly as shown: edit a field, clear it, or un-tick a habit, then press Save; *Delete this day*
   removes the entry. A save that fails (offline, expired token) is kept in the browser and retried on the next load. English and Coding show a weekday streak; every habit tile has a gauge of this week's target (5 weekdays, or `weekly_goal`).
3. **Terminal**: `python scripts/log.py --arrive 9:10 english coding` (uses `gh` auth; `--remove`, `--replace`, `--show`).

Day format (`days.json`, keyed by date):

```json
{ "2026-09-07": { "arrive": "09:10", "leave": "18:30", "wake": "07:30", "sleep": 7, "mood": 4,
                  "done": ["english", "coding", "paper"] } }
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
| `_posts/` | Blog posts. `categories: [paper, <tab>]` marks a review and its tab on /papers/ (llm-basics, llm-rl, vision-llm, vla, llm-engineering, diffusion-rl); `series`/`series_order` place it in a reading path defined in `_data/series.yml` |
| `_drafts/` | Not built. Contains two post templates |
| `assets/lib/` | Self-hosted KaTeX 0.16.11 and highlight.js 11.9.0 (no CDN scripts; a CSP in `_includes/head.html` allows scripts from this site only) |
| `_config.yml` | Site info, `tracker_repo` (private data), `tracker.habits`, `arrive_goal`, `skip_weekends` |
| `.github/workflows/sync-leetcode.yml` | Hourly: leetcode repo → `_data/leetcode.json` + `coding` habit |
| `scripts/sync_leetcode.py` | The converter used by that workflow |
| `scripts/log.py` | Terminal logger (edits the private `days.json` via `gh api`) |

## Tracker settings

- **Habits**: edit `tracker.habits` in `_config.yml`. Add `weekends: true` to a habit to count weekends in its streak. Add `weekly_goal: N` to show "n / N this week" (Mon–Sun) instead of a streak; paper reading is 5/week and exercise 3/week.
- **Weekends**: with `skip_weekends: true`, Saturday and Sunday neither break nor count toward streaks.
- **Arrival goal**: `arrive_goal`.

## Writing

Posts live in `_posts/YYYY-MM-DD-slug.md`. Add `math: true` to the front matter for KaTeX. See `_posts/2026-09-07-dqn-atari.md` for the paper review format.

## Local preview (optional)

The system Ruby 4.0 is incompatible with the `github-pages` gem. Install Ruby 3.3 via rbenv, then:

```bash
bundle install && bundle exec jekyll serve --drafts
```
