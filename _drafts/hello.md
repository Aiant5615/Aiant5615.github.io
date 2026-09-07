---
title: Opening the site
categories: [diary]
---

I built this site to keep a steady record of grad school life: daily arrival times and study routines go into the [tracker](/tracker/), and papers I read get written up under [Papers](/papers/).

## How to write a post

Create `_posts/YYYY-MM-DD-slug.md`:

```markdown
---
title: Post title
categories: [diary]      # diary, study, weekly, paper — anything you like
---
Body in Markdown.
```

Add `math: true` to the front matter to render `$...$` and `$$...$$` with KaTeX.
