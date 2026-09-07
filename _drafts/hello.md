---
title: 사이트를 열며
categories: [diary]
---

대학원 생활을 꾸준히 기록하려고 사이트를 만들었습니다. 매일 출근 시간과 공부 루틴을 [트래커](/tracker/)에 남기고, 읽은 논문은 [논문 리뷰](/papers/)에 정리할 계획입니다.

## 글 쓰는 법

`_posts/YYYY-MM-DD-slug.md` 파일을 만들면 글이 됩니다.

```markdown
---
title: 글 제목
categories: [diary]      # diary, study, weekly, paper 등 자유롭게
---
본문은 마크다운으로.
```

수식이 필요하면 front matter에 `math: true`를 추가하면 `$...$`, `$$...$$`가 KaTeX로 렌더링됩니다.
