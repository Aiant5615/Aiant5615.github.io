/* LeetCode practice game: retype one of your own accepted solutions line by line.
   easy   - the whole answer line is shown as grey ghost text under what you type
   medium - about half of the tokens of each line are shown, the rest are hidden
   hard   - nothing is shown
   Points per line: easy 1, medium 2, hard 3; every wrong try reveals one more hidden token and costs one point of that
   line; Skip reveals the line for 0 points. Best scores are kept in this browser (localStorage) only. */
(function () {
  "use strict";
  const V = window.GHVault, CFG = window.LEETCODE_CONFIG || {};
  const STORE = "leetcode_game";
  const PTS = { easy: 1, medium: 2, hard: 3 };
  const LANG_NAME = { c: "C", cpp: "C++", py: "Python" };
  const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const lsGet = k => { try { return JSON.parse(localStorage.getItem(k) || "null"); } catch { return null; } };
  const lsSet = (k, v) => { try { localStorage.setItem(k, JSON.stringify(v)); } catch {} };
  const bests = () => lsGet(STORE) || {};
  const bestKey = (slug, lang, mode) => `${slug}|${lang}|${mode}`;
  const norm = s => s.trim().replace(/\s+/g, " ");
  const hash = s => { let h = 2166136261; for (const c of s) { h ^= c.charCodeAt(0); h = Math.imul(h, 16777619) >>> 0; } return h; };

  // lines worth typing: drop blank lines and comment-only lines
  function playable(text, lang) {
    const out = [];
    let inBlock = false;
    text.replace(/\r/g, "").split("\n").forEach(line => {
      const t = line.trim();
      if (lang !== "py") {
        if (inBlock) { if (t.includes("*/")) inBlock = false; return; }
        if (t.startsWith("/*")) { if (!t.includes("*/")) inBlock = true; return; }
        if (t.startsWith("//")) return;
      } else if (t.startsWith("#")) return;
      if (t) out.push(line.replace(/\s+$/, ""));
    });
    return out;
  }
  const tokens = line => line.match(/\w+|[^\w\s]+|\s+/g) || [];
  const HLJS_LANG = { c: "c", cpp: "cpp", py: "python" };

  // syntax colour per character: highlight the line with highlight.js and record the class of each character
  function charClasses(line) {
    if (!window.hljs) return line.split("").map(() => "");
    let html;
    try { html = hljs.highlight(line, { language: HLJS_LANG[G.lang] || "plaintext", ignoreIllegals: true }).value; } catch { return line.split("").map(() => ""); }
    const tpl = document.createElement("template"); tpl.innerHTML = html;
    const out = [];
    const walk = (node, cls) => node.childNodes.forEach(n => {
      if (n.nodeType === 3) { for (const _ of n.textContent) out.push(cls); }
      else walk(n, (cls + " " + (n.className || "")).trim());
    });
    walk(tpl.content, "");
    return out.length === line.length ? out : line.split("").map(() => "");
  }
  const highlightLine = line => { try { return window.hljs ? hljs.highlight(line, { language: HLJS_LANG[G.lang] || "plaintext", ignoreIllegals: true }).value : esc(line); } catch { return esc(line); } };

  let G = null;   // the current game
  async function fetchFile(path) {
    const branch = CFG.branch || "main";
    if (V.token()) return V.ghRaw(CFG.repo, path, branch);
    const r = await fetch(`https://raw.githubusercontent.com/${CFG.repo}/${branch}/${path.split("/").map(encodeURIComponent).join("/")}`, { cache: "no-store" });
    if (!r.ok) { const e = new Error(`${r.status}`); e.status = r.status; throw e; }
    return r.text();
  }
  const DESC_LANG_KEY = "leetcode_desc_lang";
  const descLang = () => { try { return localStorage.getItem(DESC_LANG_KEY) === "ko" ? "ko" : "en"; } catch { return "en"; } };
  const $ = id => document.getElementById(id);

  // The problem statement from the folder's README.md (LeetHub writes it as HTML). Only a whitelist of tags survives.
  const SAFE = new Set(["P", "PRE", "CODE", "STRONG", "B", "EM", "I", "UL", "OL", "LI", "BR", "SUP", "SUB", "H1", "H2", "H3", "H4", "HR", "SPAN", "DIV", "TABLE", "THEAD", "TBODY", "TR", "TH", "TD", "IMG", "A"]);
  function sanitize(node, out) {
    node.childNodes.forEach(n => {
      if (n.nodeType === 3) { out.appendChild(document.createTextNode(n.textContent)); return; }
      if (n.nodeType !== 1 || !SAFE.has(n.tagName)) { if (n.nodeType === 1) sanitize(n, out); return; }
      const e = document.createElement(n.tagName.toLowerCase());
      if (n.tagName === "A" && /^https?:/.test(n.getAttribute("href") || "")) { e.href = n.getAttribute("href"); e.target = "_blank"; e.rel = "noopener"; }
      if (n.tagName === "IMG" && /^https:/.test(n.getAttribute("src") || "")) { e.src = n.getAttribute("src"); e.alt = n.getAttribute("alt") || ""; e.loading = "lazy"; }
      sanitize(n, e); out.appendChild(e);
    });
  }
  function descriptionHtml(readme) {
    const doc = new DOMParser().parseFromString(readme, "text/html");
    const first = doc.body.querySelector("h2"); if (first) first.remove();                 // the title line, shown in our own header
    const diff = doc.body.querySelector("h3"); if (diff && /^(easy|medium|hard)$/i.test(diff.textContent.trim())) diff.remove();
    const hr = doc.body.querySelector("hr"); if (hr && hr === doc.body.firstElementChild) hr.remove();
    const out = document.createElement("div"); sanitize(doc.body, out);
    return out.innerHTML.trim() || `<p class="muted">No description in this folder.</p>`;
  }
  const descBody = () => (G.desc && G.desc[G.descLang]) || `<p class="muted">${G.descLang === "ko" ? "설명을 불러오는 중…" : "Loading the description…"}</p>`;
  const descPane = () => `<aside class="game-desc"><div class="game-desc-head"><b>${esc(G.problem.id != null ? "#" + G.problem.id + " " : "")}${esc(G.problem.title)}</b>${G.problem.difficulty ? ` <span class="diff diff-${G.problem.difficulty.toLowerCase()}">${esc(G.problem.difficulty)}</span>` : ""}${G.problem.url ? ` <a class="small" href="${esc(G.problem.url)}" target="_blank" rel="noopener">LeetCode ↗</a>` : ""}<div class="seg game-desc-lang" style="margin-left:auto">${["en", "ko"].map(l => `<button type="button" data-dl="${l}" class="${l === G.descLang ? "on" : ""}">${l === "en" ? "En" : "Ko"}</button>`).join("")}</div></div><div class="game-desc-body">${descBody()}</div></aside>`;
  function wireDesc(root) {
    root.querySelectorAll(".game-desc-lang button").forEach(b => b.addEventListener("click", () => {
      G.descLang = b.dataset.dl; try { localStorage.setItem(DESC_LANG_KEY, G.descLang); } catch {}
      root.querySelectorAll(".game-desc-lang button").forEach(x => x.classList.toggle("on", x === b));
      root.querySelector(".game-desc-body").innerHTML = descBody();
      const input = $("game-input"); if (input) input.focus();
    }));
  }

  function lineMask(line, idx, mode) {
    // which tokens start visible: easy all, hard none, medium roughly half (deterministic per line)
    const tk = tokens(line);
    return tk.map((t, i) => /^\s+$/.test(t) ? true : mode === "easy" ? true : mode === "hard" ? false : (hash(`${idx}:${i}:${t}`) % 100) < 55);
  }

  function ghostHtml(line, mask) {
    const tk = tokens(line), cc = charClasses(line);
    let pos = 0, html = "";
    tk.forEach((t, i) => {
      // hard mode shows nothing for hidden tokens (not even their length); medium shows dots of the same length
      if (mask[i]) { for (const ch of t) { html += `<span class="pending ${cc[pos]}">${esc(ch)}</span>`; pos++; } }
      else { html += G.mode === "hard" ? "" : `<span class="ghost-hidden">${"•".repeat(t.length)}</span>`; pos += t.length; }
    });
    return html;
  }

  function render() {
    const root = $("lc-game"); if (!root || !G) return;
    const p = G.problem, done = G.idx >= G.lines.length;
    const best = bests()[bestKey(p.slug, G.lang, G.mode)];
    const head = `<div class="game-head">
        <div><b>${LANG_NAME[G.lang]}</b> <span class="muted">· ${G.mode} mode</span></div>
        <div class="game-meta"><span>line <b>${Math.min(G.idx + 1, G.lines.length)}</b>/${G.lines.length}</span><span>score <b id="game-score">${G.score}</b>/${G.max}</span>${best ? `<span class="muted">best ${best.best}/${best.max}</span>` : ""}<button type="button" class="btn btn-sm" id="game-quit">Quit</button></div>
      </div>
      <div class="game-track" title="${G.idx} of ${G.lines.length} lines done"><div class="road"></div><span class="flag">🏁</span><span class="runner" id="game-runner" style="left:${(G.prevPct || 0).toFixed(1)}%">${done ? "🎉" : "🏃"}</span></div>`;
    if (done) {
      const secs = Math.round((Date.now() - G.t0) / 1000), pct = Math.round(G.score / G.max * 100);
      root.innerHTML = `<div class="game-layout">` + descPane() + `<div class="game-main">` + head + `<div class="game-done">
          <div class="game-done-score">${G.score} / ${G.max} <span class="unit">points · ${pct}%</span></div>
          <div class="muted">${G.lines.length} lines · ${G.wrong} wrong tries · ${G.skipped} skipped · ${Math.floor(secs / 60)}m ${secs % 60}s${G.recorded ? (G.newBest ? " · 🏆 new best" : "") : " · not recorded"}</div>
          <div class="form-actions" style="margin-top:.8rem"><button type="button" class="btn btn-primary" id="game-again">Play again</button>
            ${["easy", "medium", "hard"].filter(m => m !== G.mode).map(m => `<button type="button" class="btn" data-mode="${m}">Try ${m}</button>`).join("")}
            <button type="button" class="btn" id="game-random">Random problem</button></div>
          <details class="more" style="margin-top:.8rem"><summary>Show the solution</summary><pre class="game-code"><code class="hljs">${G.lines.map(highlightLine).join("\n")}</code></pre></details></div></div></div>`;
      moveRunner(); wireDesc(root);
      root.querySelector("#game-again").addEventListener("click", () => start(G.problem, G.lang, G.mode));
      root.querySelectorAll("[data-mode]").forEach(b => b.addEventListener("click", () => start(G.problem, G.lang, b.dataset.mode)));
      root.querySelector("#game-random").addEventListener("click", () => window.LCGame.random());
      root.querySelector("#game-quit").addEventListener("click", quit);
      return;
    }
    const line = G.lines[G.idx], indent = line.match(/^\s*/)[0];
    const doneLines = G.lines.slice(Math.max(0, G.idx - 6), G.idx).map(l => `<div class="game-prev hljs">${highlightLine(l)}</div>`).join("");
    const left = G.lines.length - G.idx - 1;
    const upcoming = G.mode === "hard" ? "" : G.lines.slice(G.idx + 1, G.idx + 4).map(l => `<div class="game-next"><span style="width:${Math.min(60, l.trim().length)}ch;margin-left:${l.match(/^\s*/)[0].length}ch"></span></div>`).join("");
    root.innerHTML = `<div class="game-layout">` + descPane() + `<div class="game-main">` + head + `<div class="game-board" id="game-board">${doneLines}
        <div class="game-line"><pre class="ghost hljs" id="game-ghost">${ghostHtml(line, G.mask)}</pre><input id="game-input" class="typed" type="text" spellcheck="false" autocomplete="off" autocapitalize="off" autocorrect="off" aria-label="Type the next line"></div>
        ${upcoming}${left > 0 ? `<div class="game-left muted small">${left} more line${left === 1 ? "" : "s"}</div>` : ""}</div>
      <div class="game-tools"><span class="small" id="game-msg">${G.msg || ""}</span>
        <span style="margin-left:auto;display:flex;gap:.4rem"><button type="button" class="btn btn-sm" id="game-skip" title="Reveal the whole line, 0 points">Skip</button></span></div></div></div>`;
    moveRunner(); wireDesc(root);
    const input = root.querySelector("#game-input");
    input.value = G.typed != null ? G.typed : indent;
    input.focus(); input.setSelectionRange(input.value.length, input.value.length);
    input.addEventListener("keydown", ev => {
      if (ev.key === "Enter") { ev.preventDefault(); submit(input.value); }
      else if (ev.key === "Tab") { ev.preventDefault(); const s = input.selectionStart; input.value = input.value.slice(0, s) + "    " + input.value.slice(input.selectionEnd); input.setSelectionRange(s + 4, s + 4); }
      else if (ev.key === "Escape") quit();
      else if (ev.key === "ArrowRight" && input.selectionStart === input.value.length) { if (autoType(input, ev.shiftKey)) ev.preventDefault(); }
      else if (ev.key === "ArrowLeft") { ev.preventDefault(); completeLine(input); }
    });
    input.addEventListener("input", () => { G.typed = input.value; paint(input.value); });
    paint(input.value);
    root.querySelector("#game-skip").addEventListener("click", skip);
    root.querySelector("#game-quit").addEventListener("click", quit);
  }

  // the runner starts where it stood on the previous render and glides to the new position (one step per finished line)
  function moveRunner() {
    const r = $("game-runner"); if (!r || !G) return;
    const pct = G.idx / G.lines.length * 100;
    requestAnimationFrame(() => requestAnimationFrame(() => { r.style.left = pct.toFixed(1) + "%"; if (pct > (G.prevPct || 0)) { r.classList.add("hop"); setTimeout(() => r.classList.remove("hop"), 500); } G.prevPct = pct; }));
  }

  // → fills in the next grey character (shift: the rest of the token) when it is visible; the line then scores 0
  function autoType(input, wholeToken) {
    const line = G.lines[G.idx], typed = input.value;
    if (typed !== line.slice(0, typed.length)) return false;                         // only while everything so far is right
    const tk = tokens(line); let pos = 0, i = 0;
    while (i < tk.length && pos + tk[i].length <= typed.length) { pos += tk[i].length; i++; }
    if (i >= tk.length) return false;
    const visible = G.mask[i] || /^\s+$/.test(tk[i]);
    if (!visible) return false;
    const add = wholeToken ? line.slice(typed.length, pos + tk[i].length) : line[typed.length];
    input.value = typed + add; G.typed = input.value; G.auto = true; G.linePts = 0;
    paint(input.value);
    const m = $("game-msg"); if (m) m.textContent = "auto-typed · this line scores 0";
    return true;
  }

  // ← types the whole line (hidden tokens included) for 0 points; Enter then submits it
  function completeLine(input) {
    const line = G.lines[G.idx];
    G.mask = G.mask.map(() => true);
    input.value = line; G.typed = line; G.auto = true; G.linePts = 0;
    input.setSelectionRange(line.length, line.length);
    paint(line);
    const m = $("game-msg"); if (m) m.textContent = "whole line auto-typed · this line scores 0 · press Enter";
  }

  // colour the ghost: characters already typed correctly turn solid, a typo turns red from the first mismatch
  function paint(typed) {
    const ghost = $("game-ghost"); if (!ghost || !G) return;
    const line = G.lines[G.idx];
    const tk = tokens(line), cc = charClasses(line); let pos = 0, html = "", bad = false;
    tk.forEach((t, i) => {
      let piece = "";
      for (const ch of t) {
        const typedCh = typed[pos];
        const cls = typedCh === undefined ? "pending" : (!bad && typedCh === ch) ? "ok" : (bad = true, "bad");
        const visible = G.mask[i] || /^\s+$/.test(t);
        const shown = typedCh !== undefined ? esc(typedCh) : visible ? esc(ch) : G.mode === "hard" ? "" : "•";   // what was typed, never the answer
        piece += visible || typedCh !== undefined ? `<span class="${cls} ${cc[pos]}">${shown}</span>` : `<span class="ghost-hidden">${shown}</span>`;
        pos++;
      }
      html += piece;
    });
    if (typed.length > line.length) html += `<span class="bad">${esc(typed.slice(line.length))}</span>`;
    ghost.innerHTML = html;
  }

  function reveal(why) {   // → true if a token was uncovered
    const hidden = G.mask.map((v, i) => v ? -1 : i).filter(i => i >= 0 && !/^\s+$/.test(tokens(G.lines[G.idx])[i]));
    if (hidden.length) G.mask[hidden[0]] = true;
    G.linePts = Math.max(0, G.linePts - 1);
    if (why === "hint") G.msg = hidden.length ? `Hint: one more token revealed · this line is now worth ${G.linePts}` : `Nothing left to reveal · this line is now worth ${G.linePts}`;
    return hidden.length > 0;
  }

  function submit(typed) {
    if (!G) return;
    const line = G.lines[G.idx];
    if (norm(typed) === norm(line)) {
      G.score += G.linePts; next(G.auto ? "✓ auto-typed · +0" : `✓ +${G.linePts}`);
      return;
    }
    G.wrong++;
    const a = norm(typed), b = norm(line);
    let k = 0; while (k < a.length && k < b.length && a[k] === b[k]) k++;
    const where = k >= a.length ? "your line is too short" : k >= b.length ? "your line is too long" : `first difference at column ${k + 1}`;
    const uncovered = reveal("wrong");
    G.msg = `✗ ${where}${uncovered ? " · one more token revealed" : ""} · this line is now worth ${G.linePts}`;
    G.typed = typed;
    render();
    const board = $("game-board"); if (board) { board.classList.remove("shake"); void board.offsetWidth; board.classList.add("shake"); }
  }

  function skip() { if (!G) return; G.skipped++; G.linePts = 0; next("skipped · 0"); }

  function next(msg) {
    G.idx++; G.typed = null; G.msg = msg; G.auto = false;
    if (G.idx < G.lines.length) { G.mask = lineMask(G.lines[G.idx], G.idx, G.mode); G.linePts = PTS[G.mode]; }
    else {
      G.recorded = !!V.token();                       // records are kept only for the owner (vault unlocked)
      if (G.recorded) {
        const key = bestKey(G.problem.slug, G.lang, G.mode), all = bests(), prev = all[key];
        G.newBest = !prev || G.score > prev.best;
        all[key] = { best: G.newBest ? G.score : prev.best, max: G.max, plays: (prev ? prev.plays : 0) + 1, lastAt: new Date().toISOString().slice(0, 10) };
        lsSet(STORE, all);
        document.dispatchEvent(new CustomEvent("lc-game-finished"));
      }
    }
    render();
  }

  function quit() { G = null; const root = $("lc-game"); if (root) root.innerHTML = ""; $("lc-game-card").hidden = true; $("lc-body").classList.remove("playing"); window.scrollTo({ top: 0, behavior: "smooth" }); }

  async function start(problem, lang, mode) {
    const card = $("lc-game-card"), root = $("lc-game");
    $("lc-body").classList.add("playing"); card.hidden = false; window.scrollTo({ top: 0 });
    root.innerHTML = `<div class="muted">Loading ${esc(problem.title)} (${LANG_NAME[lang]})…</div>`;
    const desc = {
      en: fetchFile(`${problem.path}/README.md`).then(descriptionHtml, () => `<p class="muted">No description in this folder.</p>`),
      ko: fetchFile(`${problem.path}/README.ko.md`).then(descriptionHtml, () => `<p class="muted">한국어 설명이 아직 없습니다.</p>`)
    };
    let text;
    try { text = await fetchFile(`${problem.path}/${problem.langs[lang].file}`); }
    catch (e) { root.innerHTML = `<span style="color:var(--danger)">Could not load the solution: ${esc(e.message)}</span>`; return; }
    const lines = playable(text, lang);
    if (!lines.length) { root.innerHTML = `<span class="muted">Nothing to type in this file.</span>`; return; }
    G = { problem, lang, mode, lines, idx: 0, score: 0, max: lines.length * PTS[mode], linePts: PTS[mode], wrong: 0, skipped: 0, t0: Date.now(), typed: null, msg: "", newBest: false, recorded: false, auto: false, desc: {}, descLang: descLang() };
    G.mask = lineMask(lines[0], 0, mode);
    render();
    Object.entries(desc).forEach(([l, pr]) => pr.then(html => {
      if (!G || G.problem !== problem) return;
      G.desc[l] = html;
      const d = root.querySelector(".game-desc-body"); if (d && G.descLang === l) d.innerHTML = html;
    }));
  }

  // the launcher: problem + language + mode pickers
  function renderLauncher(root, problems) {
    const playable_ = problems.filter(p => Object.keys(p.langs || {}).length);
    if (!playable_.length) { root.innerHTML = ""; return; }
    const sel = (id, opts) => `<select id="${id}" class="search" style="margin:0;min-width:0">${opts}</select>`;
    root.innerHTML = `<div class="form-actions">
        ${sel("game-problem", playable_.map((p, i) => `<option value="${i}">${esc((p.id != null ? "#" + p.id + " " : "") + p.title)}</option>`).join(""))}
        ${sel("game-lang", "")}
        <div class="seg" id="game-mode">${["easy", "medium", "hard"].map((m, i) => `<button type="button" data-m="${m}" class="${i === 0 ? "on" : ""}">${m}</button>`).join("")}</div>
        <button type="button" class="btn btn-primary" id="game-start">▶ Play</button><button type="button" class="btn" id="game-random2" title="Random problem, language and mode">🎲</button>
      </div><p class="small muted" style="margin:.5rem 0 0">easy shows the whole line in grey, medium about half of it, hard nothing. Points per line 1 / 2 / 3; a wrong try reveals a token and costs a point; → auto-types a grey character and ← the whole line, but that line then scores 0.</p>`;
    const langSel = root.querySelector("#game-lang"), probSel = root.querySelector("#game-problem");
    const fillLangs = () => { const p = playable_[+probSel.value]; langSel.innerHTML = Object.keys(p.langs).map(k => `<option value="${k}">${LANG_NAME[k] || k}</option>`).join(""); };
    probSel.addEventListener("change", fillLangs); fillLangs();
    let mode = "easy";
    root.querySelectorAll("#game-mode button").forEach(b => b.addEventListener("click", () => { mode = b.dataset.m; root.querySelectorAll("#game-mode button").forEach(x => x.classList.toggle("on", x === b)); }));
    root.querySelector("#game-start").addEventListener("click", () => start(playable_[+probSel.value], langSel.value, mode));
    root.querySelector("#game-random2").addEventListener("click", () => window.LCGame.random());
    window.LCGame._problems = playable_;
  }

  window.LCGame = {
    start, renderLauncher, bests,
    random() {
      const ps = window.LCGame._problems || []; if (!ps.length) return;
      const p = ps[Math.floor(Math.random() * ps.length)], langs = Object.keys(p.langs), modes = ["easy", "medium", "hard"];
      start(p, langs[Math.floor(Math.random() * langs.length)], modes[Math.floor(Math.random() * modes.length)]);
    },
    bestFor(slug) {   // best percentage over all languages and modes, for the table badge
      let best = null;
      Object.entries(bests()).forEach(([k, v]) => { if (k.startsWith(slug + "|")) { const pct = Math.round(v.best / v.max * 100); if (best == null || pct > best) best = pct; } });
      return best;
    }
  };
})();
