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

  let G = null;   // the current game
  const $ = id => document.getElementById(id);

  function lineMask(line, idx, mode) {
    // which tokens start visible: easy all, hard none, medium roughly half (deterministic per line)
    const tk = tokens(line);
    return tk.map((t, i) => /^\s+$/.test(t) ? true : mode === "easy" ? true : mode === "hard" ? false : (hash(`${idx}:${i}:${t}`) % 100) < 55);
  }

  function ghostHtml(line, mask) {
    const tk = tokens(line);
    // hard mode shows nothing for hidden tokens (not even their length); medium shows dots of the same length
    return tk.map((t, i) => mask[i] ? esc(t) : G.mode === "hard" ? "" : `<span class="ghost-hidden">${"•".repeat(t.length)}</span>`).join("");
  }

  function render() {
    const root = $("lc-game"); if (!root || !G) return;
    const p = G.problem, done = G.idx >= G.lines.length;
    const best = bests()[bestKey(p.slug, G.lang, G.mode)];
    const head = `<div class="game-head">
        <div><b>${esc(p.id != null ? "#" + p.id + " " : "")}${esc(p.title)}</b> <span class="muted">· ${LANG_NAME[G.lang]} · ${G.mode}</span>
          ${p.url ? `<a class="small" href="${esc(p.url)}" target="_blank" rel="noopener">problem ↗</a>` : ""}</div>
        <div class="game-meta"><span>line <b>${Math.min(G.idx + 1, G.lines.length)}</b>/${G.lines.length}</span><span>score <b id="game-score">${G.score}</b>/${G.max}</span>${best ? `<span class="muted">best ${best.best}/${best.max}</span>` : ""}<button type="button" class="btn btn-sm" id="game-quit">Quit</button></div>
      </div>
      <div class="game-progress"><i style="width:${(G.idx / G.lines.length * 100).toFixed(1)}%"></i></div>`;
    if (done) {
      const secs = Math.round((Date.now() - G.t0) / 1000), pct = Math.round(G.score / G.max * 100);
      root.innerHTML = head + `<div class="game-done">
          <div class="game-done-score">${G.score} / ${G.max} <span class="unit">points · ${pct}%</span></div>
          <div class="muted">${G.lines.length} lines · ${G.wrong} wrong tries · ${G.skipped} skipped · ${Math.floor(secs / 60)}m ${secs % 60}s${G.newBest ? " · 🏆 new best" : ""}</div>
          <div class="form-actions" style="margin-top:.8rem"><button type="button" class="btn btn-primary" id="game-again">Play again</button>
            ${["easy", "medium", "hard"].filter(m => m !== G.mode).map(m => `<button type="button" class="btn" data-mode="${m}">Try ${m}</button>`).join("")}
            <button type="button" class="btn" id="game-random">Random problem</button></div>
          <details class="more" style="margin-top:.8rem"><summary>Show the solution</summary><pre class="game-code"><code>${esc(G.lines.join("\n"))}</code></pre></details></div>`;
      root.querySelector("#game-again").addEventListener("click", () => start(G.problem, G.lang, G.mode));
      root.querySelectorAll("[data-mode]").forEach(b => b.addEventListener("click", () => start(G.problem, G.lang, b.dataset.mode)));
      root.querySelector("#game-random").addEventListener("click", () => window.LCGame.random());
      root.querySelector("#game-quit").addEventListener("click", quit);
      return;
    }
    const line = G.lines[G.idx], indent = line.match(/^\s*/)[0];
    const doneLines = G.lines.slice(Math.max(0, G.idx - 6), G.idx).map(l => `<div class="game-prev">${esc(l)}</div>`).join("");
    const left = G.lines.length - G.idx - 1;
    const upcoming = G.mode === "hard" ? "" : G.lines.slice(G.idx + 1, G.idx + 4).map(l => `<div class="game-next"><span style="width:${Math.min(60, l.trim().length)}ch;margin-left:${l.match(/^\s*/)[0].length}ch"></span></div>`).join("");
    root.innerHTML = head + `<div class="game-board" id="game-board">${doneLines}
        <div class="game-line"><pre class="ghost" id="game-ghost">${ghostHtml(line, G.mask)}</pre><input id="game-input" class="typed" type="text" spellcheck="false" autocomplete="off" autocapitalize="off" autocorrect="off" aria-label="Type the next line"></div>
        ${upcoming}${left > 0 ? `<div class="game-left muted small">${left} more line${left === 1 ? "" : "s"}</div>` : ""}</div>
      <div class="game-tools"><span class="small" id="game-msg">${G.msg || "Type the line and press Enter · Tab inserts 4 spaces · Esc quits"}</span>
        <span style="margin-left:auto;display:flex;gap:.4rem"><button type="button" class="btn btn-sm" id="game-hint" title="Reveal one more token (costs a point of this line)">Hint</button><button type="button" class="btn btn-sm" id="game-skip" title="Reveal the whole line, 0 points">Skip</button></span></div>`;
    const input = root.querySelector("#game-input");
    input.value = G.typed != null ? G.typed : indent;
    input.focus(); input.setSelectionRange(input.value.length, input.value.length);
    input.addEventListener("keydown", ev => {
      if (ev.key === "Enter") { ev.preventDefault(); submit(input.value); }
      else if (ev.key === "Tab") { ev.preventDefault(); const s = input.selectionStart; input.value = input.value.slice(0, s) + "    " + input.value.slice(input.selectionEnd); input.setSelectionRange(s + 4, s + 4); }
      else if (ev.key === "Escape") quit();
    });
    input.addEventListener("input", () => { G.typed = input.value; paint(input.value); });
    paint(input.value);
    root.querySelector("#game-hint").addEventListener("click", () => { reveal("hint"); G.typed = input.value; render(); });
    root.querySelector("#game-skip").addEventListener("click", skip);
    root.querySelector("#game-quit").addEventListener("click", quit);
  }

  // colour the ghost: characters already typed correctly turn solid, a typo turns red from the first mismatch
  function paint(typed) {
    const ghost = $("game-ghost"); if (!ghost || !G) return;
    const line = G.lines[G.idx];
    const tk = tokens(line); let pos = 0, html = "", bad = false;
    tk.forEach((t, i) => {
      let piece = "";
      for (const ch of t) {
        const typedCh = typed[pos];
        const cls = typedCh === undefined ? "" : (!bad && typedCh === ch) ? "ok" : (bad = true, "bad");
        const visible = G.mask[i] || /^\s+$/.test(t);
        const shown = visible ? esc(ch) : typedCh !== undefined ? esc(typedCh) : G.mode === "hard" ? "" : "•";
        piece += cls ? `<span class="${cls}">${shown}</span>` : visible ? shown : `<span class="ghost-hidden">${shown}</span>`;
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
      G.score += G.linePts; next(`✓ +${G.linePts}`);
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
    G.idx++; G.typed = null; G.msg = msg;
    if (G.idx < G.lines.length) { G.mask = lineMask(G.lines[G.idx], G.idx, G.mode); G.linePts = PTS[G.mode]; }
    else {
      const key = bestKey(G.problem.slug, G.lang, G.mode), all = bests(), prev = all[key];
      G.newBest = !prev || G.score > prev.best;
      all[key] = { best: G.newBest ? G.score : prev.best, max: G.max, plays: (prev ? prev.plays : 0) + 1, lastAt: new Date().toISOString().slice(0, 10) };
      lsSet(STORE, all);
      document.dispatchEvent(new CustomEvent("lc-game-finished"));
    }
    render();
  }

  function quit() { G = null; const root = $("lc-game"); if (root) root.innerHTML = ""; $("lc-game-card").hidden = true; }

  async function start(problem, lang, mode) {
    const card = $("lc-game-card"), root = $("lc-game");
    card.hidden = false; card.scrollIntoView({ behavior: "smooth", block: "start" });
    root.innerHTML = `<div class="muted">Loading ${esc(problem.title)} (${LANG_NAME[lang]})…</div>`;
    let text;
    try { text = await V.ghRaw(CFG.repo, `${problem.path}/${problem.langs[lang].file}`, CFG.branch || "main"); }
    catch (e) { root.innerHTML = `<span style="color:var(--danger)">Could not load the solution: ${esc(e.message)}</span>`; return; }
    const lines = playable(text, lang);
    if (!lines.length) { root.innerHTML = `<span class="muted">Nothing to type in this file.</span>`; return; }
    G = { problem, lang, mode, lines, idx: 0, score: 0, max: lines.length * PTS[mode], linePts: PTS[mode], wrong: 0, skipped: 0, t0: Date.now(), typed: null, msg: "", newBest: false };
    G.mask = lineMask(lines[0], 0, mode);
    render();
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
      </div><p class="small muted" style="margin:.5rem 0 0">easy shows the whole line in grey, medium about half of it, hard nothing. Points per line 1 / 2 / 3; a wrong try reveals a token and costs a point. Scores stay in this browser.</p>`;
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
