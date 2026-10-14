/* LeetCode page. Owner-only: after the vault is unlocked (vault.js) the solution index leetcode.json is read from the
   private tracker repo through the GitHub API, and solution files from the solutions repo the same way. */
(function () {
  "use strict";
  const CFG = window.LEETCODE_CONFIG || {};
  const V = window.GHVault, CACHE_KEY = "leetcode_cache";
  let D = {}, P = [];
  const LANGS = [["c", "C"], ["cpp", "C++"], ["py", "Python"]];
  const pad = n => String(n).padStart(2, "0");
  const keyOf = d => `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`;
  const parseKey = k => { const [y, m, d] = k.split("-").map(Number); return new Date(y, m - 1, d); };
  const addDays = (d, n) => { const x = new Date(d); x.setDate(x.getDate() + n); return x; };
  const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const svgEl = (t, a = {}) => { const e = document.createElementNS("http://www.w3.org/2000/svg", t); for (const k in a) e.setAttribute(k, a[k]); return e; };
  const MON = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
  const TODAY = new Date(); TODAY.setHours(0, 0, 0, 0);

  // activity per day: first solves and every accepted re-submission (l.days), so reviewing a problem counts too
  let byDay = {}, days = [], total = 0, all3 = 0;
  function setData(data) {
    D = data || {}; P = Array.isArray(D.problems) ? D.problems : [];
    byDay = {};
    P.forEach(p => Object.values(p.langs || {}).forEach(l => (l.days && l.days.length ? l.days : [l.date]).forEach(d => { if (d) byDay[d] = (byDay[d] || 0) + 1; })));
    days = Object.keys(byDay).sort();
    total = P.length; all3 = P.filter(p => LANGS.every(([k]) => p.langs[k])).length;
  }
  function streak() { let d = TODAY, n = 0; if (!byDay[keyOf(d)]) d = addDays(d, -1); while (byDay[keyOf(d)]) { n++; d = addDays(d, -1); } return n; }

  function renderStats(root) {
    const tile = (l, v) => `<div class="stat stat-lc"><div class="stat-label">${l}</div><div class="stat-value">${v}</div></div>`;
    root.innerHTML = `<div class="grid grid-4">
      ${tile("✅ Solved", `${total}<span class="unit">${total === 1 ? "problem" : "problems"}</span>`)}
      ${tile("🔥 Daily streak", `${streak()}<span class="unit">${streak() === 1 ? "day" : "days"}</span>`)}
      ${tile("🌐 All three languages", `${all3}<span class="unit">/ ${total}</span>`)}
      ${tile("📅 Last solve", days.length ? days[days.length - 1] : "–")}
    </div>`;
  }

  function renderHeatmap(root) {
    const cell = 12, gap = 3, step = cell + gap, left = 22, top = 18;
    const w = root.clientWidth, weeks = w < 200 ? 26 : Math.max(8, Math.min(26, Math.floor((w - 40 - left) / step)));
    const start = addDays(TODAY, -((TODAY.getDay() + 6) % 7) - (weeks - 1) * 7);
    const H = top + 7 * step + 4;
    const labels = svgEl("svg", { class: "heat-labels", width: left, height: H, viewBox: `0 0 ${left} ${H}` });
    [["Mon", 0], ["Wed", 2], ["Fri", 4], ["Sun", 6]].forEach(([t, r]) => { const x = svgEl("text", { x: 0, y: top + r * step + cell - 2 }); x.textContent = t; labels.appendChild(x); });
    const svg = svgEl("svg", { class: "heatmap", width: weeks * step, height: H, viewBox: `0 0 ${weeks * step} ${H}` });
    let lastMonth = -1;
    for (let c = 0; c < weeks; c++) {
      const mon = addDays(start, c * 7);
      if (mon.getMonth() !== lastMonth) {
        if (c > 0 || addDays(mon, 6).getMonth() === mon.getMonth()) { const fits = c * step + 24 <= weeks * step; const t = svgEl("text", fits ? { x: c * step, y: 10 } : { x: weeks * step, y: 10, "text-anchor": "end" }); t.textContent = MON[mon.getMonth()]; svg.appendChild(t); }
        lastMonth = mon.getMonth();
      }
      for (let r = 0; r < 7; r++) {
        const d = addDays(mon, r); if (d > TODAY) continue;
        const k = keyOf(d), n = byDay[k] || 0, lv = n === 0 ? 0 : Math.min(4, n + 1);
        const rect = svgEl("rect", { x: c * step, y: top + r * step, width: cell, height: cell, class: `heat-${lv}${k === keyOf(TODAY) ? " heat-today" : ""}` });
        const t = svgEl("title"); t.textContent = `${k} · ${n ? n + " solution file(s)" : "nothing"}`; rect.appendChild(t); svg.appendChild(rect);
      }
    }
    root.innerHTML = ""; const outer = document.createElement("div"); outer.className = "heat-outer"; outer.appendChild(labels);
    const wrap = document.createElement("div"); wrap.className = "heatmap-wrap"; wrap.style.cssText = "flex:1;min-width:0"; wrap.appendChild(svg); outer.appendChild(wrap); root.appendChild(outer);
  }

  function renderTable(root, filterRoot) {
    let diff = "";
    const draw = () => {
      const rows = P.filter(p => !diff || p.difficulty === diff).map((p, i) => {
        const chips = LANGS.map(([k, l]) => p.langs[k]
          ? `<span class="lang on" data-p="${i}" data-l="${k}" title="first solved ${p.langs[k].date || "?"}">${l}</span>`
          : `<span class="lang off">${l}</span>`).join("");
        const url = p.url || `https://leetcode.com/problems/${p.slug}/`;
        const best = window.LCGame ? window.LCGame.bestFor(p.slug) : null;
        return `<tr data-i="${i}"><td class="num">${p.id ?? ""}</td><td><a href="${esc(url)}" target="_blank" rel="noopener">${esc(p.title)}</a>${best != null ? ` <span class="game-badge" title="best practice score">🎮 ${best}%</span>` : ""}</td>
          <td>${p.difficulty ? `<span class="diff diff-${p.difficulty.toLowerCase()}">${p.difficulty}</span>` : ""}</td>
          <td>${chips}</td><td class="num hide-sm">${p.first || ""}</td>
          <td class="small"><button type="button" class="btn btn-sm game-play" data-p="${i}" title="Retype this solution line by line">▶ Play</button> <a href="https://github.com/${CFG.repo}/tree/${CFG.branch || "main"}/${encodeURIComponent(p.path)}" target="_blank" rel="noopener">GitHub ↗</a></td></tr>`;
      });
      root.innerHTML = rows.length ? `<div style="overflow-x:auto"><table class="lc-table"><thead><tr><th>#</th><th>Problem</th><th>Difficulty</th><th>Languages</th><th class="hide-sm">First solved</th><th></th></tr></thead><tbody>${rows.join("")}</tbody></table></div>` : `<div class="empty">Nothing matches.</div>`;
      root.querySelectorAll(".lang.on").forEach(ch => ch.addEventListener("click", () => toggleCode(ch)));
      root.querySelectorAll(".game-play").forEach(b => b.addEventListener("click", () => { const p = P.filter(x => !diff || x.difficulty === diff)[+b.dataset.p]; window.LCGame.start(p, Object.keys(p.langs)[0], "easy"); }));
    };
    if (filterRoot) {
      filterRoot.innerHTML = `<div class="seg">${["", "Easy", "Medium", "Hard"].map(d => `<button type="button" data-d="${d}" class="${d === diff ? "on" : ""}">${d || "All"}</button>`).join("")}</div>`;
      filterRoot.querySelectorAll("button").forEach(b => b.addEventListener("click", () => { diff = b.dataset.d; filterRoot.querySelectorAll("button").forEach(x => x.classList.toggle("on", x === b)); draw(); }));
    }
    draw();
    async function toggleCode(chip) {
      const tr = chip.closest("tr"), next = tr.nextElementSibling;
      const already = next && next.classList.contains("lc-code") && next.dataset.l === chip.dataset.l;
      if (next && next.classList.contains("lc-code")) next.remove();
      tr.querySelectorAll(".lang.open").forEach(x => x.classList.remove("open"));
      if (already) return;
      chip.classList.add("open");
      const p = P.filter(x => !diff || x.difficulty === diff)[+chip.dataset.p], f = p.langs[chip.dataset.l].file;
      const row = document.createElement("tr"); row.className = "lc-code"; row.dataset.l = chip.dataset.l;
      row.innerHTML = `<td colspan="6"><div class="code-head"><span>${esc(f)}</span><a href="https://github.com/${CFG.repo}/blob/${CFG.branch || "main"}/${encodeURIComponent(p.path)}/${encodeURIComponent(f)}" target="_blank" rel="noopener">open on GitHub ↗</a></div><pre><code>Loading…</code></pre></td>`;
      tr.after(row);
      try {
        const txt = await V.ghRaw(CFG.repo, `${p.path}/${f}`, CFG.branch || "main");
        const code = row.querySelector("code"); code.textContent = txt;
        code.className = `language-${{ c: "c", cpp: "cpp", py: "python" }[chip.dataset.l]}`;
        if (window.hljs) hljs.highlightElement(code);
      } catch { row.querySelector("code").textContent = "Could not load the file."; }
    }
  }

  const $ = id => document.getElementById(id);
  function render() {
    const empty = total === 0;
    $("lc-onboarding").hidden = !empty; $("lc-sections").hidden = empty; $("lc-stats").hidden = empty;
    $("lc-updated").textContent = D.updated ? `Last synced ${D.updated.replace("T", " ").slice(0, 16)} KST` : "";
    if (!empty) { renderStats($("lc-stats")); renderHeatmap($("lc-heatmap")); renderTable($("lc-table"), $("lc-filter")); if (window.LCGame) window.LCGame.renderLauncher($("lc-game-launch"), P); }
  }
  document.addEventListener("lc-game-finished", () => { if (total) renderTable($("lc-table"), $("lc-filter")); });
  async function boot() {
    const connected = !!V.token();
    $("lc-body").hidden = !connected;
    if (!connected) { setData({}); return; }
    try { const c = JSON.parse(V.lsGet(CACHE_KEY) || "null"); if (c && c.data) { setData(c.data); render(); } } catch {}   // cached copy first
    try {
      setData(JSON.parse(await V.ghRaw(CFG.trackerRepo, "leetcode.json", CFG.trackerBranch || "main")));
      V.lsSet(CACHE_KEY, JSON.stringify({ at: Date.now(), data: D })); render();
    } catch (e) {
      if (e.status === 404) { setData({}); render(); $("lc-updated").textContent = "leetcode.json has not been synced to the private repo yet."; }
      else $("lc-updated").innerHTML = `<span style="color:var(--danger)">Could not load the LeetCode index: ${esc(e.message)}</span>`;
    }
  }
  document.addEventListener("DOMContentLoaded", () => {
    V.renderConnect($("lc-connect"), { repo: CFG.trackerRepo, onChange: boot });
    boot();
  });
})();
