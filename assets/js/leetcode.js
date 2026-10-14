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
    const H = top + 5 * step + 4;                                   // Mon-Fri only
    const labels = svgEl("svg", { class: "heat-labels", width: left, height: H, viewBox: `0 0 ${left} ${H}` });
    [["Mon", 0], ["Wed", 2], ["Fri", 4]].forEach(([t, r]) => { const x = svgEl("text", { x: 0, y: top + r * step + cell - 2 }); x.textContent = t; labels.appendChild(x); });
    const svg = svgEl("svg", { class: "heatmap", width: weeks * step, height: H, viewBox: `0 0 ${weeks * step} ${H}` });
    let lastMonth = -1;
    for (let c = 0; c < weeks; c++) {
      const mon = addDays(start, c * 7);
      if (mon.getMonth() !== lastMonth) {
        if (c > 0 || addDays(mon, 6).getMonth() === mon.getMonth()) { const fits = c * step + 24 <= weeks * step; const t = svgEl("text", fits ? { x: c * step, y: 10 } : { x: weeks * step, y: 10, "text-anchor": "end" }); t.textContent = MON[mon.getMonth()]; svg.appendChild(t); }
        lastMonth = mon.getMonth();
      }
      for (let r = 0; r < 5; r++) {                                 // weekends are not shown
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
      const rows = P.filter(p => !diff || p.difficulty === diff).sort((a, b) => (a.id || 0) - (b.id || 0)).map(p => {
        const url = p.url || `https://leetcode.com/problems/${p.slug}/`;
        const best = window.LCGame ? window.LCGame.bestFor(p.slug) : null;
        return `<tr><td class="num">${p.id ?? ""}</td><td><a href="${esc(url)}" target="_blank" rel="noopener">${esc(p.title)}</a>${best != null ? ` <span class="game-badge" title="best practice score">🎮 ${best}%</span>` : ""}</td>
          <td>${p.difficulty ? `<span class="diff diff-${p.difficulty.toLowerCase()}">${p.difficulty}</span>` : ""}</td>
          <td class="small"><button type="button" class="btn btn-sm game-play" data-slug="${esc(p.slug)}" title="Retype this solution line by line">▶ Play</button> <a href="https://github.com/${CFG.repo}/tree/${CFG.branch || "main"}/${encodeURIComponent(p.path)}" target="_blank" rel="noopener">GitHub ↗</a></td></tr>`;
      });
      root.innerHTML = rows.length ? `<div style="overflow-x:auto"><table class="lc-table"><thead><tr><th>#</th><th>Problem</th><th>Difficulty</th><th></th></tr></thead><tbody>${rows.join("")}</tbody></table></div>` : `<div class="empty">Nothing matches.</div>`;
      root.querySelectorAll(".game-play").forEach(b => b.addEventListener("click", () => { const p = P.find(x => x.slug === b.dataset.slug); if (p) window.LCGame.start(p, Object.keys(p.langs)[0], "easy"); }));
    };
    if (filterRoot) {
      filterRoot.innerHTML = `<div class="seg">${["", "Easy", "Medium", "Hard"].map(d => `<button type="button" data-d="${d}" class="${d === diff ? "on" : ""}">${d || "All"}</button>`).join("")}</div>`;
      filterRoot.querySelectorAll("button").forEach(b => b.addEventListener("click", () => { diff = b.dataset.d; filterRoot.querySelectorAll("button").forEach(x => x.classList.toggle("on", x === b)); draw(); }));
    }
    draw();
  }

  const $ = id => document.getElementById(id);

  // practice-count heatmaps: one per language, a cell per problem (sorted by number), darker = more games played
  const PRACTICE = [["py", "Python", "37, 99, 235"], ["c", "C", "220, 38, 38"], ["cpp", "C++", "234, 88, 12"]];   // base colours (RGB)
  function renderPractice(root) {
    const plays = {};   // slug -> lang -> games played (all modes)
    Object.entries(window.LCGame ? window.LCGame.bests() : {}).forEach(([k, v]) => { const [slug, lang] = k.split("|"); (plays[slug] = plays[slug] || {})[lang] = (plays[slug][lang] || 0) + (v.plays || 0); });
    const probs = [...P].sort((a, b) => (a.id || 0) - (b.id || 0));
    root.innerHTML = PRACTICE.map(([lang, name, rgb]) => {
      const cells = probs.map(p => {
        const n = (plays[p.slug] || {})[lang] || 0, has = !!p.langs[lang];
        const lv = n === 0 ? 0 : Math.min(4, n);
        const style = lv ? `background:rgba(${rgb},${(0.25 + 0.75 * lv / 4).toFixed(2)});color:${lv >= 3 ? "#fff" : "inherit"};border-color:transparent` : "";
        return `<button type="button" class="pcell${has ? "" : " none"}" style="${style}" data-slug="${esc(p.slug)}" data-lang="${lang}" title="#${p.id ?? "?"} ${esc(p.title)} · ${has ? `${n} game${n === 1 ? "" : "s"} in ${name}` : `no ${name} solution`}" ${has ? "" : "disabled"}>${p.id ?? "?"}</button>`;
      }).join("");
      const total_ = probs.reduce((a, p) => a + ((plays[p.slug] || {})[lang] || 0), 0);
      return `<div class="card practice-card"><div class="practice-head"><b style="color:rgb(${rgb})">${name}</b><span class="small muted">${total_} game${total_ === 1 ? "" : "s"} · click a problem to play</span></div><div class="pgrid">${cells}</div></div>`;
    }).join("");
    root.querySelectorAll(".pcell:not([disabled])").forEach(b => b.addEventListener("click", () => { const p = P.find(x => x.slug === b.dataset.slug); if (p && window.LCGame) window.LCGame.start(p, b.dataset.lang, "easy"); }));
  }

  function render() {
    const empty = total === 0;
    $("lc-onboarding").hidden = !empty; $("lc-sections").hidden = empty;
    if (!empty) { renderHeatmap($("lc-heatmap")); renderPractice($("lc-practice")); renderTable($("lc-table"), $("lc-filter")); if (window.LCGame) window.LCGame.renderLauncher($("lc-game-launch"), P); }
  }
  document.addEventListener("lc-game-finished", () => { if (total) { renderTable($("lc-table"), $("lc-filter")); renderPractice($("lc-practice")); } });
  async function boot() {
    const connected = !!V.token();
    $("lc-body").hidden = !connected;
    if (!connected) { setData({}); return; }
    try { const c = JSON.parse(V.lsGet(CACHE_KEY) || "null"); if (c && c.data) { setData(c.data); render(); } } catch {}   // cached copy first
    try {
      setData(JSON.parse(await V.ghRaw(CFG.trackerRepo, "leetcode.json", CFG.trackerBranch || "main")));
      V.lsSet(CACHE_KEY, JSON.stringify({ at: Date.now(), data: D })); render();
    } catch (e) {
      if (e.status === 404) { setData({}); render(); $("lc-onboarding").textContent = "leetcode.json has not been synced to the private repo yet."; }
      else $("lc-onboarding").innerHTML = `<span style="color:var(--danger)">Could not load the LeetCode index: ${esc(e.message)}</span>`;
    }
  }
  document.addEventListener("DOMContentLoaded", () => {
    V.renderConnect($("lc-connect"), { repo: CFG.trackerRepo, onChange: boot });
    boot();
  });
})();
