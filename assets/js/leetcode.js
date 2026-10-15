/* LeetCode page. Owner-only: after the vault is unlocked (vault.js) the solution index leetcode.json is read from the
   private tracker repo through the GitHub API, and solution files from the solutions repo the same way. */
(function () {
  "use strict";
  const CFG = window.LEETCODE_CONFIG || {};
  const V = window.GHVault, CACHE_KEY = "leetcode_cache";
  const PUB = Array.isArray(window.LC_PUBLIC) ? window.LC_PUBLIC : [];   // public problem list (no dates), for the game
  let D = {}, P = [];
  const LANGS = [["py", "Python"], ["c", "C"], ["cpp", "C++"]];
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

  let lcYear = TODAY.getFullYear();
  function renderHeatmap(root) {
    const years = [...new Set([...days.map(k => +k.slice(0, 4)), TODAY.getFullYear()])].sort((a, b) => b - a);
    if (!years.includes(lcYear)) lcYear = years[0];
    const draw = () => window.YearHeat.draw(root, {
      years, year: lcYear, onYear: y => { lcYear = y; draw(); },
      cellFor: k => { const n = byDay[k] || 0; return { cls: `heat-${n === 0 ? 0 : Math.min(4, n + 1)}`, title: `${k} · ${n ? n + " solution file(s)" : "nothing"}` }; }
    });
    draw(); root._redraw = draw;
  }

  function renderTable(root, filterRoot) {
    let diff = "";
    const draw = () => {
      const rows = P.filter(p => !diff || p.difficulty === diff).sort((a, b) => (a.id || 0) - (b.id || 0)).map(p => {
        const url = p.url || `https://leetcode.com/problems/${p.slug}/`;
        const best = V.token() && window.LCGame ? window.LCGame.bestFor(p.slug) : null;
        return `<tr><td class="num">${p.id ?? ""}</td><td><a href="${esc(url)}" target="_blank" rel="noopener">${esc(p.title)}</a>${best != null ? ` <span class="game-badge" title="best practice score">🎮 ${best}%</span>` : ""}</td>
          <td>${p.difficulty ? `<span class="diff diff-${p.difficulty.toLowerCase()}">${p.difficulty}</span>` : ""}</td>
          <td class="small">${LANGS.map(([k, l]) => p.langs[k] ? `<button type="button" class="btn btn-sm game-play" data-slug="${esc(p.slug)}" data-lang="${k}" title="Play this solution in ${l}">${l}</button>` : "").join(" ")}</td></tr>`;
      });
      root.innerHTML = rows.length ? `<div style="overflow-x:auto"><table class="lc-table"><thead><tr><th>#</th><th>Problem</th><th>Difficulty</th><th></th></tr></thead><tbody>${rows.join("")}</tbody></table></div>` : `<div class="empty">Nothing matches.</div>`;
      root.querySelectorAll(".game-play").forEach(b => b.addEventListener("click", () => { const p = P.find(x => x.slug === b.dataset.slug); if (p) window.LCGame.start(p, b.dataset.lang, "easy"); }));
    };
    if (filterRoot) {
      filterRoot.innerHTML = `<div class="seg">${["", "Easy", "Medium", "Hard"].map(d => `<button type="button" data-d="${d}" class="${d === diff ? "on" : ""}">${d || "All"}</button>`).join("")}</div>`;
      filterRoot.querySelectorAll("button").forEach(b => b.addEventListener("click", () => { diff = b.dataset.d; filterRoot.querySelectorAll("button").forEach(x => x.classList.toggle("on", x === b)); draw(); }));
    }
    draw();
  }

  const $ = id => document.getElementById(id);

  // practice-count heatmaps: one per language, one cell per problem (by number), darker = more games played here
  const PRACTICE = [["py", "Python", "37, 99, 235"], ["c", "C", "234, 179, 8"], ["cpp", "C++", "220, 38, 38"]];   // blue, yellow, red
  function renderPractice(root) {
    const plays = {};   // slug -> lang -> games played (all modes)
    Object.entries(window.LCGame ? window.LCGame.bests() : {}).forEach(([k, v]) => { const [slug, lang] = k.split("|"); (plays[slug] = plays[slug] || {})[lang] = (plays[slug][lang] || 0) + (v.plays || 0); });
    const probs = [...P].sort((a, b) => (a.id || 0) - (b.id || 0));
    root.innerHTML = PRACTICE.map(([lang, name]) => `<div class="card practice-card"><div class="practice-head"><b>${name}</b></div><div class="pheat" data-lang="${lang}"></div></div>`).join("");
    PRACTICE.forEach(([lang, name, rgb]) => {
      const items = probs.map(p => {
        const n = (plays[p.slug] || {})[lang] || 0, has = !!p.langs[lang], lv = Math.min(4, n);
        return { fill: lv ? `rgba(${rgb},${(0.3 + 0.7 * lv / 4).toFixed(2)})` : "", cls: has ? "heat-0" : "heat-0 pheat-none",
          title: `#${p.id ?? "?"} ${p.title} · ${has ? `${n} game${n === 1 ? "" : "s"} in ${name}` : `no ${name} solution`}`,
          click: has ? () => window.LCGame.start(p, lang, "easy") : null };
      });
      window.YearHeat.grid(root.querySelector(`.pheat[data-lang="${lang}"]`), items);
    });
  }

  function renderPublic() {
    const empty = P.length === 0;
    $("lc-onboarding").hidden = !empty; $("lc-sections").hidden = empty;
    if (!empty) { renderTable($("lc-table"), $("lc-filter")); if (window.LCGame) window.LCGame.renderLauncher($("lc-game-launch"), P); }
  }
  function renderPrivate() {
    const on = !!V.token() && P.length > 0;
    $("lc-private").hidden = !on;
    if (on) { renderHeatmap($("lc-heatmap")); renderPractice($("lc-practice")); }
  }
  document.addEventListener("lc-game-finished", () => { renderTable($("lc-table"), $("lc-filter")); renderPrivate(); });
  async function boot() {
    setData({ problems: PUB }); renderPublic(); renderPrivate();          // anyone: problem list + game
    $("lc-private-msg").textContent = "";
    if (!V.token()) return;
    try { const c = JSON.parse(V.lsGet(CACHE_KEY) || "null"); if (c && c.data) { setData(c.data); renderPublic(); renderPrivate(); } } catch {}   // cached copy first
    try {
      setData(JSON.parse(await V.ghRaw(CFG.trackerRepo, "leetcode.json", CFG.trackerBranch || "main")));
      V.lsSet(CACHE_KEY, JSON.stringify({ at: Date.now(), data: D })); renderPublic(); renderPrivate();
    } catch (e) {
      $("lc-private-msg").innerHTML = e.status === 404 ? "leetcode.json has not been synced to the private repo yet." : `<span style="color:var(--danger)">Could not load the private LeetCode index: ${esc(e.message)}</span>`;
    }
  }
  document.addEventListener("DOMContentLoaded", () => {
    V.renderConnect($("lc-connect"), { repo: CFG.trackerRepo, onChange: boot });
    boot();
  });
})();
