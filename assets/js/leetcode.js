/* LeetCode page + home summary. Reads window.LEETCODE_DATA (from _data/leetcode.json). */
(function () {
  "use strict";
  const D = window.LEETCODE_DATA || {};
  const CFG = window.LEETCODE_CONFIG || {};
  const P = Array.isArray(D.problems) ? D.problems : [];
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
  const byDay = {};
  P.forEach(p => Object.values(p.langs || {}).forEach(l => (l.days && l.days.length ? l.days : [l.date]).forEach(d => { if (d) byDay[d] = (byDay[d] || 0) + 1; })));
  const days = Object.keys(byDay).sort();
  function streak() { let d = TODAY, n = 0; if (!byDay[keyOf(d)]) d = addDays(d, -1); while (byDay[keyOf(d)]) { n++; d = addDays(d, -1); } return n; }
  function best() { let b = 0, c = 0, prev = null; days.forEach(k => { const d = parseKey(k); c = prev && (d - prev) / 864e5 === 1 ? c + 1 : 1; b = Math.max(b, c); prev = d; }); return b; }
  const total = P.length, all3 = P.filter(p => LANGS.every(([k]) => p.langs[k])).length;
  const perLang = Object.fromEntries(LANGS.map(([k]) => [k, P.filter(p => p.langs[k]).length]));
  const week0 = addDays(TODAY, -((TODAY.getDay() + 6) % 7));
  const thisWeek = days.filter(k => parseKey(k) >= week0).length;

  function renderSummary(root) {   // home card
    if (!total) { root.innerHTML = `<div class="empty">No solutions yet.</div>`; return; }
    root.innerHTML = `<div class="today-bar"><span class="chip on">✅ <span class="n">${total}</span> solved</span><span class="chip">🔥 <span class="n">${streak()}</span>-day streak</span>` +
      LANGS.map(([k, l]) => `<span class="chip">${l} <span class="n">${perLang[k]}</span></span>`).join("") + `</div>`;
  }

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
        return `<tr data-i="${i}"><td class="num">${p.id ?? ""}</td><td><a href="${esc(url)}" target="_blank" rel="noopener">${esc(p.title)}</a></td>
          <td>${p.difficulty ? `<span class="diff diff-${p.difficulty.toLowerCase()}">${p.difficulty}</span>` : ""}</td>
          <td>${chips}</td><td class="num hide-sm">${p.first || ""}</td>
          <td class="small"><a href="https://github.com/${CFG.repo}/tree/${CFG.branch || "main"}/${encodeURIComponent(p.path)}" target="_blank" rel="noopener">GitHub ↗</a></td></tr>`;
      });
      root.innerHTML = rows.length ? `<div style="overflow-x:auto"><table class="lc-table"><thead><tr><th>#</th><th>Problem</th><th>Difficulty</th><th>Languages</th><th class="hide-sm">First solved</th><th></th></tr></thead><tbody>${rows.join("")}</tbody></table></div>` : `<div class="empty">Nothing matches.</div>`;
      root.querySelectorAll(".lang.on").forEach(ch => ch.addEventListener("click", () => toggleCode(ch)));
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
      const raw = `https://raw.githubusercontent.com/${CFG.repo}/${CFG.branch || "main"}/${p.path}/${f}`;
      const row = document.createElement("tr"); row.className = "lc-code"; row.dataset.l = chip.dataset.l;
      row.innerHTML = `<td colspan="6"><div class="code-head"><span>${esc(f)}</span><a href="https://github.com/${CFG.repo}/blob/${CFG.branch || "main"}/${encodeURIComponent(p.path)}/${encodeURIComponent(f)}" target="_blank" rel="noopener">open on GitHub ↗</a></div><pre><code>Loading…</code></pre></td>`;
      tr.after(row);
      try {
        const txt = await (await fetch(raw)).text();
        const code = row.querySelector("code"); code.textContent = txt;
        code.className = `language-${{ c: "c", cpp: "cpp", py: "python" }[chip.dataset.l]}`;
        if (window.hljs) hljs.highlightElement(code);
      } catch { row.querySelector("code").textContent = "Could not load the file."; }
    }
  }

  document.addEventListener("DOMContentLoaded", () => {
    const $ = id => document.getElementById(id);
    if ($("lc-summary")) renderSummary($("lc-summary"));
    if ($("lc-stats")) {
      const empty = total === 0;
      $("lc-onboarding").hidden = !empty; $("lc-sections").hidden = empty; $("lc-stats").hidden = empty;
      if ($("lc-updated")) $("lc-updated").textContent = D.updated ? `Last synced ${D.updated.replace("T", " ").slice(0, 16)} KST` : "";
      if (!empty) { renderStats($("lc-stats")); renderHeatmap($("lc-heatmap")); renderTable($("lc-table"), $("lc-filter")); }
    }
  });
})();
