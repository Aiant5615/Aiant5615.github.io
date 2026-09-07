/* Routine tracker — reads /assets/data/days.json and renders stats and charts. No dependencies. */
(function () {
  "use strict";
  const CFG = window.TRACKER_CONFIG || {};
  const HABITS = Array.isArray(CFG.habits) ? CFG.habits : [];
  const HKEYS = HABITS.map(h => h.key);
  const HMAP = Object.fromEntries(HABITS.map(h => [h.key, h]));
  const REPO = CFG.repo || "";
  const SKIP_WEEKENDS = !!CFG.skipWeekends;

  // ───────── utils ─────────
  const pad = n => String(n).padStart(2, "0");
  const keyOf = d => `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`;
  const parseKey = k => { const [y, m, d] = k.split("-").map(Number); return new Date(y, m - 1, d); };
  const addDays = (d, n) => { const x = new Date(d); x.setDate(x.getDate() + n); return x; };
  const isWeekend = d => d.getDay() === 0 || d.getDay() === 6;
  const WD = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
  const MON = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
  const toMin = v => {
    if (v == null || v === "") return null;
    if (typeof v === "number") return v;               // in case YAML parsed 09:10 as the integer 550
    const m = String(v).match(/^(\d{1,2}):(\d{2})/);
    return m ? +m[1] * 60 + +m[2] : null;
  };
  const fmtMin = m => m == null ? "–" : `${pad(Math.floor(m / 60) % 24)}:${pad(Math.round(m % 60))}`;
  const fmtNum = (v, d = 1) => v == null || isNaN(v) ? "–" : (Math.round(v * 10 ** d) / 10 ** d).toString();
  const avg = arr => { const a = arr.filter(v => v != null && !isNaN(v)); return a.length ? a.reduce((s, v) => s + v, 0) / a.length : null; };
  const pct = (n, d) => d ? Math.round(100 * n / d) : null;
  const esc = s => String(s).replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const el = (tag, attrs = {}, html) => { const e = document.createElement(tag); for (const k in attrs) e.setAttribute(k, attrs[k]); if (html != null) e.innerHTML = html; return e; };
  const svgEl = (tag, attrs = {}) => { const e = document.createElementNS("http://www.w3.org/2000/svg", tag); for (const k in attrs) e.setAttribute(k, attrs[k]); return e; };
  const GOAL = toMin(CFG.arriveGoal) ?? 9 * 60;
  const TODAY = new Date(); TODAY.setHours(0, 0, 0, 0);
  const TODAY_KEY = keyOf(TODAY);
  const moodStr = m => m == null ? "–" : ["", "😩", "😕", "😐", "🙂", "😄"][Math.max(1, Math.min(5, Math.round(m)))];
  const habitSkipsWeekend = h => SKIP_WEEKENDS && !(HMAP[h] && HMAP[h].weekends);

  // ───────── data ─────────
  let E = {}, KEYS = [];
  function normalize(k, raw) {
    const done = new Set();
    if (Array.isArray(raw.done)) raw.done.forEach(x => done.add(String(x)));
    HKEYS.forEach(h => { if (raw[h] === true) done.add(h); });
    const arrive = toMin(raw.arrive), wake = toMin(raw.wake);
    let leave = toMin(raw.leave);
    let hours = raw.hours != null ? +raw.hours : null;
    if (arrive != null && leave != null) { if (leave < arrive) leave += 1440; hours = (leave - arrive) / 60; }   // leaving after midnight
    return {
      key: k, date: parseKey(k), done, arrive, leave, wake, hours,
      sleep: raw.sleep != null && raw.sleep !== "" ? +raw.sleep : null,
      mood: raw.mood != null && raw.mood !== "" ? +raw.mood : null,
      focus: raw.focus != null && raw.focus !== "" ? +raw.focus : null,
      note: raw.note ? String(raw.note) : ""
    };
  }
  function load(raw) {
    E = {}; Object.keys(raw || {}).forEach(k => { if (/^\d{4}-\d{2}-\d{2}$/.test(k) && raw[k] && typeof raw[k] === "object") E[k] = normalize(k, raw[k]); });
    KEYS = Object.keys(E).sort();
  }
  const get = k => E[k] || null;

  // ───────── stats ─────────
  // consecutive days where pred(entry) holds. With skipWk, weekends neither count nor break. Starts from yesterday if today is empty.
  function streak(pred, skipWk) {
    let d = TODAY, n = 0, guard = 0;
    const ok = x => pred(get(keyOf(x)));
    if (!ok(d)) d = addDays(d, -1);
    while (guard++ < 5000) {
      if (skipWk && isWeekend(d)) { d = addDays(d, -1); continue; }
      if (!ok(d)) break;
      n++; d = addDays(d, -1);
    }
    return n;
  }
  function bestStreak(pred, skipWk) {
    if (!KEYS.length) return 0;
    let best = 0, cur = 0;
    for (let d = parseKey(KEYS[0]); d <= TODAY; d = addDays(d, 1)) {
      if (skipWk && isWeekend(d)) continue;
      if (pred(get(keyOf(d)))) { cur++; best = Math.max(best, cur); } else cur = 0;
    }
    return best;
  }
  function weekRange(d = TODAY) { const mon = addDays(d, -((d.getDay() + 6) % 7)); return [mon, addDays(mon, 6)]; }
  function inRange(from, to) { return KEYS.filter(k => { const d = parseKey(k); return d >= from && d <= to; }).map(k => E[k]); }
  const thisWeek = () => { const [a, b] = weekRange(); return inRange(a, b); };
  const lastWeek = () => { const [a] = weekRange(); return inRange(addDays(a, -7), addDays(a, -1)); };
  const last7 = () => inRange(addDays(TODAY, -6), TODAY);
  const last30 = () => inRange(addDays(TODAY, -29), TODAY);
  const thisMonth = () => inRange(new Date(TODAY.getFullYear(), TODAY.getMonth(), 1), TODAY);
  const hasHabit = h => e => !!e && e.done.has(h);
  const logged = e => !!e;
  const isoWeek = d => { const x = new Date(Date.UTC(d.getFullYear(), d.getMonth(), d.getDate())); const day = x.getUTCDay() || 7; x.setUTCDate(x.getUTCDate() + 4 - day); const y0 = new Date(Date.UTC(x.getUTCFullYear(), 0, 1)); return `${x.getUTCFullYear()}-W${pad(Math.ceil(((x - y0) / 86400000 + 1) / 7))}`; };

  // ───────── today bar (home/tracker) ─────────
  function renderToday(root) {
    const t = get(TODAY_KEY);
    let html = `<span class="muted">${WD[TODAY.getDay()]}, ${MON[TODAY.getMonth()]} ${TODAY.getDate()}</span>`;
    if (!t) {
      html += `<span class="chip">Not logged yet</span>`;
      HABITS.forEach(h => html += `<span class="chip">${h.emoji} ${esc(h.label)}</span>`);
      html += `<a class="btn btn-sm btn-primary" href="/tracker/#log">Log today</a>`;
    } else {
      if (t.arrive != null) html += `<span class="chip ${t.arrive <= GOAL ? "on" : ""}">🏢 In at <span class="n">${fmtMin(t.arrive)}</span></span>`;
      HABITS.forEach(h => html += `<span class="chip ${t.done.has(h.key) ? "on" : ""}">${h.emoji} ${esc(h.label)}</span>`);
      if (t.mood != null) html += `<span class="chip">${moodStr(t.mood)}</span>`;
    }
    const s = streak(logged, SKIP_WEEKENDS);
    if (s > 0) html += `<span class="chip on">🔥 <span class="n">${s}</span>-day logging streak</span>`;
    root.innerHTML = html;
  }

  // ───────── stat tiles ─────────
  function renderStats(root) {
    const wk = thisWeek(), mo = thisMonth(), l7 = last7();
    const moArr = mo.filter(e => e.arrive != null);
    const onTime = moArr.filter(e => e.arrive <= GOAL).length;
    const tile = (label, value, sub) => `<div class="stat"><div class="stat-label">${label}</div><div class="stat-value">${value}</div><div class="stat-sub">${sub}</div></div>`;
    const wkLabel = SKIP_WEEKENDS ? "(weekdays)" : "";
    const tiles = [
      tile("📝 Logging streak", `${streak(logged, SKIP_WEEKENDS)}<span class="unit">days</span>`, `best ${bestStreak(logged, SKIP_WEEKENDS)} · ${wk.length} logged this week ${wkLabel}`),
      tile("🏢 Avg arrival this month", fmtMin(avg(moArr.map(e => e.arrive))), moArr.length ? `by ${fmtMin(GOAL)} on ${onTime}/${moArr.length} days (${pct(onTime, moArr.length)}%)` : "no data"),
      tile("⏱ Avg hours in lab", `${fmtNum(avg(mo.map(e => e.hours)))}<span class="unit">h</span>`, `avg leave ${fmtMin(avg(mo.map(e => e.leave)))} · this month`),
      tile("🌙 Sleep · Mood · Focus", `${fmtNum(avg(mo.map(e => e.sleep)))}<span class="unit">h</span> · ${moodStr(avg(mo.map(e => e.mood)))} · ${fmtNum(avg(mo.map(e => e.focus)))}<span class="unit">h</span>`, "monthly averages")
    ];
    root.innerHTML = `<div class="grid grid-4">${tiles.join("")}</div>`;
    const hb = HABITS.map(h => {
      const p = hasHabit(h.key), sk = habitSkipsWeekend(h.key);
      const n7 = l7.filter(p).length, m = mo.filter(p).length;
      const moDen = sk ? mo.filter(e => !isWeekend(e.date)).length : mo.length;
      return tile(`${h.emoji} ${esc(h.label)}`, `${streak(p, sk)}<span class="unit">-day streak</span>`, `last 7 days: ${n7} · this month: ${moDen ? pct(m, moDen) + "%" : "–"} · best ${bestStreak(p, sk)}`);
    });
    if (hb.length) root.insertAdjacentHTML("beforeend", `<div class="grid grid-4" style="margin-top:.9rem">${hb.join("")}</div>`);
  }

  // ───────── heatmap ─────────
  function renderHeatmap(root, selectRoot) {
    let metric = "all";
    const cell = 12, gap = 3, step = cell + gap, left = 22, top = 18;
    const weeksFor = () => { const w = root.clientWidth; return w < 200 ? 26 : Math.max(8, Math.min(26, Math.floor((w - 40 - left) / step))); };   // unknown width (hidden tab) → full 26 weeks
    const level = e => {
      if (!e) return 0;
      if (metric === "all") { if (!HKEYS.length) return 4; const r = [...e.done].filter(k => HMAP[k]).length / HKEYS.length; return r === 0 ? 0 : Math.max(1, Math.ceil(r * 4)); }
      if (metric === "arrive") { if (e.arrive == null) return 0; const d = e.arrive - GOAL; return d <= 0 ? 4 : d <= 30 ? 3 : d <= 60 ? 2 : 1; }
      return e.done.has(metric) ? 4 : 0;
    };
    const tip = e => {
      if (!e) return "no entry";
      const parts = [];
      if (e.arrive != null) parts.push(`in at ${fmtMin(e.arrive)}`);
      const hs = HABITS.filter(h => e.done.has(h.key)).map(h => h.label);
      parts.push(hs.length ? hs.join(", ") : "nothing checked");
      if (e.note) parts.push(e.note);
      return parts.join(" · ");
    };
    function draw() {
      const weeks = weeksFor();
      const start = addDays(TODAY, -((TODAY.getDay() + 6) % 7) - (weeks - 1) * 7);   // Monday of the first week
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
          const d = addDays(mon, r);
          if (d > TODAY) continue;
          const k = keyOf(d), e = get(k);
          const rect = svgEl("rect", { x: c * step, y: top + r * step, width: cell, height: cell, class: `heat-${level(e)}${k === TODAY_KEY ? " heat-today" : ""}` });
          const title = svgEl("title"); title.textContent = `${k} (${WD[d.getDay()]}) · ${tip(e)}`; rect.appendChild(title);
          if (e) { rect.style.cursor = "pointer"; rect.addEventListener("click", () => fillForm(k)); }
          svg.appendChild(rect);
        }
      }
      root.innerHTML = "";
      const outer = el("div", { class: "heat-outer" }); outer.appendChild(labels);
      const wrap = el("div", { class: "heatmap-wrap", style: "flex:1;min-width:0" }); wrap.appendChild(svg); outer.appendChild(wrap);
      root.appendChild(outer); wrap.scrollLeft = wrap.scrollWidth;
      root.insertAdjacentHTML("beforeend", `<div class="legend">Less <i style="background:var(--heat-0)"></i><i style="background:var(--heat-1)"></i><i style="background:var(--heat-2)"></i><i style="background:var(--heat-3)"></i><i style="background:var(--heat-4)"></i> More</div>`);
    }
    if (selectRoot) {
      const opts = [["all", "All"], ...HABITS.map(h => [h.key, `${h.emoji} ${h.label}`]), ["arrive", "🏢 Arrival"]];
      selectRoot.innerHTML = `<div class="seg">${opts.map(([v, l]) => `<button type="button" data-v="${v}" class="${v === metric ? "on" : ""}">${esc(l)}</button>`).join("")}</div>`;
      selectRoot.querySelectorAll("button").forEach(b => b.addEventListener("click", () => { metric = b.dataset.v; selectRoot.querySelectorAll("button").forEach(x => x.classList.toggle("on", x === b)); draw(); }));
    }
    draw();
    let rt, lastWeeks = weeksFor();
    const onResize = () => { clearTimeout(rt); rt = setTimeout(() => { const w = weeksFor(); if (w !== lastWeeks) { lastWeeks = w; draw(); } }, 150); };
    if (window.ResizeObserver) new ResizeObserver(onResize).observe(root); else window.addEventListener("resize", onResize);
  }

  // ───────── arrival chart (last 30 days) ─────────
  function renderArriveChart(root) {
    const pts = last30().filter(e => e.arrive != null);
    if (pts.length < 2) { root.innerHTML = `<div class="empty">The trend chart appears once there are arrival times on at least 2 days in the last 30.</div>`; return; }
    const W = 640, H = 220, L = 46, R = 14, T = 14, B = 30;
    const vals = pts.map(p => p.arrive);
    let lo = Math.min(...vals, GOAL), hi = Math.max(...vals, GOAL);
    lo = Math.floor((lo - 20) / 30) * 30; hi = Math.ceil((hi + 20) / 30) * 30;
    const tickStep = hi - lo > 240 ? 60 : 30;
    const x0 = addDays(TODAY, -29);
    const x = d => L + ((d - x0) / 86400000) * (W - L - R) / 29;
    const y = v => T + (hi - v) * (H - T - B) / (hi - lo);
    const svg = svgEl("svg", { class: "chart", viewBox: `0 0 ${W} ${H}` });
    const axis = svgEl("g", { class: "axis" });
    for (let v = lo; v <= hi; v += tickStep) {
      axis.appendChild(svgEl("line", { x1: L, x2: W - R, y1: y(v), y2: y(v), class: "grid-line" }));
      const t = svgEl("text", { x: L - 6, y: y(v) + 3, "text-anchor": "end" }); t.textContent = fmtMin(v); axis.appendChild(t);
    }
    for (let i = 0; i <= 29; i += 5) { const d = addDays(x0, i); const t = svgEl("text", { x: x(d), y: H - 8, "text-anchor": "middle" }); t.textContent = `${d.getMonth() + 1}/${d.getDate()}`; axis.appendChild(t); }
    svg.appendChild(axis);
    svg.appendChild(svgEl("line", { x1: L, x2: W - R, y1: y(GOAL), y2: y(GOAL), class: "goal" }));
    const gl = svgEl("text", { x: W - R, y: y(GOAL) - 4, "text-anchor": "end" }); gl.textContent = `goal ${fmtMin(GOAL)}`; gl.style.fill = "var(--warn)"; gl.style.fontSize = "10px"; svg.appendChild(gl);
    const path = pts.map((p, i) => `${i ? "L" : "M"}${x(p.date).toFixed(1)},${y(p.arrive).toFixed(1)}`).join(" ");
    svg.appendChild(svgEl("path", { d: `${path} L${x(pts[pts.length - 1].date).toFixed(1)},${H - B} L${x(pts[0].date).toFixed(1)},${H - B} Z`, class: "area" }));
    svg.appendChild(svgEl("path", { d: path, class: "line" }));
    pts.forEach(p => {
      const c = svgEl("circle", { cx: x(p.date), cy: y(p.arrive), r: 3.5, class: `dot${p.arrive > GOAL ? " late" : ""}` });
      const t = svgEl("title"); t.textContent = `${p.key} (${WD[p.date.getDay()]}) in at ${fmtMin(p.arrive)}${p.leave != null ? ` · out at ${fmtMin(p.leave)}` : ""}`; c.appendChild(t);
      svg.appendChild(c);
    });
    root.innerHTML = ""; root.appendChild(svg);
    const late = vals.filter(v => v > GOAL).length;
    root.insertAdjacentHTML("beforeend", `<div class="small muted">${pts.length} of the last 30 days logged · avg arrival <b>${fmtMin(avg(vals))}</b> · later than goal on ${late} days</div>`);
  }

  // ───────── weekly summary (this week / last week) + markdown copy ─────────
  function weekRows(entries) {
    const wd = SKIP_WEEKENDS ? entries.filter(e => !isWeekend(e.date)) : entries;
    const arr = wd.filter(e => e.arrive != null);
    const onTime = arr.filter(e => e.arrive <= GOAL).length;
    const rows = [
      ["Days logged", `${entries.length}`],
      [`In by ${fmtMin(GOAL)}`, arr.length ? `${onTime} / ${arr.length}` : "–"],
      ["Avg arrival", fmtMin(avg(arr.map(e => e.arrive)))],
      ["Avg hours in lab", arr.length ? `${fmtNum(avg(wd.map(e => e.hours)))}h` : "–"],
    ];
    HABITS.forEach(h => { const base = habitSkipsWeekend(h.key) ? wd : entries; rows.push([`${h.emoji} ${h.label}`, `${base.filter(hasHabit(h.key)).length} days`]); });
    rows.push(["Avg mood", moodStr(avg(entries.map(e => e.mood)))]);
    rows.push(["Avg sleep", entries.some(e => e.sleep != null) ? `${fmtNum(avg(entries.map(e => e.sleep)))}h` : "–"]);
    return rows;
  }
  function renderWeekly(root) {
    const a = weekRows(thisWeek()), b = weekRows(lastWeek());
    const [mon] = weekRange();
    const head = ["Metric", `This week (${isoWeek(mon)})`, `Last week (${isoWeek(addDays(mon, -7))})`];
    const shortHead = ["Metric", "This wk", "Last wk"];
    root.innerHTML = `<div class="table-caption">${isoWeek(mon)} vs ${isoWeek(addDays(mon, -7))}</div><div style="overflow-x:auto"><table class="summary-table"><thead><tr>${shortHead.map(h => `<th>${esc(h)}</th>`).join("")}</tr></thead><tbody>${a.map((r, i) => `<tr><td>${esc(r[0])}</td><td>${esc(r[1])}</td><td class="muted">${esc(b[i][1])}</td></tr>`).join("")}</tbody></table></div>
      <div class="md-copy"><button type="button" class="btn btn-sm" id="weekly-md">Copy as Markdown for the weekly review</button> <span class="small muted" id="weekly-hint"></span></div>`;
    root.querySelector("#weekly-md").addEventListener("click", () => {
      const md = `| ${head.join(" | ")} |\n|---|---|---|\n` + a.map((r, i) => `| ${r[0]} | ${r[1]} | ${b[i][1]} |`).join("\n") + "\n";
      navigator.clipboard.writeText(md).then(() => { root.querySelector("#weekly-hint").textContent = "Copied ✓"; }).catch(() => { root.querySelector("#weekly-hint").textContent = "Copy failed"; });
    });
  }

  // ───────── monthly summary ─────────
  function renderMonthly(root) {
    const months = {};
    KEYS.forEach(k => { const m = k.slice(0, 7); (months[m] = months[m] || []).push(E[k]); });
    const ms = Object.keys(months).sort().slice(-6).reverse();
    if (!ms.length) { root.innerHTML = `<div class="empty">Monthly comparison appears as entries accumulate.</div>`; return; }
    const th = ["Month", "Logged", "Avg arrival", "On time", "Avg hours", ...HABITS.map(h => h.emoji)];
    const rows = ms.map(m => {
      const es = months[m], wd = SKIP_WEEKENDS ? es.filter(e => !isWeekend(e.date)) : es, arr = wd.filter(e => e.arrive != null);
      const onTime = arr.filter(e => e.arrive <= GOAL).length;
      const cells = [`${MON[+m.slice(5) - 1]} ${m.slice(0, 4)}`, `${es.length}`, fmtMin(avg(arr.map(e => e.arrive))), arr.length ? `${pct(onTime, arr.length)}%` : "–", arr.length ? `${fmtNum(avg(wd.map(e => e.hours)))}h` : "–",
        ...HABITS.map(h => { const base = habitSkipsWeekend(h.key) ? wd : es; const p = pct(base.filter(hasHabit(h.key)).length, base.length); return p == null ? "–" : bar(p); })];
      return `<tr>${cells.map(c => `<td>${c}</td>`).join("")}</tr>`;
    });
    root.innerHTML = `<div style="overflow-x:auto"><table><thead><tr>${th.map(h => `<th title="${esc(HMAP[h]?.label || "")}">${esc(h)}</th>`).join("")}</tr></thead><tbody>${rows.join("")}</tbody></table></div>`;
  }
  const bar = p => `<div class="bar-cell"><div class="bar" style="width:${p}%"></div><span>${p}%</span></div>`;

  // ───────── weekday pattern ─────────
  function renderWeekday(root) {
    if (KEYS.length < 3) { root.innerHTML = `<div class="empty">Weekday patterns appear as entries accumulate.</div>`; return; }
    const byDay = [1, 2, 3, 4, 5, 6, 0].map(dow => ({ dow, es: KEYS.map(k => E[k]).filter(e => e.date.getDay() === dow) }));
    const rows = byDay.map(({ dow, es }) => {
      const arr = es.filter(e => e.arrive != null);
      const cells = [WD[dow], `${es.length}`, fmtMin(avg(arr.map(e => e.arrive))), arr.length ? `${pct(arr.filter(e => e.arrive <= GOAL).length, arr.length)}%` : "–",
        ...HABITS.map(h => { const p = pct(es.filter(hasHabit(h.key)).length, es.length); return p == null ? "–" : bar(p); })];
      return `<tr class="${isWeekend(new Date(2024, 0, 7 + dow)) ? "dim" : ""}">${cells.map(c => `<td>${c}</td>`).join("")}</tr>`;
    });
    const th = ["Day", "Logged", "Avg arrival", "On time", ...HABITS.map(h => h.emoji)];
    root.innerHTML = `<div style="overflow-x:auto"><table class="log-table"><thead><tr>${th.map(h => `<th title="${esc(HMAP[h]?.label || "")}">${esc(h)}</th>`).join("")}</tr></thead><tbody>${rows.join("")}</tbody></table></div>`;
  }

  // ───────── recent log table ─────────
  function renderRecent(root, days = 14) {
    const rows = [];
    for (let i = 0; i < days; i++) {
      const d = addDays(TODAY, -i), k = keyOf(d), e = get(k);
      const dl = `${d.getMonth() + 1}/${d.getDate()} <span class="muted">${WD[d.getDay()]}</span>`;
      if (!e) { rows.push(`<tr class="dim"><td>${dl}</td><td colspan="7" class="small">no entry</td></tr>`); continue; }
      const hab = HABITS.map(h => `<span title="${esc(h.label)}" style="opacity:${e.done.has(h.key) ? 1 : .18}">${h.emoji}</span>`).join(" ");
      rows.push(`<tr style="cursor:pointer" data-k="${k}"><td>${dl}</td>
        <td class="num ${e.arrive != null && e.arrive > GOAL ? "late" : ""}">${fmtMin(e.arrive)}</td>
        <td class="num hide-sm">${fmtMin(e.leave)}</td><td class="num hide-sm">${e.hours != null ? fmtNum(e.hours) + "h" : "–"}</td>
        <td class="emojis">${hab}</td><td class="mood hide-sm">${moodStr(e.mood)}</td>
        <td class="num hide-sm">${e.sleep != null ? fmtNum(e.sleep) + "h" : "–"}</td><td class="note">${esc(e.note)}</td></tr>`);
    }
    root.innerHTML = `<div style="overflow-x:auto"><table class="log-table"><thead><tr><th>Date</th><th>In</th><th class="hide-sm">Out</th><th class="hide-sm">Hours</th><th>Habits</th><th class="hide-sm">Mood</th><th class="hide-sm">Sleep</th><th>Note</th></tr></thead><tbody>${rows.join("")}</tbody></table></div>`;
    root.querySelectorAll("tr[data-k]").forEach(tr => tr.addEventListener("click", () => fillForm(tr.dataset.k)));
  }

  // ───────── log form → prefilled GitHub Issue form ─────────
  let form = null;
  function initForm(root) {
    form = root;
    const habitChecks = HABITS.map(h => `<label class="check"><input type="checkbox" name="done" value="${h.key}"> ${h.emoji} ${esc(h.label)}</label>`).join("");
    root.innerHTML = `
      <div class="form-grid">
        <div class="field"><label for="tr-date">Date</label><input id="tr-date" type="date" name="date" value="${TODAY_KEY}"></div>
        <div class="field"><label for="tr-arrive">Arrived at</label><input id="tr-arrive" type="time" name="arrive" step="300"></div>
        <div class="field"><label for="tr-leave">Left at</label><input id="tr-leave" type="time" name="leave" step="300"></div>
        <fieldset class="field field-wide" style="border:0;padding:0;margin:0"><legend class="small muted" style="padding:0;margin-bottom:.25rem">Done today</legend><div class="checks">${habitChecks}</div></fieldset>
        <div class="field field-wide"><label for="tr-note">Note · one-line retro</label><textarea id="tr-note" name="note" placeholder="What I did today, what's next"></textarea></div>
      </div>
      <details class="more"><summary>More (wake-up · sleep · mood · focus)</summary>
        <div class="form-grid">
          <div class="field"><label for="tr-wake">Woke up at</label><input id="tr-wake" type="time" name="wake" step="300"></div>
          <div class="field"><label for="tr-sleep">Sleep (hours)</label><input id="tr-sleep" type="number" name="sleep" step="0.5" min="0" max="16" placeholder="7"></div>
          <div class="field"><label for="tr-mood">Mood (1–5)</label><select id="tr-mood" name="mood"><option value="">–</option><option value="5">😄 5 great</option><option value="4">🙂 4 good</option><option value="3">😐 3 okay</option><option value="2">😕 2 meh</option><option value="1">😩 1 rough</option></select></div>
          <div class="field"><label for="tr-focus">Focus (hours)</label><input id="tr-focus" type="number" name="focus" step="0.5" min="0" max="16" placeholder="3"></div>
        </div>
      </details>
      <div class="form-actions">
        <button type="button" class="btn btn-primary" id="tr-save">Save to GitHub ↗</button>
        <button type="button" class="btn" id="tr-copy">Copy YAML</button>
        <span class="small muted" id="tr-hint"></span>
      </div>
      <pre class="yaml-preview"><code id="tr-yaml"></code></pre>
      <p class="small muted">Save opens a prefilled GitHub Issue form. Press <b>Submit</b> and the entry is committed automatically and shows up here in a minute or two. Submitting the same date again overwrites it.
      On your phone, add <a href="https://github.com/${REPO}/issues/new?template=${encodeURIComponent(CFG.issueTemplate || "log.yml")}" target="_blank" rel="noopener">this Issue form</a> to your home screen to log without opening the site.
      From a terminal: <code>python scripts/log.py --arrive 9:10 english coding</code>.</p>`;
    root.querySelectorAll(".check input").forEach(i => i.addEventListener("change", () => { i.closest(".check").classList.toggle("on", i.checked); update(); }));
    root.querySelectorAll("input,textarea,select").forEach(i => i.addEventListener("input", update));
    root.querySelector("[name=date]").addEventListener("change", ev => { const e = get(ev.target.value); if (e) fillForm(ev.target.value, true); else update(); });
    root.querySelector("#tr-copy").addEventListener("click", () => {
      navigator.clipboard.writeText(yaml()).then(() => flash("Copied ✓")).catch(() => flash("Copy failed — copy the YAML below manually"));
    });
    root.querySelector("#tr-save").addEventListener("click", () => {
      const d = val("date"); if (!d) return flash("Please enter a date");
      const q = new URLSearchParams({ template: CFG.issueTemplate || "log.yml", title: `log: ${d}`, date: d });
      const done = [...root.querySelectorAll("[name=done]:checked")].map(i => i.value);
      [["arrive", val("arrive").slice(0, 5)], ["leave", val("leave").slice(0, 5)], ["wake", val("wake").slice(0, 5)], ["sleep", val("sleep")], ["mood", val("mood")], ["focus", val("focus")], ["note", val("note")]]
        .forEach(([k, v]) => { if (v) q.set(k, v); });
      if (done.length) q.set("done", done.join(", "));
      window.open(`https://github.com/${REPO}/issues/new?${q.toString()}`, "_blank", "noopener");
    });
    root._update = update;
    if (get(TODAY_KEY)) fillForm(TODAY_KEY, true); else update();

    function val(n) { const i = root.querySelector(`[name=${n}]`); return i ? i.value.trim() : ""; }
    function yaml() {
      const lines = [];
      if (val("arrive")) lines.push(`arrive: "${val("arrive").slice(0, 5)}"`);
      if (val("leave")) lines.push(`leave: "${val("leave").slice(0, 5)}"`);
      if (val("wake")) lines.push(`wake: "${val("wake").slice(0, 5)}"`);
      if (val("sleep")) lines.push(`sleep: ${+val("sleep")}`);
      if (val("mood")) lines.push(`mood: ${+val("mood")}`);
      if (val("focus")) lines.push(`focus: ${+val("focus")}`);
      lines.push(`done: [${[...root.querySelectorAll("[name=done]:checked")].map(i => i.value).join(", ")}]`);
      if (val("note")) lines.push(`note: ${JSON.stringify(val("note"))}`);
      return lines.join("\n") + "\n";
    }
    function update() {
      root.querySelector("#tr-yaml").textContent = yaml();
      const d = val("date");
      root.querySelector("#tr-save").textContent = get(d) ? "Save to GitHub (overwrite) ↗" : "Save to GitHub ↗";
      root.querySelector("#tr-hint").textContent = `→ _data/days/${d || "YYYY-MM-DD"}.yml`;
    }
    function flash(msg) { root.querySelector("#tr-hint").textContent = msg; setTimeout(update, 4000); }
  }
  function fillForm(k, silent) {
    if (!form) return;
    const e = get(k); if (!e) return;
    const set = (n, v) => { const i = form.querySelector(`[name=${n}]`); if (i) i.value = v ?? ""; };
    set("date", k); set("arrive", e.arrive != null ? fmtMin(e.arrive) : ""); set("leave", e.leave != null ? fmtMin(e.leave) : "");
    set("wake", e.wake != null ? fmtMin(e.wake) : ""); set("sleep", e.sleep ?? ""); set("mood", e.mood ?? ""); set("focus", e.focus ?? ""); set("note", e.note);
    form.querySelectorAll("[name=done]").forEach(i => { i.checked = e.done.has(i.value); i.closest(".check").classList.toggle("on", i.checked); });
    if (e.wake != null || e.sleep != null || e.mood != null || e.focus != null) form.querySelector("details.more").open = true;
    form._update();
    if (!silent) form.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  // ───────── CSV export ─────────
  function initExport(btn) {
    btn.addEventListener("click", () => {
      const head = ["date", "weekday", "arrive", "leave", "hours", ...HKEYS, "wake", "sleep", "mood", "focus", "note"];
      const cell = v => `"${String(v ?? "").replace(/"/g, '""').replace(/–/g, "")}"`;
      const rows = KEYS.map(k => { const e = E[k]; return [k, WD[e.date.getDay()], fmtMin(e.arrive), fmtMin(e.leave), e.hours != null ? fmtNum(e.hours, 2) : "", ...HKEYS.map(h => e.done.has(h) ? 1 : 0), fmtMin(e.wake), e.sleep, e.mood, e.focus, e.note].map(cell).join(","); });
      const blob = new Blob(["﻿" + [head.join(","), ...rows].join("\n")], { type: "text/csv;charset=utf-8" });
      const a = el("a", { href: URL.createObjectURL(blob), download: `routine-${TODAY_KEY}.csv` }); document.body.appendChild(a); a.click(); a.remove();
    });
  }

  // ───────── mount ─────────
  function mount() {
    const $ = id => document.getElementById(id);
    const empty = KEYS.length === 0;
    if ($("onboarding")) $("onboarding").hidden = !empty;
    if ($("data-sections")) $("data-sections").hidden = empty;
    if ($("today-bar")) renderToday($("today-bar"));
    if ($("stats")) { $("stats").hidden = empty; if (!empty) renderStats($("stats")); }
    if ($("log-form")) initForm($("log-form"));
    if ($("heatmap")) renderHeatmap($("heatmap"), $("heatmap-select"));
    if ($("arrive-chart")) renderArriveChart($("arrive-chart"));
    if ($("weekly")) renderWeekly($("weekly"));
    if ($("monthly")) renderMonthly($("monthly"));
    if ($("weekday")) renderWeekday($("weekday"));
    if ($("recent-log")) renderRecent($("recent-log"));
    if ($("export-csv")) initExport($("export-csv"));
    if ($("total-days")) $("total-days").textContent = KEYS.length;
  }
  document.addEventListener("DOMContentLoaded", () => {
    fetch((CFG.dataUrl || "/assets/data/days.json") + "?t=" + Date.now(), { cache: "no-store" })   // bypass the 10-minute CDN cache
      .then(r => r.ok ? r.json() : {})
      .catch(() => ({}))
      .then(raw => { load(raw); mount(); });
  });
})();
