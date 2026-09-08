/* Routine tracker — reads /assets/data/days.json and renders stats and charts. No dependencies. */
(function () {
  "use strict";
  const CFG = window.TRACKER_CONFIG || {};
  const HABITS = Array.isArray(CFG.habits) ? CFG.habits : [];
  const HKEYS = HABITS.map(h => h.key);
  const HMAP = Object.fromEntries(HABITS.map(h => [h.key, h]));
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
  const fmt12 = m => { if (m == null) return ""; const h = Math.floor(m / 60) % 24, mi = Math.round(m % 60); return `${h % 12 || 12}:${pad(mi)} ${h < 12 ? "AM" : "PM"}`; };
  // "9:10 AM", "9:10pm", "09:10", "0910", "9" → minutes since midnight, or null
  const parseTime = t => {
    t = String(t || "").trim().toLowerCase(); if (!t) return null;
    const m = t.match(/^(\d{1,2})(?::?(\d{2}))?\s*(a\.?m\.?|p\.?m\.?)?$/); if (!m) return null;
    let h = +m[1]; const mi = +(m[2] || 0), ap = m[3] ? m[3][0] : null;
    if (mi > 59) return null;
    if (ap) { if (h < 1 || h > 12) return null; h = h % 12 + (ap === "p" ? 12 : 0); } else if (h > 23) return null;
    return h * 60 + mi;
  };
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
  const MOOD_LABEL = ["", "rough", "meh", "okay", "good", "great"];
  const moodLabel = m => m == null ? "" : MOOD_LABEL[Math.max(1, Math.min(5, Math.round(m)))];
  const plural = (n, w) => `${n} ${n === 1 ? w : w + "s"}`;
  const habitSkipsWeekend = h => SKIP_WEEKENDS && !(HMAP[h] && HMAP[h].weekends);

  // ───────── data (RAW = what's stored in the private repo; E = derived view incl. LeetCode auto-check) ─────────
  let RAW = {}, E = {}, KEYS = [];
  const LC_DAYS = (() => { const m = {}; ((window.LEETCODE_DATA && window.LEETCODE_DATA.problems) || []).forEach(p => Object.values(p.langs || {}).forEach(l => { if (l.date) m[l.date] = (m[l.date] || 0) + 1; })); return m; })();
  const LC_HABIT = CFG.leetcodeHabit && HMAP[CFG.leetcodeHabit] ? CFG.leetcodeHabit : null;
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
      note: raw.note ? String(raw.note) : "", auto: new Set()
    };
  }
  function rebuild() {
    E = {};
    new Set([...Object.keys(RAW), ...(LC_HABIT ? Object.keys(LC_DAYS) : [])]).forEach(k => {
      if (!/^\d{4}-\d{2}-\d{2}$/.test(k)) return;
      const raw = RAW[k] && typeof RAW[k] === "object" ? RAW[k] : (LC_DAYS[k] ? {} : null);
      if (!raw) return;
      const e = normalize(k, raw);
      if (LC_HABIT && LC_DAYS[k] && !e.done.has(LC_HABIT)) { e.done.add(LC_HABIT); e.auto.add(LC_HABIT); }
      E[k] = e;
    });
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

  // ───────── today bar: one-tap logging (arrival / leave open the clock, habits toggle, mood picks) ─────────
  function renderToday(root) {
    const t = get(TODAY_KEY);
    const chip = (cls, html, attrs = "") => `<button type="button" class="chip quick ${cls}" ${attrs}>${html}</button>`;
    let html = `<span class="muted">${WD[TODAY.getDay()]}, ${MON[TODAY.getMonth()]} ${TODAY.getDate()}</span>`;
    const hasIn = t && t.arrive != null, hasOut = t && t.leave != null;
    html += chip(hasIn ? (t.arrive <= GOAL ? "on" : "on late") : "", hasIn ? `🏢 In <span class="n">${fmt12(t.arrive)}</span>` : "🏢 Arrived…", `data-time="arrive" title="${hasIn ? "Change" : "Set"} today's arrival time"`);
    html += chip(hasOut ? "on" : "", hasOut ? `🏠 Out <span class="n">${fmt12(t.leave)}</span>` : "🏠 Left…", `data-time="leave" title="${hasOut ? "Change" : "Set"} today's leave time"`);
    HABITS.forEach(h => { const on = !!(t && t.done.has(h.key)), auto = !!(t && t.auto.has(h.key)); html += chip(on ? "on" : "", `${h.emoji} ${esc(h.label)}`, auto ? `disabled title="Checked automatically from today's LeetCode solution"` : `data-habit="${h.key}" title="${on ? "Un-check" : "Check"} ${esc(h.label)} for today"`); });
    html += chip(t && t.mood != null ? "on" : "", t && t.mood != null ? `Mood ${moodStr(t.mood)} ${moodLabel(t.mood)}` : "Mood…", `data-mood title="Set today's mood"`);
    const s = streak(logged, SKIP_WEEKENDS);
    if (s > 0) html += `<span class="chip on"><span class="n">🔥 ${s}-day logging streak</span></span>`;
    html += `<span class="small muted" id="quick-status" aria-live="polite"></span>`;
    root.innerHTML = html;
    const status = msg => { const el0 = document.getElementById("quick-status"); if (el0) el0.innerHTML = msg; };
    const quickSave = async f => {
      root.querySelectorAll("button.chip").forEach(b => b.disabled = true); status("Saving…");
      try { await saveEntry(TODAY_KEY, f, false); renderAll(); status("✓ Saved"); }
      catch (err) { setPending(TODAY_KEY, f, false); renderToday(root); status(`<span style="color:var(--danger)">Save failed: ${esc(err.message)}.</span> Kept on this device — <button type="button" class="btn btn-sm" data-retry>Retry</button>`); root.querySelector("[data-retry]")?.addEventListener("click", () => flushPending(true)); }
    };
    root.querySelectorAll("[data-time]").forEach(b => b.addEventListener("click", () => {
      const k = b.dataset.time;
      openClock({ initial: t && t[k] != null ? t[k] : null, title: k === "arrive" ? "Arrived at" : "Left at", doneLabel: "Save", onDone: m => { if (m != null) quickSave({ [k]: fmtMin(m) }); } });
    }));
    root.querySelectorAll("[data-habit]").forEach(b => b.addEventListener("click", () => { const h = b.dataset.habit; quickSave(t && t.done.has(h) ? { remove: [h] } : { done: [h] }); }));
    root.querySelector("[data-mood]")?.addEventListener("click", ev => {
      const seg = el("span", { class: "seg", role: "group", "aria-label": "Mood" }, [5, 4, 3, 2, 1].map(v => `<button type="button" data-v="${v}" class="${t && t.mood === v ? "on" : ""}" title="${MOOD_LABEL[v]}">${moodStr(v)} ${v}</button>`).join(""));
      ev.currentTarget.replaceWith(seg);
      seg.querySelectorAll("button").forEach(x => x.addEventListener("click", () => quickSave({ mood: x.dataset.v })));
    });
  }

  // ───────── stat tiles ─────────
  function renderStats(root) {
    const wk = thisWeek(), mo = thisMonth(), l7 = last7();
    const moArr = mo.filter(e => e.arrive != null);
    const onTime = moArr.filter(e => e.arrive <= GOAL).length;
    const tile = (label, value, sub) => `<div class="stat"><div class="stat-label">${label}</div><div class="stat-value">${value}</div><div class="stat-sub">${sub}</div></div>`;
    const tiles = [
      tile("🌅 Wake-up time", fmt12(avg(mo.map(e => e.wake))) || "–", mo.some(e => e.wake != null) ? `this week ${fmt12(avg(wk.map(e => e.wake))) || "–"} · earliest ${fmt12(Math.min(...mo.filter(e => e.wake != null).map(e => e.wake)))} · this month` : "no wake-up times yet"),
      tile("🏢 Avg arrival this month", fmt12(avg(moArr.map(e => e.arrive))) || "–", moArr.length ? `by ${fmt12(GOAL)} on ${onTime}/${moArr.length} days (${pct(onTime, moArr.length)}%)` : "no data"),
      tile("⏱ Avg hours in lab", `${fmtNum(avg(mo.map(e => e.hours)))}<span class="unit">h</span>`, `avg leave ${fmt12(avg(mo.map(e => e.leave))) || "–"} · this month`),
      tile("🌙 Sleep · Mood", `${fmtNum(avg(mo.map(e => e.sleep)))}<span class="unit">h</span> · ${moodStr(avg(mo.map(e => e.mood)))}<span class="unit">${moodLabel(avg(mo.map(e => e.mood)))}</span>`, "monthly averages")
    ];
    root.innerHTML = `<div class="grid grid-4">${tiles.join("")}</div>`;
    // habit tiles: the value is the streak (or n / goal) and a gauge shows this week's progress toward the weekly target
    const gauge = (n, goal) => { const met = n >= goal; if (goal <= 7) return `<div class="pips ${met ? "met" : ""}" role="progressbar" aria-valuenow="${n}" aria-valuemin="0" aria-valuemax="${goal}">${Array.from({ length: goal }, (_, i) => `<i class="${i < n ? "on" : ""}"></i>`).join("")}</div>`; return `<div class="gauge ${met ? "met" : ""}" role="progressbar" aria-valuenow="${n}" aria-valuemin="0" aria-valuemax="${goal}"><div class="gauge-fill" style="width:${Math.min(100, 100 * n / goal)}%"></div></div>`; };
    const habitTile = (h, value, n, goal) => `<div class="stat"><div class="stat-label">${h.emoji} ${esc(h.label)}</div><div class="stat-value">${value}</div>${gauge(n, goal)}<div class="stat-sub">${n} / ${goal} this week${n >= goal ? " ✓" : ""}</div></div>`;
    const hb = HABITS.map(h => {
      const p = hasHabit(h.key), sk = habitSkipsWeekend(h.key);
      const goal = +h.weekly_goal || (sk ? 5 : 7);            // explicit weekly goal, else every weekday (or every day)
      const n = (sk && !h.weekly_goal ? wk.filter(e => !isWeekend(e.date)) : wk).filter(p).length;
      if (h.weekly_goal) return habitTile(h, `<span class="${n >= goal ? "goal-met" : ""}">${n}</span><span class="unit">/ ${goal}</span>`, n, goal);
      const st = streak(p, sk);
      return habitTile(h, `${st}<span class="unit">${sk ? "weekday" : "day"} streak</span>`, n, goal);
    });
    if (hb.length) root.insertAdjacentHTML("beforeend", `<div class="grid grid-4" style="margin-top:.9rem">${hb.join("")}</div>`);
  }

  // ───────── heatmap ─────────
  let heatMetric = "all";   // survives re-renders after a save
  function renderHeatmap(root, selectRoot) {
    let metric = heatMetric;
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
      if (e.arrive != null) parts.push(`in at ${fmt12(e.arrive)}`);
      const hs = HABITS.filter(h => e.done.has(h.key)).map(h => h.label);
      parts.push(hs.length ? hs.join(", ") : "nothing checked");
      return parts.join(" · ");
    };
    function draw() {
      const weeks = weeksFor();
      const hw = document.getElementById("heatmap-weeks"); if (hw) hw.textContent = `last ${weeks} weeks`;
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
          if (e) { rect.style.cursor = "pointer"; rect.setAttribute("tabindex", "0"); rect.setAttribute("role", "button"); rect.setAttribute("aria-label", `${k}: ${tip(e)}. Load into the form`); rect.addEventListener("click", () => fillForm(k)); rect.addEventListener("keydown", ev => { if (ev.key === "Enter" || ev.key === " ") { ev.preventDefault(); fillForm(k); } }); }
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
      selectRoot.querySelectorAll("button").forEach(b => b.addEventListener("click", () => { metric = heatMetric = b.dataset.v; selectRoot.querySelectorAll("button").forEach(x => x.classList.toggle("on", x === b)); draw(); }));
    }
    draw();
    let rt, lastWeeks = weeksFor();
    const onResize = () => { clearTimeout(rt); rt = setTimeout(() => { const w = weeksFor(); if (w !== lastWeeks) { lastWeeks = w; (root._redraw || draw)(); } }, 150); };
    if (!root._observed) { root._observed = true; if (window.ResizeObserver) new ResizeObserver(onResize).observe(root); else window.addEventListener("resize", onResize); }
    root._redraw = draw;
  }

  // ───────── times chart (last 30 days): arrival, leave and wake-up on one time axis ─────────
  function renderTimesChart(root) {
    const pts = last30().filter(e => e.arrive != null || e.leave != null || e.wake != null);
    if (pts.length < 2) { root.innerHTML = `<div class="empty">The chart appears once at least 2 days in the last 30 have an arrival, leave or wake-up time.</div>`; return; }
    const series = [
      { key: "wake", cls: "wake", label: "Woke up" },
      { key: "arrive", cls: "arrive", label: "Arrived" },
      { key: "leave", cls: "leave", label: "Left" },
    ];
    const W = 640, H = 260, L = 50, R = 14, T = 14, B = 30;
    const vals = pts.flatMap(e => series.map(s => e[s.key]).filter(v => v != null));
    let lo = Math.min(...vals, GOAL), hi = Math.max(...vals, GOAL);
    lo = Math.floor((lo - 30) / 60) * 60; hi = Math.ceil((hi + 30) / 60) * 60;
    const tickStep = hi - lo > 600 ? 120 : 60;
    const x0 = addDays(TODAY, -29);
    const x = d => L + ((d - x0) / 86400000) * (W - L - R) / 29;
    const y = v => T + (hi - v) * (H - T - B) / (hi - lo);
    const svg = svgEl("svg", { class: "chart", viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": `Wake-up, arrival and leave times over the last 30 days` });
    const axis = svgEl("g", { class: "axis" });
    for (let v = lo; v <= hi; v += tickStep) {
      axis.appendChild(svgEl("line", { x1: L, x2: W - R, y1: y(v), y2: y(v), class: "grid-line" }));
      const t = svgEl("text", { x: L - 6, y: y(v) + 3, "text-anchor": "end" }); t.textContent = fmt12(v); axis.appendChild(t);
    }
    for (let i = 0; i <= 29; i += 5) { const d = addDays(x0, i); const t = svgEl("text", { x: x(d), y: H - 8, "text-anchor": "middle" }); t.textContent = `${d.getMonth() + 1}/${d.getDate()}`; axis.appendChild(t); }
    svg.appendChild(axis);
    svg.appendChild(svgEl("line", { x1: L, x2: W - R, y1: y(GOAL), y2: y(GOAL), class: "goal" }));
    const gl = svgEl("text", { x: W - R, y: y(GOAL) - 4, "text-anchor": "end" }); gl.textContent = `arrival goal ${fmt12(GOAL)}`; gl.style.fill = "var(--warn)"; gl.style.fontSize = "10px"; svg.appendChild(gl);
    series.forEach(sr => {
      const ps = pts.filter(e => e[sr.key] != null);
      if (ps.length >= 2) svg.appendChild(svgEl("path", { d: ps.map((e, i) => `${i ? "L" : "M"}${x(e.date).toFixed(1)},${y(e[sr.key]).toFixed(1)}`).join(" "), class: `line line-${sr.cls}` }));
      ps.forEach(e => {
        const late = sr.key === "arrive" && e.arrive > GOAL;
        const c = svgEl("circle", { cx: x(e.date), cy: y(e[sr.key]), r: 3.5, class: `dot dot-${sr.cls}${late ? " late" : ""}` });
        const t = svgEl("title"); t.textContent = `${e.key} (${WD[e.date.getDay()]}) · ${sr.label} ${fmt12(e[sr.key])}${late ? " (after goal)" : ""}`; c.appendChild(t); svg.appendChild(c);
      });
    });
    root.innerHTML = ""; root.appendChild(svg);
    const arr = pts.map(e => e.arrive).filter(v => v != null), late = arr.filter(v => v > GOAL).length;
    const item = sr => { const a = avg(pts.map(e => e[sr.key])); return `<span><i class="sw sw-${sr.cls}"></i> ${sr.label} · avg <b>${fmt12(a) || "–"}</b></span>`; };
    root.insertAdjacentHTML("beforeend", `<div class="legend legend-left">${series.map(item).join(" &nbsp; ")}</div><div class="small muted">${pts.length} of the last 30 days logged${arr.length ? ` · arrived after the goal on ${late} of ${arr.length} days` : ""}</div>`);
  }

  // ───────── sleep & mood chart (last 30 days, two axes) ─────────
  function renderSleepMoodChart(root) {
    const pts = last30().filter(e => e.sleep != null || e.mood != null);
    if (pts.length < 2) { root.innerHTML = `<div class="empty">The sleep and mood chart appears once at least 2 days in the last 30 have sleep hours or a mood.</div>`; return; }
    const W = 640, H = 220, L = 40, R = 36, T = 16, B = 30;
    const x0 = addDays(TODAY, -29), x = d => L + ((d - x0) / 86400000) * (W - L - R) / 29;
    const sleepVals = pts.filter(e => e.sleep != null).map(e => e.sleep);
    const sHi = Math.max(10, Math.ceil((Math.max(...sleepVals, 8) + 1) / 2) * 2), sLo = 0;
    const yS = v => T + (sHi - v) * (H - T - B) / (sHi - sLo);          // left axis: sleep hours
    const yM = v => T + (5 - v) * (H - T - B) / 4;                       // right axis: mood 1..5 (1 at bottom)
    const svg = svgEl("svg", { class: "chart chart-sm", viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Sleep hours and mood over the last 30 days" });
    const axis = svgEl("g", { class: "axis" });
    for (let v = 0; v <= sHi; v += 2) { axis.appendChild(svgEl("line", { x1: L, x2: W - R, y1: yS(v), y2: yS(v), class: "grid-line" })); const t = svgEl("text", { x: L - 6, y: yS(v) + 3, "text-anchor": "end" }); t.textContent = `${v}h`; axis.appendChild(t); }
    for (let v = 1; v <= 5; v++) { const t = svgEl("text", { x: W - R + 6, y: yM(v) + 3, "text-anchor": "start" }); t.textContent = moodStr(v); axis.appendChild(t); }
    for (let i = 0; i <= 29; i += 5) { const d = addDays(x0, i); const t = svgEl("text", { x: x(d), y: H - 8, "text-anchor": "middle" }); t.textContent = `${d.getMonth() + 1}/${d.getDate()}`; axis.appendChild(t); }
    svg.appendChild(axis);
    const series = [
      { key: "sleep", cls: "sleep", y: e => yS(e.sleep), label: "Sleep (h)", fmt: e => `${fmtNum(e.sleep)}h` },
      { key: "mood", cls: "mood", y: e => yM(e.mood), label: "Mood", fmt: e => `${moodStr(e.mood)} ${e.mood}` },
    ];
    series.forEach(sr => {
      const ps = pts.filter(e => e[sr.key] != null);
      if (ps.length >= 2) svg.appendChild(svgEl("path", { d: ps.map((e, i) => `${i ? "L" : "M"}${x(e.date).toFixed(1)},${sr.y(e).toFixed(1)}`).join(" "), class: `line line-${sr.cls}` }));
      ps.forEach(e => { const c = svgEl("circle", { cx: x(e.date), cy: sr.y(e), r: 3.5, class: `dot dot-${sr.cls}` }); const t = svgEl("title"); t.textContent = `${e.key} (${WD[e.date.getDay()]}) ${sr.label}: ${sr.fmt(e)}`; c.appendChild(t); svg.appendChild(c); });
    });
    root.innerHTML = ""; root.appendChild(svg);
    const sAvg = avg(pts.map(e => e.sleep)), mAvg = avg(pts.map(e => e.mood));
    root.insertAdjacentHTML("beforeend", `<div class="legend legend-left"><i class="sw sw-sleep"></i> Sleep, left axis · avg <b>${fmtNum(sAvg) || "–"}h</b> &nbsp; <i class="sw sw-mood"></i> Mood, right axis · avg <b>${mAvg != null ? fmtNum(mAvg) : "–"}</b> ${moodStr(mAvg)}</div>`);
  }

  // ───────── weekly summary (this week / last week) + markdown copy ─────────
  function weekRows(entries) {
    const wd = SKIP_WEEKENDS ? entries.filter(e => !isWeekend(e.date)) : entries;
    const arr = wd.filter(e => e.arrive != null);
    const onTime = arr.filter(e => e.arrive <= GOAL).length;
    const rows = [
      ["Days logged", `${entries.length}`],
      [`In by ${fmt12(GOAL)}`, arr.length ? `${onTime} / ${arr.length}` : "–"],
      ["Avg arrival", fmt12(avg(arr.map(e => e.arrive))) || "–"],
      ["Avg hours in lab", arr.length ? `${fmtNum(avg(wd.map(e => e.hours)))}h` : "–"],
    ];
    HABITS.forEach(h => { const goal = +h.weekly_goal || 0, base = habitSkipsWeekend(h.key) && !goal ? wd : entries, n = base.filter(hasHabit(h.key)).length; rows.push([`${h.emoji} ${h.label}${goal ? ` (goal ${goal}/wk)` : ""}`, goal ? `${n} / ${goal}${n >= goal ? " ✓" : ""}` : plural(n, "day")]); });
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
    const th = ["Month", "Logged", "Avg arrival", "On time", "Avg hours", ...HABITS.map(h => `${h.emoji} <span class="hide-sm">${esc(h.label)}</span>`)];
    const rows = ms.map(m => {
      const es = months[m], wd = SKIP_WEEKENDS ? es.filter(e => !isWeekend(e.date)) : es, arr = wd.filter(e => e.arrive != null);
      const onTime = arr.filter(e => e.arrive <= GOAL).length;
      const cells = [`${MON[+m.slice(5) - 1]} ${m.slice(0, 4)}`, `${es.length}`, fmt12(avg(arr.map(e => e.arrive))) || "–", arr.length ? `${pct(onTime, arr.length)}%` : "–", arr.length ? `${fmtNum(avg(wd.map(e => e.hours)))}h` : "–",
        ...HABITS.map(h => { const base = habitSkipsWeekend(h.key) ? wd : es; const p = pct(base.filter(hasHabit(h.key)).length, base.length); return p == null ? "–" : bar(p); })];
      return `<tr>${cells.map(c => `<td>${c}</td>`).join("")}</tr>`;
    });
    root.innerHTML = `<div style="overflow-x:auto"><table><thead><tr>${th.map(h => `<th>${h}</th>`).join("")}</tr></thead><tbody>${rows.join("")}</tbody></table></div>${habitLegend()}`;
  }
  const habitLegend = () => `<div class="small muted show-sm" style="margin-top:.4rem">${HABITS.map(h => `${h.emoji} ${esc(h.label)}`).join(" · ")}</div>`;
  const bar = p => `<div class="bar-cell"><div class="bar" style="width:${p}%"></div><span>${p}%</span></div>`;

  // ───────── weekday pattern ─────────
  function renderWeekday(root) {
    if (KEYS.length < 3) { root.innerHTML = `<div class="empty">Weekday patterns appear as entries accumulate.</div>`; return; }
    const byDay = [1, 2, 3, 4, 5, 6, 0].map(dow => ({ dow, es: KEYS.map(k => E[k]).filter(e => e.date.getDay() === dow) }));
    const rows = byDay.map(({ dow, es }) => {
      const arr = es.filter(e => e.arrive != null);
      const cells = [WD[dow], `${es.length}`, fmt12(avg(arr.map(e => e.arrive))) || "–", arr.length ? `${pct(arr.filter(e => e.arrive <= GOAL).length, arr.length)}%` : "–",
        ...HABITS.map(h => { const p = pct(es.filter(hasHabit(h.key)).length, es.length); return p == null ? "–" : bar(p); })];
      return `<tr class="${isWeekend(new Date(2024, 0, 7 + dow)) ? "dim" : ""}">${cells.map(c => `<td>${c}</td>`).join("")}</tr>`;
    });
    const th = ["Day", "Logged", "Avg arrival", "On time", ...HABITS.map(h => `${h.emoji} <span class="hide-sm">${esc(h.label)}</span>`)];
    root.innerHTML = `<div style="overflow-x:auto"><table class="log-table"><thead><tr>${th.map(h => `<th>${h}</th>`).join("")}</tr></thead><tbody>${rows.join("")}</tbody></table></div>${habitLegend()}`;
  }

  // ───────── recent log table ─────────
  function renderRecent(root, days = 14) {
    const rows = [];
    for (let i = 0; i < days; i++) {
      const d = addDays(TODAY, -i), k = keyOf(d), e = get(k);
      const dl = `${d.getMonth() + 1}/${d.getDate()} <span class="muted">${WD[d.getDay()]}</span>`;
      if (!e) { rows.push(`<tr class="dim"><td>${dl}</td><td colspan="6" class="small">no entry</td></tr>`); continue; }
      const hab = HABITS.map(h => `<span title="${esc(h.label)}" style="opacity:${e.done.has(h.key) ? 1 : .18}">${h.emoji}</span>`).join(" ");
      rows.push(`<tr style="cursor:pointer" data-k="${k}" tabindex="0" role="button" aria-label="Load ${k} into the form"><td>${dl}</td>
        <td class="num ${e.arrive != null && e.arrive > GOAL ? "late" : ""}">${fmt12(e.arrive) || "–"}</td>
        <td class="num hide-sm">${fmt12(e.leave) || "–"}</td><td class="num hide-sm">${e.hours != null ? fmtNum(e.hours) + "h" : "–"}</td>
        <td class="emojis">${hab}</td><td class="mood hide-sm">${moodStr(e.mood)}</td>
        <td class="num hide-sm">${e.sleep != null ? fmtNum(e.sleep) + "h" : "–"}</td></tr>`);
    }
    root.innerHTML = `<div style="overflow-x:auto"><table class="log-table"><thead><tr><th>Date</th><th>In</th><th class="hide-sm">Out</th><th class="hide-sm">Hours</th><th>Habits</th><th class="hide-sm">Mood</th><th class="hide-sm">Sleep</th></tr></thead><tbody>${rows.join("")}</tbody></table></div>`;
    root.querySelectorAll("tr[data-k]").forEach(tr => { tr.addEventListener("click", () => fillForm(tr.dataset.k)); tr.addEventListener("keydown", ev => { if (ev.key === "Enter" || ev.key === " ") { ev.preventDefault(); fillForm(tr.dataset.k); } }); });
  }

  // ───────── time field + clock-face picker (12-hour, AM/PM) ─────────
  const timeField = n => `<div class="timefield"><input id="tr-${n}" type="text" name="${n}" autocomplete="off" placeholder="h:mm AM"><button type="button" class="clock" data-for="${n}" aria-label="Pick a time">🕒</button></div>`;
  let picker = null;
  function closePicker() { if (!picker) return; picker.el.remove(); if (picker.backdrop) picker.backdrop.remove(); picker = null; document.removeEventListener("click", outsideClick); }
  function outsideClick(ev) { if (!picker || !ev.target.isConnected) return; if (!picker.el.contains(ev.target) && !ev.target.closest(".clock")) closePicker(); }
  // opts: { initial: minutes|null, anchor: input (popover) — or omitted for a centered modal, title, allowClear, doneLabel, onDone(minutes|null) }
  function openClock(opts) {
    closePicker();
    const nowMin = () => { const d = new Date(); return d.getHours() * 60 + d.getMinutes(); };
    const start = opts.initial ?? nowMin();
    let h = Math.floor(start / 60) % 24, mi = start % 60, mode = "h", dragging = false;
    const R = 100, S = 2 * R + 16, C = S / 2;
    const pos = (deg, r) => [C + r * Math.sin(deg * Math.PI / 180), C - r * Math.cos(deg * Math.PI / 180)];
    const box = el("div", { class: "tpick" + (opts.anchor ? "" : " tpick-modal"), role: "dialog", "aria-label": opts.title || "Pick a time" });
    const render = () => {
      const ap = h < 12 ? "AM" : "PM", h12 = h % 12 || 12;
      const sel = mode === "h" ? (h % 12) * 30 : mi * 6, [hx, hy] = pos(sel, 76);
      const labels = mode === "h" ? [12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11].map((n, i) => ({ t: n, deg: i * 30, on: n === h12 }))
                                  : [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55].map(n => ({ t: pad(n), deg: n * 6, on: n === mi }));
      box.innerHTML = `${opts.title ? `<div class="tp-title">${esc(opts.title)}</div>` : ""}
        <div class="tp-row"><span class="tp-cur"><button type="button" class="tp-seg ${mode === "h" ? "on" : ""}" data-mode="h" aria-label="Hour">${h12}</button><span class="tp-colon">:</span><button type="button" class="tp-seg ${mode === "m" ? "on" : ""}" data-mode="m" aria-label="Minute">${pad(mi)}</button></span>
          <span class="seg"><button type="button" data-ap="AM" class="${ap === "AM" ? "on" : ""}">AM</button><button type="button" data-ap="PM" class="${ap === "PM" ? "on" : ""}">PM</button></span></div>
        <svg class="tp-clock" viewBox="0 0 ${S} ${S}" width="${S}" height="${S}">
          <circle cx="${C}" cy="${C}" r="${R}" class="tp-face"/>
          <line x1="${C}" y1="${C}" x2="${hx}" y2="${hy}" class="tp-hand"/>
          <circle cx="${hx}" cy="${hy}" r="17" class="tp-knob"/>
          <circle cx="${C}" cy="${C}" r="3.5" class="tp-center"/>
          ${labels.map(l => { const [x, y] = pos(l.deg, 76); return `<text x="${x}" y="${y}" class="tp-num ${l.on ? "on" : ""}">${l.t}</text>`; }).join("")}
        </svg>
        <div class="tp-hint muted small">${mode === "h" ? "Tap or drag to pick the hour" : "Tap or drag to pick the minute"}</div>
        <div class="tp-row"><span>${opts.allowClear ? `<button type="button" class="btn btn-sm" data-clear>Clear</button> ` : ""}<button type="button" class="btn btn-sm" data-now>Now</button></span><span><button type="button" class="btn btn-sm" data-cancel>Cancel</button> <button type="button" class="btn btn-sm btn-primary" data-done>${esc(opts.doneLabel || "Done")}</button></span></div>`;
    };
    const degFrom = ev => { const r = box.querySelector("svg").getBoundingClientRect(); const x = (ev.clientX - r.left) * S / r.width - C, y = (ev.clientY - r.top) * S / r.height - C; let d = Math.atan2(x, -y) * 180 / Math.PI; return d < 0 ? d + 360 : d; };
    const setFrom = ev => { const d = degFrom(ev); if (mode === "h") h = Math.round(d / 30) % 12 + (h >= 12 ? 12 : 0); else mi = Math.round(d / 6) % 60; render(); };
    const onMove = ev => { if (dragging) setFrom(ev); };
    const endDrag = () => { if (!dragging) return; dragging = false; document.removeEventListener("pointermove", onMove); document.removeEventListener("pointerup", endDrag); document.removeEventListener("pointercancel", endDrag); if (mode === "h") { mode = "m"; render(); } };
    box.addEventListener("pointerdown", ev => { if (!ev.target.closest("svg")) return; ev.preventDefault(); dragging = true; setFrom(ev); document.addEventListener("pointermove", onMove); document.addEventListener("pointerup", endDrag); document.addEventListener("pointercancel", endDrag); });
    box.addEventListener("click", ev => {
      ev.stopPropagation();
      const b = ev.target.closest("button"); if (!b) return;
      if (b.dataset.mode) { mode = b.dataset.mode; render(); }
      else if (b.dataset.ap) { h = h % 12 + (b.dataset.ap === "PM" ? 12 : 0); render(); }
      else if (b.hasAttribute("data-now")) { const n = nowMin(); h = Math.floor(n / 60); mi = n % 60; mode = "m"; render(); }
      else if (b.hasAttribute("data-clear")) { closePicker(); opts.onDone(null); }
      else if (b.hasAttribute("data-cancel")) closePicker();
      else if (b.hasAttribute("data-done")) { closePicker(); opts.onDone(h * 60 + mi); }
    });
    render();
    if (opts.anchor) { opts.anchor.closest(".timefield").appendChild(box); picker = { el: box }; }
    else { const bd = el("div", { class: "tp-backdrop" }); bd.addEventListener("click", closePicker); document.body.appendChild(bd); document.body.appendChild(box); picker = { el: box, backdrop: bd }; }
    setTimeout(() => document.addEventListener("click", outsideClick), 0);
  }
  const fieldClock = input => openClock({ initial: parseTime(input.value), anchor: input, allowClear: true, onDone: m => { input.value = m == null ? "" : fmt12(m); input.classList.remove("bad"); input.dispatchEvent(new Event("input", { bubbles: true })); } });

  // ───────── private storage: GitHub Contents API with a token stored only in this browser ─────────
  const TOKEN_KEY = "gh_token", USER_KEY = "gh_user", CACHE_KEY = "tracker_cache";
  const lsGet = k => { try { return localStorage.getItem(k); } catch { return null; } };
  const lsSet = (k, v) => { try { v == null ? localStorage.removeItem(k) : localStorage.setItem(k, v); } catch {} };
  const token = () => lsGet(TOKEN_KEY) || "";
  const DREPO = CFG.trackerRepo || "", DBRANCH = CFG.trackerBranch || "main", DFILE = CFG.trackerFile || "days.json";
  let fileSha = null;
  async function gh(path, opts = {}) {
    const r = await fetch(`https://api.github.com${path}`, { ...opts, cache: "no-store", headers: { Authorization: `Bearer ${token()}`, Accept: "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28", ...(opts.body ? { "Content-Type": "application/json" } : {}), ...(opts.headers || {}) } });
    if (!r.ok) { let msg = `${r.status}`; try { msg += " " + ((await r.json()).message || ""); } catch {} const e = new Error(msg); e.status = r.status; throw e; }
    return r.json();
  }
  const b64enc = str => btoa(unescape(encodeURIComponent(str)));
  const b64dec = b => decodeURIComponent(escape(atob(b.replace(/\n/g, ""))));
  async function loadRemote() {
    try { const r = await gh(`/repos/${DREPO}/contents/${DFILE}?ref=${DBRANCH}`); fileSha = r.sha; RAW = JSON.parse(b64dec(r.content) || "{}") || {}; }
    catch (e) { if (e.status === 404) { fileSha = null; RAW = {}; } else throw e; }   // 404: file not created yet
    lsSet(CACHE_KEY, JSON.stringify({ at: Date.now(), raw: RAW }));
    rebuild();
  }
  // Merge one day's changes: times/numbers replaced when given, habits added (or removed via `remove`), notes appended.
  function mergeInto(raw, f, replace) {
    const cur = replace ? {} : { ...(raw || {}) };
    ["arrive", "leave", "wake"].forEach(k => { if (f[k]) cur[k] = f[k]; });
    ["sleep", "mood"].forEach(k => { if (f[k] !== undefined && f[k] !== null && f[k] !== "") cur[k] = +f[k]; });
    let done = Array.isArray(cur.done) ? [...cur.done] : [];
    (f.done || []).forEach(d => { if (!done.includes(d)) done.push(d); });
    (f.remove || []).forEach(d => { done = done.filter(x => x !== d); });
    cur.done = done;
    const n = (f.note || "").trim();
    if (n && !(cur.note || "").includes(n)) cur.note = cur.note ? `${cur.note} · ${n}` : n;
    if (!cur.note) delete cur.note;
    return cur;
  }
  async function writeDays(next, message, attempt = 0) {
    if (!token()) throw new Error("not connected");
    const sorted = Object.fromEntries(Object.keys(next).sort().map(k => [k, next[k]]));
    const body = { message, content: b64enc(JSON.stringify(sorted, null, 1) + "\n"), branch: DBRANCH };
    if (fileSha) body.sha = fileSha;
    try { const r = await gh(`/repos/${DREPO}/contents/${DFILE}`, { method: "PUT", body: JSON.stringify(body) }); fileSha = r.content.sha; }
    catch (e) { if ((e.status === 409 || e.status === 422) && attempt === 0) { await loadRemote(); return writeDays(next, message, 1); } throw e; }
    RAW = sorted; lsSet(CACHE_KEY, JSON.stringify({ at: Date.now(), raw: RAW })); rebuild();
  }
  async function saveEntry(date, f, replace) { return writeDays({ ...RAW, [date]: mergeInto(RAW[date], f, replace) }, `log: ${date}`); }
  async function deleteEntry(date) { const next = { ...RAW }; delete next[date]; return writeDays(next, `delete: ${date}`); }
  // A save that failed (offline, expired token) is kept on this device and retried on the next load or on demand.
  const PENDING_KEY = "tracker_pending";
  const setPending = (date, f, replace) => lsSet(PENDING_KEY, JSON.stringify({ date, f, replace, at: Date.now() }));
  async function flushPending(manual) {
    let p; try { p = JSON.parse(lsGet(PENDING_KEY) || "null"); } catch { p = null; }
    if (!p || !p.date) return false;
    try { await saveEntry(p.date, p.f, p.replace); try { localStorage.removeItem(PENDING_KEY); } catch {} renderAll(); const s = document.getElementById("quick-status"); if (s) s.textContent = `✓ Saved the pending change for ${p.date}`; return true; }
    catch (e) { if (manual) { const s = document.getElementById("quick-status"); if (s) s.innerHTML = `<span style="color:var(--danger)">Still failing: ${esc(e.message)}</span>`; } return false; }
  }
  function renderConnect(root) {
    const user = lsGet(USER_KEY);
    const draw = (msg = "") => {
      root.innerHTML = token()
        ? `<details class="more"><summary>🔓 Private · unlocked${user ? ` as @${esc(user)}` : ""} ${msg ? `· <span class="muted">${esc(msg)}</span>` : ""}</summary>
             <div class="card"><button type="button" class="btn btn-sm" id="gh-remove">Lock</button></div></details>`
        : `<details class="more" open><summary>🔒 Private ${msg ? `· <span class="muted">${esc(msg)}</span>` : ""}</summary>
             <div class="card"><div class="form-actions"><input type="password" id="gh-token" class="search" style="margin:0;flex:1;min-width:200px" placeholder="Password" autocomplete="current-password" aria-label="Password"><button type="button" class="btn btn-primary" id="gh-save">Unlock</button></div></div></details>`;
      root.querySelector("#gh-remove")?.addEventListener("click", () => { lsSet(TOKEN_KEY, null); lsSet(USER_KEY, null); lsSet(CACHE_KEY, null); RAW = {}; rebuild(); draw("locked"); boot(); });
      root.querySelector("#gh-token")?.addEventListener("keydown", ev => { if (ev.key === "Enter") root.querySelector("#gh-save").click(); });
      root.querySelector("#gh-save")?.addEventListener("click", async () => {
        const rawIn = root.querySelector("#gh-token").value;
        const t = rawIn.replace(/[^\x21-\x7E]/g, "");   // keep printable ASCII only: strips spaces, Korean text, "…", curly quotes, zero-width characters
        if (!t) return;
        if (!/^(github_pat_|ghp_|gho_)[A-Za-z0-9_]{20,}$/.test(t)) { draw("wrong password"); return; }
        lsSet(TOKEN_KEY, t);
        try {
          const u = await gh("/user");
          try { await gh(`/repos/${DREPO}`); } catch (e) { throw new Error(`cannot see ${DREPO} (${e.status})`); }
          try { await gh(`/repos/${DREPO}/commits?per_page=1`); } catch (e) { throw new Error(`no write access to ${DREPO} (${e.message})`); }
          lsSet(USER_KEY, u.login); draw("unlocked ✓");
        }
        catch (e) { lsSet(TOKEN_KEY, null); draw(`rejected: ${e.message}`); return; }
        boot();
      });
    };
    draw();
  }

  // ───────── the one log form: date, times, numbers, habits, note; merges into the selected day ─────────
  let form = null, logDate = TODAY_KEY, formDirty = false;
  function renderLogForm(root) {
    form = root;
    const d = parseKey(logDate), isToday = logDate === TODAY_KEY, e = get(logDate);
    const habitChecks = HABITS.map(h => { const on = !!(e && e.done.has(h.key)), auto = !!(e && e.auto.has(h.key)); return `<label class="check ${on ? "on" : ""}${auto ? " auto" : ""}" title="${auto ? "Checked automatically from that day's LeetCode solution" : ""}"><input type="checkbox" name="done" value="${h.key}" ${on ? "checked" : ""} ${auto ? "disabled" : ""}> ${h.emoji} ${esc(h.label)}</label>`; }).join("");
    root.innerHTML = `
      <div class="quick-date"><button type="button" class="btn btn-sm" data-shift="-1" aria-label="Previous day">◀</button><span class="date-pick"><span class="quick-date-label">${isToday ? "Today" : `${WD[d.getDay()]}, ${MON[d.getMonth()]} ${d.getDate()}`} <span class="muted" aria-hidden="true">📅</span></span><input type="date" name="date" value="${logDate}" max="${TODAY_KEY}" aria-label="Date to log" title="Pick a date"></span><button type="button" class="btn btn-sm" data-shift="1" aria-label="Next day" ${isToday ? "disabled" : ""}>▶</button><span class="small muted">${e ? "· saved entry shown below" : "· no entry yet"}</span>${isToday ? "" : `<button type="button" class="btn btn-sm" data-today>Today</button>`}</div>
      <div class="form-grid">
        <div class="field"><label for="tr-arrive">Arrived at</label>${timeField("arrive")}</div>
        <div class="field"><label for="tr-leave">Left at</label>${timeField("leave")}</div>
        <div class="field"><label for="tr-wake">Woke up at</label>${timeField("wake")}</div>
        <div class="field"><label for="tr-sleep">Sleep (hours)</label><input id="tr-sleep" type="number" name="sleep" step="0.5" min="0" max="16" placeholder="7"></div>
        <div class="field"><label for="tr-mood">Mood (1–5)</label><select id="tr-mood" name="mood"><option value="">–</option><option value="5">😄 5 great</option><option value="4">🙂 4 good</option><option value="3">😐 3 okay</option><option value="2">😕 2 meh</option><option value="1">😩 1 rough</option></select></div>
        <fieldset class="field field-wide" style="border:0;padding:0;margin:0"><legend class="small muted" style="padding:0;margin-bottom:.25rem">Done</legend><div class="checks">${habitChecks}</div></fieldset>
      </div>
      <div class="form-actions">
        <button type="button" class="btn btn-primary" id="tr-save">Save</button>
        <span class="small muted" id="tr-hint"></span>
        ${e ? `<button type="button" class="btn btn-sm btn-danger" id="tr-delete" style="margin-left:auto">Delete this day</button>` : ""}
      </div>`;
    // prefill from the stored entry
    const set = (n, v) => { const i = root.querySelector(`[name=${n}]`); if (i) i.value = v ?? ""; };
    if (e) { set("arrive", fmt12(e.arrive)); set("leave", fmt12(e.leave)); set("wake", fmt12(e.wake)); set("sleep", e.sleep ?? ""); set("mood", e.mood ?? ""); }
    formDirty = false;
    const val = n => { const i = root.querySelector(`[name=${n}]`); return i ? i.value.trim() : ""; };
    const tval = n => { const m = parseTime(val(n)); return m == null ? "" : fmtMin(m); };
    const flash = msg => { root.querySelector("#tr-hint").innerHTML = msg; };
    const setDate = k => { if (!/^\d{4}-\d{2}-\d{2}$/.test(k) || k > TODAY_KEY) return; if (formDirty && !confirm("Discard unsaved changes on this day?")) return; logDate = k; renderLogForm(root); };
    root.querySelectorAll("[data-shift]").forEach(b => b.addEventListener("click", () => setDate(keyOf(addDays(d, +b.dataset.shift)))));
    root.querySelector("[name=date]").addEventListener("change", ev => setDate(ev.target.value));
    root.querySelector("[data-today]")?.addEventListener("click", () => setDate(TODAY_KEY));
    root.querySelectorAll("input:not([name=date]),textarea,select").forEach(i => i.addEventListener("input", () => { formDirty = true; }));
    root.querySelectorAll(".timefield input").forEach(i => { i.addEventListener("blur", () => { const m = parseTime(i.value); if (m != null) i.value = fmt12(m); else if (i.value.trim()) i.classList.add("bad"); }); i.addEventListener("input", () => i.classList.remove("bad")); });
    root.querySelectorAll(".timefield .clock").forEach(b => b.addEventListener("click", () => fieldClock(root.querySelector(`[name=${b.dataset.for}]`))));
    root.querySelectorAll(".check input").forEach(i => i.addEventListener("change", () => { i.closest(".check").classList.toggle("on", i.checked); formDirty = true; }));
    root.querySelector("#tr-delete")?.addEventListener("click", async () => {
      if (!confirm(`Delete the entry for ${logDate === TODAY_KEY ? "today" : logDate}? This cannot be undone.`)) return;
      const b = root.querySelector("#tr-delete"); b.disabled = true; flash("Deleting…");
      try { await deleteEntry(logDate); formDirty = false; renderAll(); const h = document.getElementById("tr-hint"); if (h) h.textContent = `✓ Deleted ${logDate}.`; }
      catch (err) { flash(`<span style="color:var(--danger)">Delete failed: ${esc(err.message)}.</span>`); b.disabled = false; }
    });
    root.querySelector("#tr-save").addEventListener("click", async () => {
      const bad = [...root.querySelectorAll(".timefield input")].filter(i => i.value.trim() && parseTime(i.value) == null);
      if (bad.length) return flash(`<span style="color:var(--danger)">Check the time in "${esc(bad[0].previousElementSibling ? bad[0].closest(".field").querySelector("label").textContent : "")}" — use 9:10 AM or 09:10.</span>`);
      const checked = [...root.querySelectorAll("[name=done]:checked")].map(i => i.value);
      const wasOn = e ? [...e.done].filter(k => !e.auto.has(k)) : [];
      const f = { arrive: tval("arrive"), leave: tval("leave"), wake: tval("wake"), sleep: val("sleep"), mood: val("mood"),
                  done: checked, remove: wasOn.filter(k => !checked.includes(k)) };
      // the form shows the whole stored day, so what is on screen is what gets saved (cleared fields are cleared)
      const empty = !f.arrive && !f.leave && !f.wake && !f.sleep && !f.mood && !checked.length;
      if (empty) { if (!e) return flash("Nothing to save yet."); if (!confirm("Everything is empty — delete this day's entry instead?")) return; root.querySelector("#tr-delete")?.click(); return; }
      const b = root.querySelector("#tr-save"); b.disabled = true; flash("Saving…");
      try { await saveEntry(logDate, f, true); formDirty = false; renderAll(); document.getElementById("tr-hint").textContent = `✓ Saved ${logDate === TODAY_KEY ? "today" : logDate}.`; }
      catch (err) {
        const why = err.status === 404 || err.status === 403 ? "No write access to " + esc(DREPO) + "." : err.status === 401 ? "Password no longer works — lock and unlock again." : "Kept on this device; it is retried when the page loads again.";
        if (!(err.status === 401 || err.status === 403 || err.status === 404)) setPending(logDate, f, true);
        flash(`<span style="color:var(--danger)">Save failed: ${esc(err.message)}.</span> ${why} <button type="button" class="btn btn-sm" data-retry>Retry</button>`);
        root.querySelector("[data-retry]")?.addEventListener("click", () => b.click());
        b.disabled = false;
      }
    });
  }
  function fillForm(k, silent) {
    if (!form || !k || !/^\d{4}-\d{2}-\d{2}$/.test(k) || k > TODAY_KEY) return;
    if (k !== logDate) { if (formDirty && !confirm("Discard unsaved changes on this day?")) return; logDate = k; renderLogForm(form); }
    if (!silent) form.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  // ───────── CSV export ─────────
  function initExport(btn) {
    btn.addEventListener("click", () => {
      const head = ["date", "weekday", "arrive", "leave", "hours", ...HKEYS, "wake", "sleep", "mood"];
      const cell = v => `"${String(v ?? "").replace(/"/g, '""').replace(/–/g, "")}"`;
      const rows = KEYS.map(k => { const e = E[k]; return [k, WD[e.date.getDay()], fmtMin(e.arrive), fmtMin(e.leave), e.hours != null ? fmtNum(e.hours, 2) : "", ...HKEYS.map(h => e.done.has(h) ? 1 : 0), fmtMin(e.wake), e.sleep, e.mood].map(cell).join(","); });
      const blob = new Blob(["﻿" + [head.join(","), ...rows].join("\n")], { type: "text/csv;charset=utf-8" });
      const a = el("a", { href: URL.createObjectURL(blob), download: `routine-${TODAY_KEY}.csv` }); document.body.appendChild(a); a.click(); a.remove();
    });
  }

  // ───────── public heatmap (home): per-day habit counts from _data/heatmap.json + LeetCode days, no token needed ─────────
  function renderPublicHeatmap(root) {
    const pub = window.PUBLIC_HEATMAP || {}; const days = pub.days || {};
    const nH = Math.max(1, HKEYS.length);
    const countOf = k => { const d = days[k]; let n = d ? d.n : 0; if (LC_HABIT && LC_DAYS[k] && !(d && d.c)) n += 1; return d || LC_DAYS[k] ? n : null; };
    const anyData = Object.keys(days).length || Object.keys(LC_DAYS).length;
    const note = document.getElementById("pub-heat-note");
    if (!anyData) { root.innerHTML = `<div class="empty">No activity data yet.</div>`; if (note) note.textContent = ""; return; }
    const cell = 12, gap = 3, step = cell + gap, left = 22, top = 18;
    const w = root.clientWidth, weeks = w < 200 ? 26 : Math.max(8, Math.min(26, Math.floor((w - 40 - left) / step)));
    const start = addDays(TODAY, -((TODAY.getDay() + 6) % 7) - (weeks - 1) * 7), H = top + 7 * step + 4;
    const labels = svgEl("svg", { class: "heat-labels", width: left, height: H, viewBox: `0 0 ${left} ${H}` });
    [["Mon", 0], ["Wed", 2], ["Fri", 4], ["Sun", 6]].forEach(([t, r]) => { const x = svgEl("text", { x: 0, y: top + r * step + cell - 2 }); x.textContent = t; labels.appendChild(x); });
    const svg = svgEl("svg", { class: "heatmap", width: weeks * step, height: H, viewBox: `0 0 ${weeks * step} ${H}` });
    let lastMonth = -1, logged = 0, streakN = 0;
    for (let c = 0; c < weeks; c++) {
      const mon = addDays(start, c * 7);
      if (mon.getMonth() !== lastMonth) { if (c > 0 || addDays(mon, 6).getMonth() === mon.getMonth()) { const fits = c * step + 24 <= weeks * step; const t = svgEl("text", fits ? { x: c * step, y: 10 } : { x: weeks * step, y: 10, "text-anchor": "end" }); t.textContent = MON[mon.getMonth()]; svg.appendChild(t); } lastMonth = mon.getMonth(); }
      for (let r = 0; r < 7; r++) {
        const d = addDays(mon, r); if (d > TODAY) continue;
        const k = keyOf(d), n = countOf(k);
        if (n != null) logged++;
        const lv = n == null ? 0 : n === 0 ? 1 : Math.min(4, Math.max(1, Math.ceil(n / nH * 4)));
        const rect = svgEl("rect", { x: c * step, y: top + r * step, width: cell, height: cell, class: `heat-${lv}${k === TODAY_KEY ? " heat-today" : ""}` });
        const t = svgEl("title"); t.textContent = `${k} (${WD[d.getDay()]}) · ${n == null ? "no entry" : n + " of " + HKEYS.length + " habits"}`; rect.appendChild(t); svg.appendChild(rect);
      }
    }
    { let d = TODAY; if (countOf(keyOf(d)) == null) d = addDays(d, -1); while (countOf(keyOf(d)) != null) { streakN++; d = addDays(d, -1); if (streakN > 5000) break; } }
    root.innerHTML = ""; const outer = el("div", { class: "heat-outer" }); outer.appendChild(labels);
    const wrap = el("div", { class: "heatmap-wrap", style: "flex:1;min-width:0" }); wrap.appendChild(svg); outer.appendChild(wrap); root.appendChild(outer);
    root.insertAdjacentHTML("beforeend", `<div class="legend">Less <i style="background:var(--heat-0)"></i><i style="background:var(--heat-1)"></i><i style="background:var(--heat-2)"></i><i style="background:var(--heat-3)"></i><i style="background:var(--heat-4)"></i> More</div>`);
    if (note) note.textContent = `${logged} days logged in this window · ${streakN}-day streak${pub.updated ? " · synced " + pub.updated.slice(0, 10) : ""}`;
    if (!root._observed) { root._observed = true; let rt; const w0 = weeks; if (window.ResizeObserver) new ResizeObserver(() => { clearTimeout(rt); rt = setTimeout(() => { const ww = root.clientWidth; const nw = ww < 200 ? 26 : Math.max(8, Math.min(26, Math.floor((ww - 40 - left) / step))); if (nw !== w0) renderPublicHeatmap(root); }, 150); }).observe(root); }
  }

  // ───────── mount ─────────
  const $ = id => document.getElementById(id);
  function renderAll() {
    const empty = KEYS.length === 0;
    if ($("onboarding")) $("onboarding").hidden = !empty;
    if ($("data-sections")) $("data-sections").hidden = empty;
    if ($("tracker-count")) $("tracker-count").hidden = false;
    if ($("today-bar")) renderToday($("today-bar"));
    if ($("stats")) { $("stats").hidden = empty; if (!empty) renderStats($("stats")); }
    if ($("log-form") && (!form || !formDirty)) renderLogForm($("log-form"));
    if ($("heatmap")) renderHeatmap($("heatmap"), $("heatmap-select"));
    if ($("times-chart")) renderTimesChart($("times-chart"));
    if ($("sleep-mood-chart")) renderSleepMoodChart($("sleep-mood-chart"));
    if ($("weekly")) renderWeekly($("weekly"));
    if ($("monthly")) renderMonthly($("monthly"));
    if ($("weekday")) renderWeekday($("weekday"));
    if ($("recent-log")) renderRecent($("recent-log"));
    if ($("total-days")) $("total-days").textContent = KEYS.length;
  }
  let exportBound = false;
  async function boot() {
    const connected = !!token();
    if ($("private-notice")) $("private-notice").hidden = connected;
    if ($("tracker-body")) $("tracker-body").hidden = !connected;
    if ($("today-card")) $("today-card").hidden = !connected;   // home: the whole card is owner-only
    if (!connected) {
      if ($("today-bar")) $("today-bar").innerHTML = `<span class="muted">🔒 <a href="/tracker/">Private</a></span>`;
      if ($("tracker-count")) $("tracker-count").hidden = true;
      return;
    }
    if ($("export-csv") && !exportBound) { initExport($("export-csv")); exportBound = true; }
    try { const c = JSON.parse(lsGet(CACHE_KEY) || "null"); if (c && c.raw) { RAW = c.raw; rebuild(); renderAll(); } } catch {}   // show cached data instantly
    try { await loadRemote(); renderAll(); await flushPending(false); }
    catch (e) {
      const msg = e.status === 401 || e.status === 403 ? `Password no longer works (${e.message}). Lock and unlock again.` : `Could not load ${DFILE} from ${DREPO}: ${e.message}`;
      if ($("quick-status")) $("quick-status").innerHTML = `<span style="color:var(--danger)">${esc(msg)}</span>`;
      else if ($("today-bar")) $("today-bar").innerHTML = `<span style="color:var(--danger)">${esc(msg)}</span>`;
    }
  }
  window.addEventListener("beforeunload", ev => { if (formDirty) { ev.preventDefault(); ev.returnValue = ""; } });
  document.addEventListener("DOMContentLoaded", () => { if ($("public-heatmap")) renderPublicHeatmap($("public-heatmap")); if ($("gh-connect")) renderConnect($("gh-connect")); boot(); });
})();
