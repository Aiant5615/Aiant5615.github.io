/* 루틴 트래커 — /assets/data/days.json 을 읽어 통계·차트를 그립니다. 외부 라이브러리 없음. */
(function () {
  "use strict";
  const CFG = window.TRACKER_CONFIG || {};
  const HABITS = Array.isArray(CFG.habits) ? CFG.habits : [];
  const HKEYS = HABITS.map(h => h.key);
  const HMAP = Object.fromEntries(HABITS.map(h => [h.key, h]));
  const REPO = CFG.repo || "";
  const SKIP_WEEKENDS = !!CFG.skipWeekends;

  // ───────── 유틸 ─────────
  const pad = n => String(n).padStart(2, "0");
  const keyOf = d => `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`;
  const parseKey = k => { const [y, m, d] = k.split("-").map(Number); return new Date(y, m - 1, d); };
  const addDays = (d, n) => { const x = new Date(d); x.setDate(x.getDate() + n); return x; };
  const isWeekend = d => d.getDay() === 0 || d.getDay() === 6;
  const WD = ["일", "월", "화", "수", "목", "금", "토"];
  const toMin = v => {
    if (v == null || v === "") return null;
    if (typeof v === "number") return v;               // YAML 이 09:10 을 550 으로 읽은 경우 대비
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

  // ───────── 데이터 ─────────
  let E = {}, KEYS = [];
  function normalize(k, raw) {
    const done = new Set();
    if (Array.isArray(raw.done)) raw.done.forEach(x => done.add(String(x)));
    HKEYS.forEach(h => { if (raw[h] === true) done.add(h); });
    const arrive = toMin(raw.arrive), wake = toMin(raw.wake);
    let leave = toMin(raw.leave);
    let hours = raw.hours != null ? +raw.hours : null;
    if (arrive != null && leave != null) { if (leave < arrive) leave += 1440; hours = (leave - arrive) / 60; }   // 자정 넘긴 퇴근
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

  // ───────── 통계 ─────────
  // pred(entry) 가 참인 연속 일수. skipWk 면 주말은 끊지도 세지도 않음. 오늘이 비어 있으면 어제부터.
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

  // ───────── 오늘 바 (홈/트래커) ─────────
  function renderToday(root) {
    const t = get(TODAY_KEY);
    let html = `<span class="muted">${TODAY.getMonth() + 1}월 ${TODAY.getDate()}일 (${WD[TODAY.getDay()]})</span>`;
    if (!t) {
      html += `<span class="chip">아직 기록 없음</span>`;
      HABITS.forEach(h => html += `<span class="chip">${h.emoji} ${esc(h.label)}</span>`);
      html += `<a class="btn btn-sm btn-primary" href="/tracker/#log">오늘 기록하기</a>`;
    } else {
      if (t.arrive != null) html += `<span class="chip ${t.arrive <= GOAL ? "on" : ""}">🏢 출근 <span class="n">${fmtMin(t.arrive)}</span></span>`;
      HABITS.forEach(h => html += `<span class="chip ${t.done.has(h.key) ? "on" : ""}">${h.emoji} ${esc(h.label)}</span>`);
      if (t.mood != null) html += `<span class="chip">${moodStr(t.mood)}</span>`;
    }
    const s = streak(logged, SKIP_WEEKENDS);
    if (s > 0) html += `<span class="chip on">🔥 <span class="n">${s}</span>일 연속 기록</span>`;
    root.innerHTML = html;
  }

  // ───────── 통계 타일 ─────────
  function renderStats(root) {
    const wk = thisWeek(), mo = thisMonth(), l7 = last7();
    const moArr = mo.filter(e => e.arrive != null);
    const onTime = moArr.filter(e => e.arrive <= GOAL).length;
    const tile = (label, value, sub) => `<div class="stat"><div class="stat-label">${label}</div><div class="stat-value">${value}</div><div class="stat-sub">${sub}</div></div>`;
    const wkLabel = SKIP_WEEKENDS ? "평일 기준" : "";
    const tiles = [
      tile("📝 기록 스트릭", `${streak(logged, SKIP_WEEKENDS)}<span class="unit">일</span>`, `최고 ${bestStreak(logged, SKIP_WEEKENDS)}일 · 이번 주 ${wk.length}일 기록 ${wkLabel}`),
      tile("🏢 이번 달 평균 출근", fmtMin(avg(moArr.map(e => e.arrive))), moArr.length ? `목표 ${fmtMin(GOAL)} 이내 ${onTime}/${moArr.length}일 (${pct(onTime, moArr.length)}%)` : "기록 없음"),
      tile("⏱ 이번 달 평균 체류", `${fmtNum(avg(mo.map(e => e.hours)))}<span class="unit">시간</span>`, `평균 퇴근 ${fmtMin(avg(mo.map(e => e.leave)))}`),
      tile("🌙 수면 · 컨디션 · 집중", `${fmtNum(avg(mo.map(e => e.sleep)))}<span class="unit">h</span> · ${moodStr(avg(mo.map(e => e.mood)))} · ${fmtNum(avg(mo.map(e => e.focus)))}<span class="unit">h</span>`, "이번 달 평균")
    ];
    root.innerHTML = `<div class="grid grid-4">${tiles.join("")}</div>`;
    const hb = HABITS.map(h => {
      const p = hasHabit(h.key), sk = habitSkipsWeekend(h.key);
      const n7 = l7.filter(p).length, m = mo.filter(p).length;
      const moDen = sk ? mo.filter(e => !isWeekend(e.date)).length : mo.length;
      return tile(`${h.emoji} ${esc(h.label)}`, `${streak(p, sk)}<span class="unit">일 연속</span>`, `지난 7일 ${n7}일 · 이번 달 ${pct(m, moDen) ?? "–"}% · 최고 ${bestStreak(p, sk)}일`);
    });
    if (hb.length) root.insertAdjacentHTML("beforeend", `<div class="grid grid-4" style="margin-top:.9rem">${hb.join("")}</div>`);
  }

  // ───────── 히트맵 ─────────
  function renderHeatmap(root, selectRoot) {
    let metric = "all";
    const weeks = 26, cell = 12, gap = 3, step = cell + gap, left = 22, top = 18;
    const level = e => {
      if (!e) return 0;
      if (metric === "all") { if (!HKEYS.length) return 4; const r = [...e.done].filter(k => HMAP[k]).length / HKEYS.length; return r === 0 ? 0 : Math.max(1, Math.ceil(r * 4)); }
      if (metric === "arrive") { if (e.arrive == null) return 0; const d = e.arrive - GOAL; return d <= 0 ? 4 : d <= 30 ? 3 : d <= 60 ? 2 : 1; }
      return e.done.has(metric) ? 4 : 0;
    };
    const tip = e => {
      if (!e) return "기록 없음";
      const parts = [];
      if (e.arrive != null) parts.push(`출근 ${fmtMin(e.arrive)}`);
      const hs = HABITS.filter(h => e.done.has(h.key)).map(h => h.label);
      parts.push(hs.length ? hs.join(", ") : "체크 없음");
      if (e.note) parts.push(e.note);
      return parts.join(" · ");
    };
    function draw() {
      const start = addDays(TODAY, -((TODAY.getDay() + 6) % 7) - (weeks - 1) * 7);   // 시작 주의 월요일
      const H = top + 7 * step + 4;
      const labels = svgEl("svg", { class: "heat-labels", width: left, height: H, viewBox: `0 0 ${left} ${H}` });
      [["월", 0], ["수", 2], ["금", 4], ["일", 6]].forEach(([t, r]) => { const x = svgEl("text", { x: 0, y: top + r * step + cell - 2 }); x.textContent = t; labels.appendChild(x); });
      const svg = svgEl("svg", { class: "heatmap", width: weeks * step, height: H, viewBox: `0 0 ${weeks * step} ${H}` });
      let lastMonth = -1;
      for (let c = 0; c < weeks; c++) {
        const mon = addDays(start, c * 7);
        if (mon.getMonth() !== lastMonth) {
          if (c > 0 || addDays(mon, 6).getMonth() === mon.getMonth()) { const t = svgEl("text", { x: c * step, y: 10 }); t.textContent = `${mon.getMonth() + 1}월`; svg.appendChild(t); }
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
      root.insertAdjacentHTML("beforeend", `<div class="legend">적게 <i style="background:var(--heat-0)"></i><i style="background:var(--heat-1)"></i><i style="background:var(--heat-2)"></i><i style="background:var(--heat-3)"></i><i style="background:var(--heat-4)"></i> 많이</div>`);
    }
    if (selectRoot) {
      const opts = [["all", "전체"], ...HABITS.map(h => [h.key, `${h.emoji} ${h.label}`]), ["arrive", "🏢 출근"]];
      selectRoot.innerHTML = `<div class="seg">${opts.map(([v, l]) => `<button type="button" data-v="${v}" class="${v === metric ? "on" : ""}">${esc(l)}</button>`).join("")}</div>`;
      selectRoot.querySelectorAll("button").forEach(b => b.addEventListener("click", () => { metric = b.dataset.v; selectRoot.querySelectorAll("button").forEach(x => x.classList.toggle("on", x === b)); draw(); }));
    }
    draw();
  }

  // ───────── 출근 시간 차트 (최근 30일) ─────────
  function renderArriveChart(root) {
    const pts = last30().filter(e => e.arrive != null);
    if (pts.length < 2) { root.innerHTML = `<div class="empty">최근 30일 안에 출근 시간 기록이 2일 이상 있으면 추이 차트가 표시됩니다.</div>`; return; }
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
    const gl = svgEl("text", { x: W - R, y: y(GOAL) - 4, "text-anchor": "end" }); gl.textContent = `목표 ${fmtMin(GOAL)}`; gl.style.fill = "var(--warn)"; gl.style.fontSize = "10px"; svg.appendChild(gl);
    const path = pts.map((p, i) => `${i ? "L" : "M"}${x(p.date).toFixed(1)},${y(p.arrive).toFixed(1)}`).join(" ");
    svg.appendChild(svgEl("path", { d: `${path} L${x(pts[pts.length - 1].date).toFixed(1)},${H - B} L${x(pts[0].date).toFixed(1)},${H - B} Z`, class: "area" }));
    svg.appendChild(svgEl("path", { d: path, class: "line" }));
    pts.forEach(p => {
      const c = svgEl("circle", { cx: x(p.date), cy: y(p.arrive), r: 3.5, class: `dot${p.arrive > GOAL ? " late" : ""}` });
      const t = svgEl("title"); t.textContent = `${p.key} (${WD[p.date.getDay()]}) 출근 ${fmtMin(p.arrive)}${p.leave != null ? ` · 퇴근 ${fmtMin(p.leave)}` : ""}`; c.appendChild(t);
      svg.appendChild(c);
    });
    root.innerHTML = ""; root.appendChild(svg);
    const late = vals.filter(v => v > GOAL).length;
    root.insertAdjacentHTML("beforeend", `<div class="small muted">최근 30일 중 ${pts.length}일 기록 · 평균 출근 <b>${fmtMin(avg(vals))}</b> · 목표 초과 ${late}일</div>`);
  }

  // ───────── 주간 요약 (이번 주 / 지난 주) + 마크다운 복사 ─────────
  function weekRows(entries) {
    const wd = SKIP_WEEKENDS ? entries.filter(e => !isWeekend(e.date)) : entries;
    const arr = wd.filter(e => e.arrive != null);
    const onTime = arr.filter(e => e.arrive <= GOAL).length;
    const rows = [
      ["기록한 날", `${entries.length}일`],
      [`${fmtMin(GOAL)} 이전 출근`, arr.length ? `${onTime} / ${arr.length}일` : "–"],
      ["평균 출근", fmtMin(avg(arr.map(e => e.arrive)))],
      ["평균 체류", arr.length ? `${fmtNum(avg(wd.map(e => e.hours)))}h` : "–"],
    ];
    HABITS.forEach(h => { const base = habitSkipsWeekend(h.key) ? wd : entries; rows.push([`${h.emoji} ${h.label}`, `${base.filter(hasHabit(h.key)).length}일`]); });
    rows.push(["평균 컨디션", moodStr(avg(entries.map(e => e.mood)))]);
    rows.push(["평균 수면", entries.some(e => e.sleep != null) ? `${fmtNum(avg(entries.map(e => e.sleep)))}h` : "–"]);
    return rows;
  }
  function renderWeekly(root) {
    const a = weekRows(thisWeek()), b = weekRows(lastWeek());
    const [mon] = weekRange();
    const head = ["항목", `이번 주 (${isoWeek(mon)})`, `지난 주 (${isoWeek(addDays(mon, -7))})`];
    root.innerHTML = `<div style="overflow-x:auto"><table class="summary-table"><thead><tr>${head.map(h => `<th>${esc(h)}</th>`).join("")}</tr></thead><tbody>${a.map((r, i) => `<tr><td>${esc(r[0])}</td><td>${esc(r[1])}</td><td class="muted">${esc(b[i][1])}</td></tr>`).join("")}</tbody></table></div>
      <div class="md-copy"><button type="button" class="btn btn-sm" id="weekly-md">주간 회고용 마크다운 복사</button> <span class="small muted" id="weekly-hint"></span></div>`;
    root.querySelector("#weekly-md").addEventListener("click", () => {
      const md = `| ${head.join(" | ")} |\n|---|---|---|\n` + a.map((r, i) => `| ${r[0]} | ${r[1]} | ${b[i][1]} |`).join("\n") + "\n";
      navigator.clipboard.writeText(md).then(() => { root.querySelector("#weekly-hint").textContent = "복사됨 ✓"; }).catch(() => { root.querySelector("#weekly-hint").textContent = "복사 실패"; });
    });
  }

  // ───────── 월별 요약 ─────────
  function renderMonthly(root) {
    const months = {};
    KEYS.forEach(k => { const m = k.slice(0, 7); (months[m] = months[m] || []).push(E[k]); });
    const ms = Object.keys(months).sort().slice(-6).reverse();
    if (!ms.length) { root.innerHTML = `<div class="empty">기록이 쌓이면 월별 비교가 표시됩니다.</div>`; return; }
    const th = ["월", "기록", "평균 출근", "정시율", "평균 체류", ...HABITS.map(h => h.emoji)];
    const rows = ms.map(m => {
      const es = months[m], wd = SKIP_WEEKENDS ? es.filter(e => !isWeekend(e.date)) : es, arr = wd.filter(e => e.arrive != null);
      const onTime = arr.filter(e => e.arrive <= GOAL).length;
      const cells = [`${m.slice(0, 4)}.${m.slice(5)}`, `${es.length}일`, fmtMin(avg(arr.map(e => e.arrive))), arr.length ? `${pct(onTime, arr.length)}%` : "–", arr.length ? `${fmtNum(avg(wd.map(e => e.hours)))}h` : "–",
        ...HABITS.map(h => { const base = habitSkipsWeekend(h.key) ? wd : es; const p = pct(base.filter(hasHabit(h.key)).length, base.length); return p == null ? "–" : bar(p); })];
      return `<tr>${cells.map(c => `<td>${c}</td>`).join("")}</tr>`;
    });
    root.innerHTML = `<div style="overflow-x:auto"><table><thead><tr>${th.map(h => `<th title="${esc(HMAP[h]?.label || "")}">${esc(h)}</th>`).join("")}</tr></thead><tbody>${rows.join("")}</tbody></table></div>`;
  }
  const bar = p => `<div class="bar-cell"><div class="bar" style="width:${p}%"></div><span>${p}%</span></div>`;

  // ───────── 요일별 패턴 ─────────
  function renderWeekday(root) {
    if (KEYS.length < 3) { root.innerHTML = `<div class="empty">기록이 쌓이면 요일별 패턴이 표시됩니다.</div>`; return; }
    const byDay = [1, 2, 3, 4, 5, 6, 0].map(dow => ({ dow, es: KEYS.map(k => E[k]).filter(e => e.date.getDay() === dow) }));
    const rows = byDay.map(({ dow, es }) => {
      const arr = es.filter(e => e.arrive != null);
      const cells = [WD[dow], `${es.length}일`, fmtMin(avg(arr.map(e => e.arrive))), arr.length ? `${pct(arr.filter(e => e.arrive <= GOAL).length, arr.length)}%` : "–",
        ...HABITS.map(h => { const p = pct(es.filter(hasHabit(h.key)).length, es.length); return p == null ? "–" : bar(p); })];
      return `<tr class="${isWeekend(new Date(2024, 0, 7 + dow)) ? "dim" : ""}">${cells.map(c => `<td>${c}</td>`).join("")}</tr>`;
    });
    const th = ["요일", "기록", "평균 출근", "정시율", ...HABITS.map(h => h.emoji)];
    root.innerHTML = `<div style="overflow-x:auto"><table class="log-table"><thead><tr>${th.map(h => `<th title="${esc(HMAP[h]?.label || "")}">${esc(h)}</th>`).join("")}</tr></thead><tbody>${rows.join("")}</tbody></table></div>`;
  }

  // ───────── 최근 기록 표 ─────────
  function renderRecent(root, days = 14) {
    const rows = [];
    for (let i = 0; i < days; i++) {
      const d = addDays(TODAY, -i), k = keyOf(d), e = get(k);
      const dl = `${d.getMonth() + 1}/${d.getDate()} <span class="muted">${WD[d.getDay()]}</span>`;
      if (!e) { rows.push(`<tr class="dim"><td>${dl}</td><td colspan="7" class="small">기록 없음</td></tr>`); continue; }
      const hab = HABITS.map(h => `<span title="${esc(h.label)}" style="opacity:${e.done.has(h.key) ? 1 : .18}">${h.emoji}</span>`).join(" ");
      rows.push(`<tr style="cursor:pointer" data-k="${k}"><td>${dl}</td>
        <td class="num ${e.arrive != null && e.arrive > GOAL ? "late" : ""}">${fmtMin(e.arrive)}</td>
        <td class="num hide-sm">${fmtMin(e.leave)}</td><td class="num hide-sm">${e.hours != null ? fmtNum(e.hours) + "h" : "–"}</td>
        <td class="emojis">${hab}</td><td class="mood hide-sm">${moodStr(e.mood)}</td>
        <td class="num hide-sm">${e.sleep != null ? fmtNum(e.sleep) + "h" : "–"}</td><td class="note">${esc(e.note)}</td></tr>`);
    }
    root.innerHTML = `<div style="overflow-x:auto"><table class="log-table"><thead><tr><th>날짜</th><th>출근</th><th class="hide-sm">퇴근</th><th class="hide-sm">체류</th><th>습관</th><th class="hide-sm">컨디션</th><th class="hide-sm">수면</th><th>메모</th></tr></thead><tbody>${rows.join("")}</tbody></table></div>`;
    root.querySelectorAll("tr[data-k]").forEach(tr => tr.addEventListener("click", () => fillForm(tr.dataset.k)));
  }

  // ───────── 기록 폼 → GitHub Issue 폼 프리필 ─────────
  let form = null;
  function initForm(root) {
    form = root;
    const habitChecks = HABITS.map(h => `<label class="check"><input type="checkbox" name="done" value="${h.key}"> ${h.emoji} ${esc(h.label)}</label>`).join("");
    root.innerHTML = `
      <div class="form-grid">
        <div class="field"><label>날짜</label><input type="date" name="date" value="${TODAY_KEY}"></div>
        <div class="field"><label>출근</label><input type="time" name="arrive" step="300"></div>
        <div class="field"><label>퇴근</label><input type="time" name="leave" step="300"></div>
        <div class="field field-wide"><label>오늘 한 것</label><div class="checks">${habitChecks}</div></div>
        <div class="field field-wide"><label>메모 · 한 줄 회고</label><textarea name="note" placeholder="오늘 뭘 했고, 내일 뭘 할지"></textarea></div>
      </div>
      <details class="more"><summary>더 기록하기 (기상 · 수면 · 컨디션 · 집중 시간)</summary>
        <div class="form-grid">
          <div class="field"><label>기상 시간</label><input type="time" name="wake" step="300"></div>
          <div class="field"><label>수면 (시간)</label><input type="number" name="sleep" step="0.5" min="0" max="16" placeholder="7"></div>
          <div class="field"><label>컨디션 (1~5)</label><select name="mood"><option value="">–</option><option value="5">😄 5 최고</option><option value="4">🙂 4 좋음</option><option value="3">😐 3 보통</option><option value="2">😕 2 별로</option><option value="1">😩 1 최악</option></select></div>
          <div class="field"><label>집중 시간 (시간)</label><input type="number" name="focus" step="0.5" min="0" max="16" placeholder="3"></div>
        </div>
      </details>
      <div class="form-actions">
        <button type="button" class="btn btn-primary" id="tr-save">GitHub에 저장 ↗</button>
        <button type="button" class="btn" id="tr-copy">YAML 복사</button>
        <span class="small muted" id="tr-hint"></span>
      </div>
      <pre class="yaml-preview"><code id="tr-yaml"></code></pre>
      <p class="small muted">저장 버튼은 GitHub Issue 폼을 내용이 채워진 상태로 엽니다. <b>Submit</b>만 누르면 자동으로 파일이 만들어지고 1~2분 뒤 반영됩니다. 같은 날짜를 다시 제출하면 덮어씁니다.
      휴대폰에서는 <a href="https://github.com/${REPO}/issues/new?template=${encodeURIComponent(CFG.issueTemplate || "log.yml")}" target="_blank" rel="noopener">이 Issue 폼 링크</a>를 홈 화면에 추가해 두면 사이트 없이도 바로 기록할 수 있습니다.
      터미널에서는 <code>python scripts/log.py --arrive 9:10 english coding</code>.</p>`;
    root.querySelectorAll(".check input").forEach(i => i.addEventListener("change", () => { i.closest(".check").classList.toggle("on", i.checked); update(); }));
    root.querySelectorAll("input,textarea,select").forEach(i => i.addEventListener("input", update));
    root.querySelector("[name=date]").addEventListener("change", ev => { const e = get(ev.target.value); if (e) fillForm(ev.target.value, true); else update(); });
    root.querySelector("#tr-copy").addEventListener("click", () => {
      navigator.clipboard.writeText(yaml()).then(() => flash("복사됨 ✓")).catch(() => flash("복사 실패 — 아래 YAML을 직접 복사하세요"));
    });
    root.querySelector("#tr-save").addEventListener("click", () => {
      const d = val("date"); if (!d) return flash("날짜를 입력하세요");
      const q = new URLSearchParams({ template: CFG.issueTemplate || "log.yml", title: `log: ${d}`, date: d });
      const done = [...root.querySelectorAll("[name=done]:checked")].map(i => i.value);
      [["arrive", val("arrive").slice(0, 5)], ["leave", val("leave").slice(0, 5)], ["wake", val("wake").slice(0, 5)], ["sleep", val("sleep")], ["mood", val("mood")], ["focus", val("focus")], ["note", val("note")]]
        .forEach(([k, v]) => { if (v) q.set(k, v); });
      if (done.length) q.set("done", done.join(", "));
      window.open(`https://github.com/${REPO}/issues/new?${q.toString()}`, "_blank", "noopener");
    });
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
      root.querySelector("#tr-save").textContent = get(d) ? "GitHub에 저장 (덮어쓰기) ↗" : "GitHub에 저장 ↗";
      root.querySelector("#tr-hint").textContent = `→ _data/days/${d || "YYYY-MM-DD"}.yml`;
    }
    function flash(msg) { root.querySelector("#tr-hint").textContent = msg; setTimeout(update, 4000); }
    root._update = update;
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

  // ───────── CSV 내보내기 ─────────
  function initExport(btn) {
    btn.addEventListener("click", () => {
      const head = ["date", "weekday", "arrive", "leave", "hours", ...HKEYS, "wake", "sleep", "mood", "focus", "note"];
      const cell = v => `"${String(v ?? "").replace(/"/g, '""').replace(/–/g, "")}"`;
      const rows = KEYS.map(k => { const e = E[k]; return [k, WD[e.date.getDay()], fmtMin(e.arrive), fmtMin(e.leave), e.hours != null ? fmtNum(e.hours, 2) : "", ...HKEYS.map(h => e.done.has(h) ? 1 : 0), fmtMin(e.wake), e.sleep, e.mood, e.focus, e.note].map(cell).join(","); });
      const blob = new Blob(["﻿" + [head.join(","), ...rows].join("\n")], { type: "text/csv;charset=utf-8" });
      const a = el("a", { href: URL.createObjectURL(blob), download: `routine-${TODAY_KEY}.csv` }); document.body.appendChild(a); a.click(); a.remove();
    });
  }

  // ───────── 마운트 ─────────
  function mount() {
    const $ = id => document.getElementById(id);
    if ($("today-bar")) renderToday($("today-bar"));
    if ($("stats")) renderStats($("stats"));
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
    fetch(CFG.dataUrl || "/assets/data/days.json", { cache: "no-cache" })
      .then(r => r.ok ? r.json() : {})
      .catch(() => ({}))
      .then(raw => { load(raw); mount(); });
  });
})();
