/* Year heatmap shared by the tracker and LeetCode pages: a column of year buttons on the left, the whole year
   (Mon-Fri rows, one column per week) on the right. The caller describes each day through cellFor(key, date). */
(function () {
  "use strict";
  const MON = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
  const pad = n => String(n).padStart(2, "0");
  const keyOf = d => `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`;
  const addDays = (d, n) => { const x = new Date(d); x.setDate(x.getDate() + n); return x; };
  const svgEl = (t, a = {}) => { const e = document.createElementNS("http://www.w3.org/2000/svg", t); for (const k in a) e.setAttribute(k, a[k]); return e; };
  const TODAY = new Date(); TODAY.setHours(0, 0, 0, 0);

  // Draw into root. opts: years (list), year (selected), onYear(y), cellFor(key, date) -> null | {cls, fill, title, click}, legend (html)
  function draw(root, opts) {
    const { years, year, onYear, cellFor, legend = "" } = opts;
    const gap = 3, left = 24, top = 18;
    const jan1 = new Date(year, 0, 1), dec31 = new Date(year, 11, 31);
    const start = addDays(jan1, -((jan1.getDay() + 6) % 7));                        // Monday on or before Jan 1
    const weeks = Math.ceil((dec31 - start) / 864e5 / 7);
    const avail = Math.max(240, root.clientWidth - left - 90);
    const cell = Math.max(8, Math.min(12, Math.floor(avail / weeks) - gap)), step = cell + gap;
    const H = top + 5 * step + 4;
    const labels = svgEl("svg", { class: "heat-labels", width: left, height: H, viewBox: `0 0 ${left} ${H}` });
    [["Mon", 0], ["Wed", 2], ["Fri", 4]].forEach(([t, r]) => { const x = svgEl("text", { x: 0, y: top + r * step + cell - 2 }); x.textContent = t; labels.appendChild(x); });
    const svg = svgEl("svg", { class: "heatmap", width: weeks * step, height: H, viewBox: `0 0 ${weeks * step} ${H}` });
    let lastMonth = -1;
    for (let c = 0; c < weeks; c++) {
      const mon = addDays(start, c * 7);
      const firstInYear = mon.getFullYear() < year ? jan1 : mon;
      if (firstInYear.getMonth() !== lastMonth && addDays(mon, 6) >= jan1) {
        const t = svgEl("text", { x: c * step, y: 10 }); t.textContent = MON[firstInYear.getMonth()]; svg.appendChild(t); lastMonth = firstInYear.getMonth();
      }
      for (let r = 0; r < 5; r++) {                                                   // weekdays only
        const d = addDays(mon, r);
        if (d < jan1 || d > dec31 || d > TODAY) continue;
        const k = keyOf(d), info = cellFor(k, d) || {};
        const rect = svgEl("rect", { x: c * step, y: top + r * step, width: cell, height: cell, class: `${info.cls || "heat-0"}${k === keyOf(TODAY) ? " heat-today" : ""}` });
        if (info.fill) rect.style.fill = info.fill;
        const title = svgEl("title"); title.textContent = info.title || k; rect.appendChild(title);
        if (info.click) { rect.style.cursor = "pointer"; rect.setAttribute("tabindex", "0"); rect.setAttribute("role", "button"); rect.addEventListener("click", info.click); rect.addEventListener("keydown", ev => { if (ev.key === "Enter" || ev.key === " ") { ev.preventDefault(); info.click(); } }); }
        svg.appendChild(rect);
      }
    }
    root.innerHTML = `<div class="heat-year-row"><div class="heat-years">${years.map(y => `<button type="button" data-y="${y}" class="${y === year ? "on" : ""}">${y}</button>`).join("")}</div><div class="heat-outer" style="flex:1;min-width:0"></div></div>${legend ? `<div class="legend">${legend}</div>` : ""}`;
    const outer = root.querySelector(".heat-outer"); outer.appendChild(labels);
    const wrap = document.createElement("div"); wrap.className = "heatmap-wrap"; wrap.style.cssText = "flex:1;min-width:0"; wrap.appendChild(svg); outer.appendChild(wrap);
    root.querySelectorAll(".heat-years button").forEach(b => b.addEventListener("click", () => onYear(+b.dataset.y)));
    if (!root._observed) { root._observed = true; let rt, w0 = root.clientWidth; if (window.ResizeObserver) new ResizeObserver(() => { clearTimeout(rt); rt = setTimeout(() => { if (root.clientWidth !== w0) { w0 = root.clientWidth; (root._redraw || (() => {}))(); } }, 150); }).observe(root); }
  }

  // A small grid of items (problems) in the same cell size: 5 rows, column by column. items: [{title, fill, click}]
  function grid(root, items, cellSize = 12) {
    const gap = 3, step = cellSize + gap, rows = 5, cols = Math.max(1, Math.ceil(items.length / rows));
    const svg = svgEl("svg", { class: "heatmap", width: cols * step, height: rows * step, viewBox: `0 0 ${cols * step} ${rows * step}` });
    items.forEach((it, i) => {
      const c = Math.floor(i / rows), r = i % rows;
      const rect = svgEl("rect", { x: c * step, y: r * step, width: cellSize, height: cellSize, class: it.cls || "heat-0" });
      if (it.fill) rect.style.fill = it.fill;
      const title = svgEl("title"); title.textContent = it.title || ""; rect.appendChild(title);
      if (it.click) { rect.style.cursor = "pointer"; rect.setAttribute("tabindex", "0"); rect.setAttribute("role", "button"); rect.setAttribute("aria-label", it.title || ""); rect.addEventListener("click", it.click); rect.addEventListener("keydown", ev => { if (ev.key === "Enter" || ev.key === " ") { ev.preventDefault(); it.click(); } }); }
      svg.appendChild(rect);
    });
    root.innerHTML = ""; const wrap = document.createElement("div"); wrap.className = "heatmap-wrap"; wrap.appendChild(svg); root.appendChild(wrap);
  }
  window.YearHeat = { draw, grid, keyOf, addDays, TODAY };
})();
