/* Small SVG chart helpers for the report pages. No libraries. */

// Line chart of mean SHD vs sample size, with 95% CI error bars and the
// individual seed runs drawn as faint jittered dots behind the lines.
//   conditions: grouped summary rows ({ type, lambda, byN })
//   runs:       raw per-run rows for the same algorithm (may be null)
//   sizes:      sorted list of sample sizes
function shdChart(conditions, runs, sizes, title) {
  const width = 480, height = 320;
  const margin = { top: 30, right: 20, bottom: 52, left: 44 };
  const plotW = width - margin.left - margin.right;
  const plotH = height - margin.top - margin.bottom;

  // y scale covers every value the chart will show, plus CI extents
  let yMax = 0;
  for (const cond of conditions) {
    for (const n of sizes) {
      const shd = cond.byN[n]?.shd;
      if (shd) yMax = Math.max(yMax, shd.mean + (shd.ci95 || 0));
    }
  }
  for (const run of runs || []) yMax = Math.max(yMax, run.shd);
  yMax = Math.ceil(yMax * 1.05) || 1;

  const x = n => margin.left + plotW * (sizes.indexOf(n) + 0.5) / sizes.length;
  const y = v => margin.top + plotH * (1 - v / yMax);

  let svg = `<svg viewBox="0 0 ${width} ${height}" width="${width}"
    font-family="Helvetica, Arial, sans-serif" font-size="14">`;
  svg += `<text x="${width / 2}" y="14" text-anchor="middle"
    font-size="16" font-weight="bold">${title}</text>`;

  // horizontal gridlines with axis labels
  const ySteps = 5;
  for (let i = 0; i <= ySteps; i++) {
    const value = yMax * i / ySteps;
    const yy = y(value);
    svg += `<line x1="${margin.left}" y1="${yy}" x2="${width - margin.right}"
      y2="${yy}" stroke="#eee"/>`;
    svg += `<text x="${margin.left - 6}" y="${yy + 3}"
      text-anchor="end">${value.toFixed(0)}</text>`;
  }
  for (const n of sizes) {
    svg += `<text x="${x(n)}" y="${height - margin.bottom + 16}"
      text-anchor="middle">N=${n}</text>`;
  }
  svg += `<text x="12" y="${margin.top - 8}" font-size="13"
    fill="#666">SHD (lower is better)</text>`;

  // faint dots for individual seed runs, jittered per type so overlapping
  // conditions stay distinguishable
  const jitterOrder = t => typeInfo(t).order;
  for (const run of runs || []) {
    const info = typeInfo(run.type);
    const jitter = (jitterOrder(run.type) - 3) * 5 + (run.seed % 5 - 2) * 1.5;
    svg += `<circle cx="${x(run.n_rows) + jitter}" cy="${y(run.shd)}"
      r="2.5" fill="${info.color}" opacity="0.35"/>`;
  }

  // mean lines with CI error bars
  for (const cond of conditions) {
    const info = typeInfo(cond.type);
    const dash = cond.type === "perfect" ? 'stroke-dasharray="5 4"' : "";
    const points = [];
    for (const n of sizes) {
      const shd = cond.byN[n]?.shd;
      if (!shd) continue;
      points.push([x(n), y(shd.mean)]);
      if (shd.ci95 != null) {
        const top = y(shd.mean + shd.ci95);
        const bottom = y(shd.mean - shd.ci95);
        svg += `<line x1="${x(n)}" y1="${top}" x2="${x(n)}" y2="${bottom}"
          stroke="${info.color}" stroke-width="1.3"/>`;
        for (const yy of [top, bottom]) {
          svg += `<line x1="${x(n) - 4}" y1="${yy}" x2="${x(n) + 4}"
            y2="${yy}" stroke="${info.color}" stroke-width="1.3"/>`;
        }
      }
    }
    if (points.length > 1) {
      const path = points.map(p => p.join(",")).join(" ");
      svg += `<polyline points="${path}" fill="none" stroke="${info.color}"
        stroke-width="2" ${dash}/>`;
    }
    for (const [px, py] of points) {
      svg += `<circle cx="${px}" cy="${py}" r="3.2" fill="${info.color}"/>`;
    }
  }

  // legend, wrapped over two lines because there are many conditions now
  let lx = margin.left + 4;
  let ly = height - 22;
  for (const cond of conditions) {
    const info = typeInfo(cond.type);
    const label = cond.lambda == null ? info.label
                : `${info.label} λ=${cond.lambda}`;
    const widthNeeded = 12 + label.length * 7.2 + 12;
    if (lx + widthNeeded > width - margin.right) {   // wrap to next line
      lx = margin.left + 4;
      ly += 12;
    }
    svg += `<rect x="${lx}" y="${ly - 8}" width="9" height="9"
      fill="${info.color}"/>`;
    svg += `<text x="${lx + 12}" y="${ly}">${label}</text>`;
    lx += widthNeeded;
  }
  return svg + "</svg>";
}
