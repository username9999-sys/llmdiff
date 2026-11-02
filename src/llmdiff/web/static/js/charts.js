// Inline-SVG chart renderers: stacked bar, line chart, heatmap.
"use strict";

const Charts = {
  esc: Components.esc,

  // Stacked horizontal bar for win/tie/loss counts.
  stackedBar(counts, width = 300, height = 18) {
    const total = counts.win + counts.tie + counts.loss || 1;
    const w = (counts.win / total) * width;
    const t = (counts.tie / total) * width;
    const l = (counts.loss / total) * width;
    return (
      `<svg width="${width}" height="${height}" role="img" aria-label="win ${counts.win} tie ${counts.tie} loss ${counts.loss}">` +
      `<rect x="0" y="0" width="${w}" height="${height}" fill="#69db7c"/>` +
      `<rect x="${w}" y="0" width="${t}" height="${height}" fill="#adb5bd"/>` +
      `<rect x="${w + t}" y="0" width="${l}" height="${height}" fill="#ff6b6b"/>` +
      `</svg>`
    );
  },

  // Line chart of a numeric series.
  line(series, width = 300, height = 120) {
    if (!series.length) return `<svg width="${width}" height="${height}"></svg>`;
    const min = Math.min(...series);
    const max = Math.max(...series);
    const span = max - min || 1;
    const step = width / Math.max(1, series.length - 1);
    const pts = series
      .map(
        (v, i) =>
          `${(i * step).toFixed(1)},${(height - ((v - min) / span) * (height - 10) - 5).toFixed(1)}`
      )
      .join(" ");
    return (
      `<svg width="${width}" height="${height}" role="img" aria-label="line chart of ${series.length} points">` +
      `<polyline points="${pts}" fill="none" stroke="#74c0fc" stroke-width="2"/>` +
      `</svg>`
    );
  },

  // Heatmap: rows = dimensions, cols = cases, cell color from score in [0,1].
  heatmap(dimensions, scores) {
    const cell = 14;
    const rows = dimensions.length;
    const cols = rows ? scores[0].length : 0;
    const w = cols * cell + 120;
    const h = rows * cell + 20;
    let rects = "";
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const v = Math.max(0, Math.min(1, scores[r][c]));
        const red = Math.round(255 * (1 - v));
        const green = Math.round(255 * v);
        rects += `<rect x="${120 + c * cell}" y="${10 + r * cell}" width="${cell - 1}" height="${cell - 1}" fill="rgb(${red},${green},120)"/>`;
      }
      rects += `<text x="0" y="${10 + r * cell + 11}" fill="#8b94a3" font-size="10" font-family="monospace">${this.esc(dimensions[r])}</text>`;
    }
    return `<svg width="${w}" height="${h}" role="img" aria-label="heatmap ${rows}x${cols}">${rects}</svg>`;
  },
};
