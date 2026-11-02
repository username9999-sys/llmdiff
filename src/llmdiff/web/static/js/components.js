// Reusable UI components: matrix table, badges, bars, diff viewer.
"use strict";

const Components = {
  esc(s) {
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  },

  badge(verdict) {
    return `<span class="badge ${this.esc(verdict)}">${this.esc(verdict)}</span>`;
  },

  // Stacked win/tie/loss bar. counts = {win, tie, loss}
  bar(counts) {
    const total = counts.win + counts.tie + counts.loss;
    if (total === 0) return '<div class="bar"></div>';
    const w = (counts.win / total) * 100;
    const t = (counts.tie / total) * 100;
    const l = (counts.loss / total) * 100;
    return (
      `<div class="bar" role="img" aria-label="win ${counts.win}, tie ${counts.tie}, loss ${counts.loss}">` +
      `<span class="win" style="width:${w}%"></span>` +
      `<span class="tie" style="width:${t}%"></span>` +
      `<span class="loss" style="width:${l}%"></span>` +
      `</div>`
    );
  },

  // Verdict matrix table from a matrix dict ({dimensions: [...]}).
  matrixTable(matrix) {
    const rows = (matrix.dimensions || [])
      .map((d) => {
        const cls = d.losses > d.wins ? "loss" : d.wins > d.losses ? "win" : "tie";
        return (
          `<tr>` +
          `<td>${this.esc(d.dimension)}</td>` +
          `<td class="win">${d.wins}</td>` +
          `<td class="tie">${d.ties}</td>` +
          `<td class="loss">${d.losses}</td>` +
          `<td>${d.baseline_mean.toFixed(3)}</td>` +
          `<td>${d.candidate_mean.toFixed(3)}</td>` +
          `<td class="${cls}">${d.delta >= 0 ? "+" : ""}${d.delta.toFixed(3)}</td>` +
          `<td>${this.bar({ win: d.wins, tie: d.ties, loss: d.losses })}</td>` +
          `</tr>`
        );
      })
      .join("");
    return (
      `<table class="matrix"><thead><tr>` +
      `<th>dimension</th><th>win</th><th>tie</th><th>loss</th>` +
      `<th>baseline</th><th>candidate</th><th>delta</th><th></th>` +
      `</tr></thead><tbody>${rows}</tbody></table>`
    );
  },

  // Simple LCS-based line diff rendered as add/del/context lines.
  diffView(baseline, candidate) {
    const a = String(baseline).split("\n");
    const b = String(candidate).split("\n");
    const n = a.length;
    const m = b.length;
    const cap = 2000;
    if (n * m > cap * cap) {
      return `<pre class="diff"><span class="ctx">diff too large (${n}x${m} lines)</span></pre>`;
    }
    const dp = Array.from({ length: n + 1 }, () => new Array(m + 1).fill(0));
    for (let i = n - 1; i >= 0; i--) {
      for (let j = m - 1; j >= 0; j--) {
        dp[i][j] = a[i] === b[j] ? dp[i + 1][j + 1] + 1 : Math.max(dp[i + 1][j], dp[i][j + 1]);
      }
    }
    const lines = [];
    let i = 0;
    let j = 0;
    while (i < n && j < m) {
      if (a[i] === b[j]) {
        lines.push(`<span class="ctx"> ${this.esc(a[i])}</span>`);
        i++;
        j++;
      } else if (dp[i + 1][j] >= dp[i][j + 1]) {
        lines.push(`<span class="del">-${this.esc(a[i])}</span>`);
        i++;
      } else {
        lines.push(`<span class="add">+${this.esc(b[j])}</span>`);
        j++;
      }
    }
    while (i < n) lines.push(`<span class="del">-${this.esc(a[i++])}</span>`);
    while (j < m) lines.push(`<span class="add">+${this.esc(b[j++])}</span>`);
    return `<pre class="diff">${lines.join("\n")}</pre>`;
  },

  card(num, label, cls = "") {
    return `<div class="card"><span class="num ${cls}">${this.esc(num)}</span><span class="lbl">${this.esc(label)}</span></div>`;
  },
};
