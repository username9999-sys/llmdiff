// View renderers for each route.
"use strict";

const Views = {
  el: document.getElementById("app"),
  status: document.getElementById("status-line"),

  setStatus(msg) {
    if (this.status) this.status.textContent = msg;
  },

  render(html) {
    this.el.innerHTML = html;
  },

  async dashboard() {
    this.render(`<div class="panel"><h2>Dashboard</h2><p class="muted">Loading runs&hellip;</p></div>`);
    try {
      const data = await Api.listRuns(50);
      Store.update({ runs: data.runs, runCount: data.count });
      const runs = data.runs;
      const latest = runs[0];
      const cards =
        `<div class="cards">` +
        Components.card(data.count, "runs") +
        (latest
          ? Components.card(latest.result.matrix.wins, "wins", "win") +
            Components.card(latest.result.matrix.ties, "ties", "tie") +
            Components.card(latest.result.matrix.losses, "losses", "loss") +
            Components.card(latest.result.matrix.verdict, "latest verdict")
          : Components.card("-", "no runs yet")) +
        `</div>`;
      const trend =
        runs.length > 1
          ? `<div class="panel mt"><h2>Loss-rate trend</h2>${Charts.line(
              runs.slice().reverse().map((r) => r.result.matrix.regression_rate)
            )}</div>`
          : "";
      const table = latest
        ? `<div class="panel mt"><h2>Latest run matrix</h2>${Components.matrixTable(latest.result.matrix)}</div>`
        : "";
      this.render(
        `<div class="panel"><h2>Overview</h2>${cards}</div>${trend}${table}` +
          `<p class="muted mt">Tip: open <a href="#/runs">Runs</a> to inspect every run.</p>`
      );
      this.setStatus("ok");
    } catch (err) {
      this.render(`<div class="panel"><h2>Dashboard</h2><p class="loss">${Components.esc(err.message)}</p></div>`);
      this.setStatus("api error");
    }
  },

  async runs() {
    this.render(`<div class="panel"><h2>Runs</h2><p class="muted">Loading&hellip;</p></div>`);
    try {
      const data = await Api.listRuns(100);
      Store.update({ runs: data.runs, runCount: data.count });
      const rows = data.runs
        .map((r) => {
          const m = r.result.matrix;
          return (
            `<tr><td><a href="#/run/${Components.esc(r.id)}">${Components.esc(r.id)}</a></td>` +
            `<td>${Components.esc(r.created_at)}</td>` +
            `<td>${Components.badge(m.verdict)}</td>` +
            `<td class="win">${m.wins}</td><td class="tie">${m.ties}</td><td class="loss">${m.losses}</td>` +
            `<td>${m.regression_rate.toFixed(3)}</td>` +
            `<td><button data-del="${Components.esc(r.id)}">delete</button></td></tr>`
          );
        })
        .join("");
      this.render(
        `<div class="panel"><h2>Runs (${data.count})</h2>` +
          `<table class="matrix"><thead><tr><th>id</th><th>created</th><th>verdict</th><th>W</th><th>T</th><th>L</th><th>loss rate</th><th></th></tr></thead>` +
          `<tbody>${rows || '<tr><td colspan="8" class="muted">No runs yet.</td></tr>'}</tbody></table></div>`
      );
      this.el.querySelectorAll("button[data-del]").forEach((btn) => {
        btn.addEventListener("click", async () => {
          await Api.deleteRun(btn.dataset.del);
          this.runs();
        });
      });
      this.setStatus("ok");
    } catch (err) {
      this.render(`<div class="panel"><p class="loss">${Components.esc(err.message)}</p></div>`);
    }
  },

  async runDetail(id) {
    this.render(`<div class="panel"><h2>Run ${Components.esc(id)}</h2><p class="muted">Loading&hellip;</p></div>`);
    try {
      const run = await Api.getRun(id);
      Store.set("currentRun", run);
      const m = run.result.matrix;
      const cases = (run.result.cases || [])
        .filter((c) => c.losses > 0)
        .slice(0, 20)
        .map((c) => {
          const first = c.results.find((r) => r.verdict === "loss");
          return `<tr><td>${Components.esc(c.case_id)}</td><td class="loss">${c.losses}</td><td>${Components.esc(first ? first.dimension : "-")}</td></tr>`;
        })
        .join("");
      this.render(
        `<div class="panel"><h2>Run ${Components.esc(id)}</h2>` +
          `<div class="cards">` +
          Components.card(m.verdict, "verdict") +
          Components.card(m.wins, "wins", "win") +
          Components.card(m.ties, "ties", "tie") +
          Components.card(m.losses, "losses", "loss") +
          `</div></div>` +
          `<div class="panel mt"><h2>Verdict matrix</h2>${Components.matrixTable(m)}</div>` +
          `<div class="panel mt"><h2>Regressed cases (top 20)</h2>` +
          `<table class="matrix"><thead><tr><th>case</th><th>losses</th><th>first regressed dim</th></tr></thead><tbody>${cases || '<tr><td colspan="3" class="muted">none</td></tr>'}</tbody></table></div>`
      );
      this.setStatus("ok");
    } catch (err) {
      this.render(`<div class="panel"><p class="loss">${Components.esc(err.message)}</p></div>`);
    }
  },

  async compare() {
    this.render(`<div class="panel"><h2>Compare two runs</h2><p class="muted">Loading&hellip;</p></div>`);
    try {
      const data = await Api.listRuns(100);
      const opts = data.runs
        .map(
          (r) =>
            `<option value="${Components.esc(r.id)}">${Components.esc(r.id)} (${Components.esc(r.created_at)})</option>`
        )
        .join("");
      this.render(
        `<div class="panel"><h2>Compare two runs</h2>` +
          `<div class="row"><label style="flex:1"><span>A</span><select id="cmp-a">${opts}</select></label>` +
          `<label style="flex:1"><span>B</span><select id="cmp-b">${opts}</select></label>` +
          `<button id="cmp-go">Compare</button></div><div id="cmp-out" class="mt"></div></div>`
      );
      this.el.querySelector("#cmp-go").addEventListener("click", async () => {
        const a = await Api.getMatrix(this.el.querySelector("#cmp-a").value);
        const b = await Api.getMatrix(this.el.querySelector("#cmp-b").value);
        const dims = a.dimensions.map((d) => d.dimension);
        const rows = dims
          .map((name) => {
            const da = a.dimensions.find((d) => d.dimension === name);
            const db = b.dimensions.find((d) => d.dimension === name);
            if (!da || !db) return "";
            const delta = db.loss_rate - da.loss_rate;
            const cls = delta > 0 ? "loss" : delta < 0 ? "win" : "tie";
            return `<tr><td>${Components.esc(name)}</td><td>${da.loss_rate.toFixed(3)}</td><td>${db.loss_rate.toFixed(3)}</td><td class="${cls}">${delta >= 0 ? "+" : ""}${delta.toFixed(3)}</td></tr>`;
          })
          .join("");
        this.el.querySelector("#cmp-out").innerHTML =
          `<table class="matrix"><thead><tr><th>dimension</th><th>A loss rate</th><th>B loss rate</th><th>delta</th></tr></thead><tbody>${rows}</tbody></table>`;
      });
      this.setStatus("ok");
    } catch (err) {
      this.render(`<div class="panel"><p class="loss">${Components.esc(err.message)}</p></div>`);
    }
  },

  async corpus() {
    this.render(
      `<div class="panel"><h2>Corpus browser</h2>` +
        `<p class="muted">Paste JSONL corpus records, then evaluate them against the server.</p>` +
        `<label><span>Records (JSONL)</span><textarea id="corpus-input" rows="10" placeholder='{"id":"c1","prompt":"...","baseline_output":"...","candidate_output":"..."}'></textarea></label>` +
        `<label><span>Dimensions (comma separated, empty = all)</span><input type="text" id="corpus-dims"></label>` +
        `<button id="corpus-run">Evaluate</button><div id="corpus-out" class="mt"></div></div>`
    );
    this.el.querySelector("#corpus-run").addEventListener("click", async () => {
      const text = this.el.querySelector("#corpus-input").value;
      const dimsRaw = this.el.querySelector("#corpus-dims").value.trim();
      const records = text
        .split("\n")
        .map((l) => l.trim())
        .filter(Boolean)
        .map((l) => JSON.parse(l));
      const dims = dimsRaw ? dimsRaw.split(",").map((s) => s.trim()).filter(Boolean) : null;
      const out = this.el.querySelector("#corpus-out");
      out.innerHTML = `<p class="muted">Evaluating ${records.length} cases&hellip;</p>`;
      try {
        const run = await Api.createRun(records, dims, Store.get("settings").tolerance);
        out.innerHTML =
          `<p>Run <a href="#/run/${Components.esc(run.id)}">${Components.esc(run.id)}</a> created.</p>` +
          Components.matrixTable(run.result.matrix);
        this.setStatus("run created");
      } catch (err) {
        out.innerHTML = `<p class="loss">${Components.esc(err.message)}</p>`;
      }
    });
  },

  async settings() {
    const s = Store.get("settings");
    this.render(
      `<div class="panel"><h2>Settings</h2>` +
        `<label><span>Default tolerance</span><input type="number" step="0.01" min="0" max="1" id="set-tol" value="${s.tolerance}"></label>` +
        `<label><span>Default dimensions (comma separated, empty = all)</span><input type="text" id="set-dims" value="${s.dimensions ? s.dimensions.join(",") : ""}"></label>` +
        `<button id="set-save">Save</button><span id="set-msg" class="muted"></span></div>`
    );
    this.el.querySelector("#set-save").addEventListener("click", () => {
      const tol = parseFloat(this.el.querySelector("#set-tol").value) || 0.0;
      const dimsRaw = this.el.querySelector("#set-dims").value.trim();
      Store.set("settings", {
        tolerance: tol,
        dimensions: dimsRaw ? dimsRaw.split(",").map((x) => x.trim()).filter(Boolean) : null,
        theme: s.theme,
      });
      try {
        localStorage.setItem("llmdiff-settings", JSON.stringify(Store.get("settings")));
      } catch (_) {}
      this.el.querySelector("#set-msg").textContent = "saved";
    });
    try {
      const saved = JSON.parse(localStorage.getItem("llmdiff-settings") || "null");
      if (saved) Store.set("settings", { ...s, ...saved });
    } catch (_) {}
  },
};
