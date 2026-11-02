// Tiny fetch wrapper for the llmdiff server API.
"use strict";

const Api = {
  base: "",

  async _req(path, options) {
    const resp = await fetch(this.base + path, options);
    if (!resp.ok) {
      const text = await resp.text().catch(() => "");
      throw new Error(`API ${resp.status}: ${text || resp.statusText}`);
    }
    return resp.json();
  },

  health() {
    return this._req("/api/health");
  },

  listRuns(limit = 100, offset = 0) {
    return this._req(`/api/runs?limit=${limit}&offset=${offset}`);
  },

  getRun(id) {
    return this._req(`/api/runs/${encodeURIComponent(id)}`);
  },

  getMatrix(id) {
    return this._req(`/api/runs/${encodeURIComponent(id)}/matrix`);
  },

  createRun(records, dimensions = null, tolerance = 0.0) {
    return this._req("/api/runs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ records, dimensions, tolerance }),
    });
  },

  deleteRun(id) {
    return this._req(`/api/runs/${encodeURIComponent(id)}`, { method: "DELETE" });
  },
};
