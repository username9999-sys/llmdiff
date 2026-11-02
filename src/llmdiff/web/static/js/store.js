// Minimal pub/sub store for the dashboard.
"use strict";

const Store = {
  state: {
    runs: [],
    runCount: 0,
    currentRun: null,
    settings: { dimensions: null, tolerance: 0.0, theme: "dark" },
    route: "dashboard",
  },
  _subs: {},

  get(key) {
    return this.state[key];
  },

  set(key, value) {
    this.state[key] = value;
    this._emit(key, value);
  },

  update(patch) {
    Object.assign(this.state, patch);
    for (const key of Object.keys(patch)) this._emit(key, this.state[key]);
  },

  subscribe(key, fn) {
    (this._subs[key] = this._subs[key] || []).push(fn);
    return () => {
      this._subs[key] = (this._subs[key] || []).filter((f) => f !== fn);
    };
  },

  _emit(key, value) {
    for (const fn of this._subs[key] || []) fn(value);
  },
};
