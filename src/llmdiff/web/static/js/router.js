// Hash router for the dashboard SPA.
"use strict";

const Router = {
  routes: {},

  register(name, handler) {
    this.routes[name] = handler;
  },

  current() {
    const hash = window.location.hash.replace(/^#\/?/, "");
    return hash.split("?")[0] || "dashboard";
  },

  navigate(name) {
    window.location.hash = "#/" + name;
  },

  start() {
    window.addEventListener("hashchange", () => this._dispatch());
    this._dispatch();
  },

  _dispatch() {
    const name = this.current();
    Store.set("route", name);
    const handler = this.routes[name] || this.routes.dashboard;
    if (handler) handler();
    document.querySelectorAll("#nav a").forEach((a) => {
      a.classList.toggle("active", a.dataset.route === name);
    });
  },
};
