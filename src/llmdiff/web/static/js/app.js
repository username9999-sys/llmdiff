// App entry point: register routes and start the router + service worker.
"use strict";

(function () {
  Router.register("dashboard", () => Views.dashboard());
  Router.register("runs", () => Views.runs());
  Router.register("run", () => {
    const id = Router.current().split("/")[1];
    if (id) Views.runDetail(decodeURIComponent(id));
    else Views.runs();
  });
  Router.register("compare", () => Views.compare());
  Router.register("corpus", () => Views.corpus());
  Router.register("settings", () => Views.settings());

  Router.start();

  if ("serviceWorker" in navigator) {
    navigator.serviceWorker.register("sw.js").catch(() => {});
  }

  Api.health()
    .then(() => Views.setStatus("api ok"))
    .catch(() => Views.setStatus("api unreachable (offline mode)"));
})();
