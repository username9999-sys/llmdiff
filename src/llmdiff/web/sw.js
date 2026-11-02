// Service worker: cache the app shell for offline use.
"use strict";

const CACHE = "llmdiff-v1";
const SHELL = [
  "./index.html",
  "./static/css/app.css",
  "./static/js/api.js",
  "./static/js/store.js",
  "./static/js/router.js",
  "./static/js/components.js",
  "./static/js/charts.js",
  "./static/js/views.js",
  "./static/js/app.js",
];

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open(CACHE).then((c) => c.addAll(SHELL)).then(() => self.skipWaiting())
  );
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches
      .keys()
      .then((keys) => Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});

self.addEventListener("fetch", (event) => {
  if (event.request.method !== "GET") return;
  event.respondWith(
    caches.match(event.request).then(
      (hit) =>
        hit ||
        fetch(event.request)
          .then((resp) => {
            if (resp.ok && event.request.url.startsWith(self.location.origin)) {
              const copy = resp.clone();
              caches.open(CACHE).then((c) => c.put(event.request, copy));
            }
            return resp;
          })
          .catch(() => caches.match("./index.html"))
    )
  );
});
