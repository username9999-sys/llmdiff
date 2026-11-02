# site/

Static landing page for GitHub Pages. No build step, no external runtime
dependencies (plain HTML + CSS, system fonts only).

Preview locally:

```bash
cd site && python3 -m http.server 8000
# open http://127.0.0.1:8000
```

Deploys automatically to GitHub Pages on push to `main` (see
`.github/workflows/pages.yml`), served from this directory.
