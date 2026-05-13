# GitHub Actions workflows

- **`reproduce.yml`** — runs `./reproduce.sh --comp min` on Ubuntu with Python 3.11
  + `uv sync --frozen --no-dev` from `uv.lock`, then verifies the five expected PNGs
  land in `Figures/` and uploads them as a build artifact. Triggers on push and PR
  to `main` (paths-filtered to skip docs-only changes) and on `workflow_dispatch`.
- **`deploy-gh-pages.yml`** — deploys the `gh-pages` branch to GitHub Pages (no CI build of the paper here).
