# Review website

This is a static, dependency-free project overview intended for review. It can
be hosted from the `website/` directory with GitHub Pages or any static host.
The visual request flow is a browser-only simulation; it does not call the
Python coordinator or generate model tokens.

## Preview locally

From the repository root, run `python3 -m http.server 8080 --directory website`
and open `http://localhost:8080`.

## Publish with GitHub Pages

The workflow in `.github/workflows/pages.yml` publishes this folder when pushed
to the repository's default branch. In the repository settings, set **Pages →
Build and deployment → Source** to **GitHub Actions**. A public deployment
requires a GitHub repository with Pages enabled and the workflow permissions
allowed; no deployment URL can be claimed until that repository-side setup and
workflow run have succeeded.
