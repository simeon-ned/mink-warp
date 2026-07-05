# Building the documentation

Sphinx site for mink-warp (layout follows [Mink](https://github.com/kevinzakka/mink)).

## Build locally

```bash
uv sync --extra dev --group docs
make docs
```

Output: `docs/_build/index.html`

Live reload (rebuilds on edits under `docs/` **and** `src/`):

```bash
make docs-watch
```

Opens a local server (usually http://127.0.0.1:8000). Stop with Ctrl+C.

## Structure

User-facing pages are **standalone reStructuredText** under `docs/`.

```
docs/
  conf.py
  index.rst
  installation.rst
  references.rst
  source/
    concepts/           # batched design, Mink parity, architecture
    workflows/          # quickstart, batched IK, solvers, CUDA graphs
    tutorial/           # tasks & limits (narrative)
    api/                # live autodoc
    examples.rst
    benchmarks.rst
    roadmap.rst
  _static/
```

Sidebar order: Getting Started → Concepts → User Guide → API Reference → Further Reading.

## API reference (autodoc)

Live signatures come from Google-style docstrings via Sphinx autodoc:

- `docs/source/api/` — curated API index
- Prefer explicit `autoclass` / `autofunction` / `autodata` over blanket `automodule`
- Cross-references in prose use the **same qualified paths** as autodoc, e.g.
  ``:class:`~mink_warp.FrameTask```, ``:class:`~mink_warp.CollisionAvoidanceLimit```
  (top-level public imports, matching ``import mink_warp as mw``).
- After API page changes, run a clean build: ``rm -rf docs/_build && make docs``

When adding a public symbol, document it in source and list it on the matching API page.

## Doc examples

Runnable scripts under `examples/docs/` are included via ``literalinclude`` and
checked by ``tests/test_docs.py``.

## Deploy (optional)

Add a GitHub Actions workflow that runs `make docs` and publishes `docs/_build` to
GitHub Pages on pushes to `main`.
