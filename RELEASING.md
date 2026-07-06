# Releasing

## Pre-release checklist

1. Bump `version` in `pyproject.toml` and `src/mink_warp/__init__.py`.
2. Update `version` and `date-released` in `CITATION.cff`.
3. Commit the version bump, then create an annotated tag:

```sh
git tag -a vX.Y.Z -m "Release vX.Y.Z"
git push origin vX.Y.Z
```

Pushing a `v*` tag triggers `.github/workflows/release.yml` (build → smoke test → PyPI).

## Build and verify locally

```sh
rm -rf dist/
make build
```

This runs `uv build` and smoke-tests the wheel and sdist in isolated environments.

## Test on TestPyPI (optional)

```sh
UV_PUBLISH_TOKEN=<your-testpypi-token> make publish-test
```

Verify with:

```sh
uvx --extra-index-url https://test.pypi.org/simple/ \
    --index-strategy unsafe-best-match \
    --from mink-warp \
    python -c "import mink_warp; print(mink_warp.__version__)"
```

## Publish to PyPI manually (optional)

Trusted publishing via the tag push workflow is preferred. For a manual upload:

```sh
UV_PUBLISH_TOKEN=<your-pypi-token> make publish
```

## Post-release

```sh
uvx --refresh --from mink-warp python -c "import mink_warp; print(mink_warp.__version__)"
```

Create a GitHub Release from the tag (optional, for release notes only — PyPI publish does not require it).

## Releasing from a past tag

```sh
git checkout vX.Y.Z
make build
make publish
git checkout main
```
