.PHONY: sync format test smoke build publish publish-test docs docs-watch docs-clean

sync:
	uv sync --extra dev --extra examples --group docs

format:
	uv run ruff format
	uv run ruff check --fix

test:
	uv run pytest

smoke:
	MUJOCO_GL=disable uv run tests/smoke_test.py

build:
	uv build
	MUJOCO_GL=disable uv run --isolated --no-project --with dist/*.whl tests/smoke_test.py
	MUJOCO_GL=disable uv run --isolated --no-project --with dist/*.tar.gz tests/smoke_test.py
	@echo "Build and smoke test successful"

publish-test: build
	uv publish --publish-url https://test.pypi.org/legacy/

publish: build
	uv publish

docs:
	uv run --group docs sphinx-build -W -j auto docs docs/_build

docs-watch:
	uv run --group docs sphinx-autobuild -j auto docs docs/_build --watch src

docs-clean:
	rm -rf docs/_build
