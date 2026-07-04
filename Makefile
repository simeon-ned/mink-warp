.PHONY: sync format test docs docs-watch docs-clean

sync:
	uv sync --extra dev --extra examples --group docs

format:
	uv run ruff format
	uv run ruff check --fix

test:
	uv run pytest

docs:
	uv run --group docs sphinx-build -W -j auto docs docs/_build

docs-watch:
	uv run --group docs sphinx-autobuild -j auto docs docs/_build --watch src

docs-clean:
	rm -rf docs/_build
