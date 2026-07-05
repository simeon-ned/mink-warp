"""Execute runnable code examples embedded in the documentation."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

_EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"
_DOC_EXAMPLES = sorted((_EXAMPLES_DIR / "docs").glob("*.py"))


@pytest.mark.parametrize("path", _DOC_EXAMPLES, ids=lambda p: p.name)
def test_doc_example_runs(path: Path) -> None:
    result = subprocess.run(
        [sys.executable, f"docs/{path.name}"],
        check=False,
        cwd=_EXAMPLES_DIR,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"docs/{path.name} failed:\n{result.stdout}\n{result.stderr}"
    )
