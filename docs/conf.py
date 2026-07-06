"""Sphinx configuration for mink-warp (Mink-style layout)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import sphinx_book_theme

sys.path.insert(0, os.path.abspath("../src"))

project = "mink-warp"
copyright = "2026, Simeon Nedelchev and Ivan Domrachev"
author = "Simeon Nedelchev, Ivan Domrachev"

# Version from pyproject.toml when available.
try:
    import toml

    version = toml.load(Path(__file__).resolve().parent.parent / "pyproject.toml")[
        "project"
    ]["version"]
except Exception:
    version = "0.1.0"
if not str(version).isalpha():
    version = "v" + str(version)

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
    "sphinx_design",
]

source_suffix = ".rst"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "mujoco": ("https://mujoco.readthedocs.io/en/stable/", None),
}

autodoc_typehints = "signature"
autoclass_content = "class"
autodoc_class_signature = "separated"
autodoc_member_order = "bysource"
autodoc_inherit_docstrings = True
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "undoc-members": False,
    "show-inheritance": True,
    "exclude-members": "__init__, __post_init__, __new__",
}
autodoc_type_aliases = {
    "npt.ArrayLike": "ArrayLike",
}
toc_object_entries = False

exclude_patterns = [
    "_build",
    "_templates",
    "Thumbs.db",
    ".DS_Store",
    "BUILDING.md",
]

language = "en"

html_baseurl = "https://simeon-ned.github.io/mink-warp/"
html_title = "mink-warp Documentation"
html_theme_path = [sphinx_book_theme.get_html_theme_path()]
html_theme = "sphinx_book_theme"
html_show_sphinx = False
html_last_updated_fmt = ""
html_static_path = ["_static"]
html_css_files = ["refs.css"]

html_theme_options = {
    "path_to_docs": "docs/",
    "collapse_navigation": True,
    "repository_url": "https://github.com/simeon-ned/mink-warp",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_edit_page_button": True,
    "show_toc_level": 2,
}

html_context = {
    "github_user": "simeon-ned",
    "github_repo": "mink-warp",
    "github_version": "main",
    "doc_path": "docs",
}
