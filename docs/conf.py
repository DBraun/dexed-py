"""Sphinx configuration for dexed-py documentation."""

import shutil
from pathlib import Path

from dexed.version import __version__

project = "dexed-py"
copyright = "2026, David Braun"
author = "David Braun"
version = __version__
release = __version__

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
    "myst_parser",
    "sphinx_llm.txt",
]

# MyST settings
myst_enable_extensions = [
    "colon_fence",
    "fieldlist",
]
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# Theme
html_theme = "sphinx_rtd_theme"

# Autodoc settings
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}
autodoc_member_order = "bysource"
autodoc_typehints = "description"

# Napoleon settings (Google-style docstrings)
napoleon_google_docstrings = True
napoleon_numpy_docstrings = True

# Intersphinx
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

# Exclude patterns
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# Override sphinx-llm's auto-generated llms.txt with our hand-written version.
# sphinx-llm runs at build-finished priority 101; we run at 200 to copy after it.
def _override_llms_txt(app, exception):
    if exception:
        return
    src = Path(__file__).parent / "llms.txt"
    dst = Path(app.builder.outdir) / "llms.txt"
    if src.exists():
        shutil.copy2(src, dst)


def setup(app):
    app.connect("build-finished", _override_llms_txt, priority=200)
