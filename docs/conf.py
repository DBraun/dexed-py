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


def _draw_algorithm_chart(app):
    """Redraw dx7_algorithms.svg before the sources are read.

    The chart is generated from the engine's own routing table, so it is built
    here rather than committed -- a checked-in copy can fall out of step with
    src/msfa/fm_core.cc, and it churns a 1400-line diff on every layout tweak.
    Doing it at builder-inited, rather than in CI, keeps a local `make html`
    working too.
    """
    import importlib.util

    generator = Path(__file__).parent / "make_algorithm_chart.py"
    spec = importlib.util.spec_from_file_location("make_algorithm_chart", generator)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    out = Path(__file__).parent / "dx7_algorithms.svg"
    out.write_text(module.render_chart(), encoding="utf-8")
    print(f"[chart] wrote {out.name}")


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
    app.connect("builder-inited", _draw_algorithm_chart)
    app.connect("build-finished", _override_llms_txt, priority=200)
