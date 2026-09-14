# Copyright (c) 2026, The Isaac Lab Training Contributors.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sphinx configuration for the IsaacLab Training documentation."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import sphinx_book_theme

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

project = "IsaacLab Training"
author = "The Isaac Lab Training Contributors"
copyright = "2026, The Isaac Lab Training Contributors"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx_copybutton",
    "sphinx_design",
]

copybutton_selector = "div.highlight pre"

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

html_title = "IsaacLab Training Documentation"
html_theme_path = [sphinx_book_theme.get_html_theme_path()]
html_theme = "sphinx_book_theme"
html_favicon = "_static/favicon.ico"
html_show_copyright = True
html_show_sphinx = False
html_last_updated_fmt = ""
html_static_path = ["_static"]
html_css_files = ["css/custom.css"]
html_sidebars = {
    "**": ["navbar-logo.html", "icon-links.html", "search-field.html", "sbt-sidebar-nav.html"],
}
html_theme_options = {
    "path_to_docs": "docs/",
    "collapse_navigation": True,
    "repository_url": "https://github.com/ashwinvkNV/IsaacLabTraining",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_edit_page_button": True,
    "show_toc_level": 1,
    "use_sidenotes": True,
    "logo": {
        "text": "IsaacLab Training Documentation",
        "image_light": "_static/NVIDIA-logo-white.png",
        "image_dark": "_static/NVIDIA-logo-black.png",
    },
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/ashwinvkNV/IsaacLabTraining",
            "icon": "fa-brands fa-square-github",
            "type": "fontawesome",
        },
        {
            "name": "Isaac Lab",
            "url": "https://isaac-sim.github.io/IsaacLab/",
            "icon": "https://img.shields.io/badge/IsaacLab-docs-76B900.svg",
            "type": "url",
        },
    ],
    "icon_links_label": "Quick Links",
}

myst_enable_extensions = [
    "colon_fence",
    "deflist",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}

nitpicky = False
suppress_warnings = [
    "ref.doc",
    "ref.ref",
]

rst_prolog = f"""
.. |repo_root| replace:: {os.fspath(PROJECT_ROOT)}
"""
