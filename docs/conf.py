# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import sys
import os

sys.path.append(os.path.join(os.path.abspath(os.pardir)))
autodoc_mock_imports = ["pandas"]

cwd = os.getcwd()
project_root = os.path.dirname(cwd)

sys.path.insert(0, project_root)

import pandas  # noqa E402


# -- Project information -----------------------------------------------------

project = "fantasyfootball"
copyright = "2022, Mark LeBoeuf"
author = "Mark LeBoeuf"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "myst_nb",
    "autoapi.extension",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]
autoapi_dirs = ["../src"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "config.py"]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"
html_logo = "images/logo.png"
html_theme_options = {
    "logo_only": True,
    "display_version": False,
}
