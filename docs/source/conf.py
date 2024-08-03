# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys

project = "SMTB-2024-Lab-11"
copyright = "2024, Daniel"
author = "Daniel"
release = "0"

sys.path.insert(0, os.path.abspath("../../"))
sys.path.insert(0, os.path.abspath("../../src"))
sys.path.insert(0, os.path.abspath("../../src/charactertokenizer"))

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.todo",
    "sphinx.ext.mathjax",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "datasets": ("https://huggingface.co/docs/datasets/main/en", None),
    "tokenizers": ("https://huggingface.co/docs/tokenizers/main/en", None),
    "transformers": ("https://huggingface.co/docs/transformers/main/en", None),
    "torch": ("https://pytorch.org/docs/stable", None),
    "pytorch_lightning": ("https://pytorch-lightning.readthedocs.io/en/stable", None),
}

todo_include_todos = True
templates_path = ["_templates"]
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
