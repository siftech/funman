# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "funman"
copyright = "2024, SIFT"
author = "SIFT"

from funman import __version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.duration",
    "sphinx.ext.graphviz",
    "sphinx.ext.inheritance_diagram",
    "sphinx.ext.napoleon",
    "sphinxcontrib.autodoc_pydantic",
    "myst_parser",
]

autodoc_pydantic_model_show_json = True
autodoc_pydantic_settings_show_json = False

templates_path = ["_templates"]
exclude_patterns = []

inheritance_graph_attrs = dict(rankdir="TB", size='""')
graphviz_output_format = "svg"
graphviz_dot_args = [
    "-Ecolor=#6ab0de;",
    "-Epenwidth=3;",
]

version = __version__

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = 'pydata_sphinx_theme'
html_theme = "classic"
html_static_path = ["_static"]

# Napoleon configuration for handling numpy docstrings
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# Options for templates
modulefirst = True

rst_epilog = """
.. |FunmanVersion| replace:: {version_num}
""".format(
    version_num=version,
)
