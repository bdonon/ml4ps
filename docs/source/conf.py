# Configuration file for the Sphinx documentation builder.

# -- Project information

project = 'ml4ps'
copyright = '2022, Donon & Pagnier'
author = 'Donon & Pagnier'

release = '0.1'
version = '0.1.0'

# -- General configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.viewcode',
    'sphinx.ext.doctest',
    'sphinx.ext.napoleon',
    'sphinx.ext.todo',
    'sphinx.ext.intersphinx',
    'nbsphinx'
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'

# -- Options for EPUB output
epub_show_urls = 'footnote'

# Exclude certain files from the compilation
exclude_patterns = [
    'usecase/0\ Loading\ and\ preprocessing\ data',
    'usecase/1\ Training\ a\ dense\ neural\ network',
    'usecase/2\ Training\ a\ graphical\ neural\ network',
    'usecase/2\ Training\ a\ graphical\ neural\ network-Copy1',
    'usecase/3\ Loading\ a\ trained\ method',
    'powermodel.py'
]

# Display to do items
todo_include_todos = True
