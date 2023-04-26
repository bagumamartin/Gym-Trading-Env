# Configuration file for the Sphinx documentation builder.
import os
import sys
sys.path.insert(0, os.path.abspath('../../src'))
# -- Project information

project = 'Gym Trading Env'
copyright = '2023, Clement Perroud'
author = 'Clement Perroud'

release = '0.1'
version = '0.1.0'

# -- General configuration

extensions = [
    'sphinx.ext.githubpages',
    'sphinx.ext.autodoc',
    'sphinx.ext.intersphinx',
    'sphinx_copybutton',
]


templates_path = ['_templates']

# -- Options for HTML output

html_theme = 'furo'
html_title = " "
html_static_path = ['_static']
html_favicon = 'images/favicon.png'
html_theme_options = {
    "light_logo": "logo_light-bg.png",
    "dark_logo": "logo_dark-bg.png",
}
html_css_files = ["style.css"]
