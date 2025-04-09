# Configuration file for the Sphinx documentation builder
import os
import sys
import datetime

# Add the project root directory to the path
sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('_ext'))

# Project information
project = 'optrade'
copyright = f'{datetime.datetime.now().year}, Xavier Mootoo'
author = 'Xavier Mootoo'
# Extensions
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'module_tree'
]
# Templates and static files
templates_path = ['_templates']
# Update these lines
html_static_path = [os.path.abspath('_static')]
html_css_files = ['custom.css']
# Exclude patterns
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
# HTML theme
html_theme = 'sphinx_rtd_theme'
# Configure autosummary
autosummary_generate = True
add_module_names = False
modindex_common_prefix = ['optrade.']
# Configure napoleon for Google style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_use_rtype = False
napoleon_use_param = True
# Configure autodoc
# autodoc_mock_imports = ['pandas', 'torch', 'rich', 'numpy', 'pydantic', 'neptune']
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',  # Document members in the order they appear in the source
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__',

    # NEW
    # 'imported-members': False,  # Don't show imported modules
    'show-inheritance': False,  # Hide inheritance info
    'no-signature': True,  # Hide signatures to avoid parameter repetition
}

# NEW
python_use_unqualified_type_names = True

# This helps with properly documenting imported members
autodoc_inherit_docstrings = True

# -- Additional options to improve documentation build --
# Better handling of type hints in documentation
autodoc_typehints = 'description'  # Show type hints in description for better readability

# Options for docutils to avoid common warnings
docutils_tab_width = 2
trim_footnote_reference_space = True
