# Configuration file for the Sphinx documentation builder.
import os
import sys
import datetime

# Add the project root directory to the path so Sphinx can find your modules
sys.path.insert(0, os.path.abspath('..'))

# Project information
project = 'optrade'
copyright = f'{datetime.datetime.now().year}, Xavier Mootoo'
author = 'Xavier Mootoo'

# Add any Sphinx extension module names here
extensions = [
    'sphinx.ext.autodoc',      # Automatically document your modules
    'sphinx.ext.viewcode',     # Add links to source code
    'sphinx.ext.napoleon',     # Support for NumPy and Google style docstrings
    'sphinx.ext.autosummary',  # Generate summary tables
    'sphinx.ext.intersphinx',  # Link to other projects' documentation
]

# Configure autodoc
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}

# Enable autosummary with recursive option
autosummary_generate = True
autosummary_imported_members = True
# Force autosummary to generate stub files
autosummary_generate_overwrite = True
# Don't import modules listed as 'mock' imports
autodoc_mock_imports = []

# Add any paths that contain templates here
templates_path = ['_templates']

# List of patterns to exclude from source files
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '_templates']

# The theme to use for HTML and HTML Help pages
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files
html_static_path = ['_static']

# Display full module paths in API docs
add_module_names = True

# Ensure typehints are documented
napoleon_use_param = True
napoleon_use_rtype = True


# # Configuration file for the Sphinx documentation builder.
# import os
# import sys
# import datetime

# # Add the project root directory to the path so Sphinx can find your modules
# sys.path.insert(0, os.path.abspath('..'))

# # Project information
# project = 'optrade'
# copyright = f'{datetime.datetime.now().year}, Xavier Mootoo'
# author = 'Xavier Mootoo'

# # Add any Sphinx extension module names here
# extensions = [
#     'sphinx.ext.autodoc',  # Automatically document your modules
#     'sphinx.ext.viewcode',  # Add links to source code
#     'sphinx.ext.napoleon',  # Support for NumPy and Google style docstrings
#     'sphinx.ext.autosummary',  # Generate summary tables
# ]

# # Configure autodoc
# autodoc_default_options = {
#     'members': True,
#     'member-order': 'bysource',
#     'special-members': '__init__',
#     'undoc-members': True,
#     'exclude-members': '__weakref__'
# }

# # Enable autosummary with recursive option
# autosummary_generate = True
# autosummary_imported_members = True

# # Add any paths that contain templates here
# templates_path = ['_templates']

# # List of patterns to exclude from source files
# exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '_templates']

# # The theme to use for HTML and HTML Help pages
# html_theme = 'sphinx_rtd_theme'

# # Add any paths that contain custom static files
# html_static_path = ['_static']

# # Display full module paths in API docs
# add_module_names = True