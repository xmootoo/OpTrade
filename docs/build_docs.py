#!/usr/bin/env python
import os
import sys
import shutil
import subprocess
from pathlib import Path

def main():
    # Get the docs directory
    docs_dir = Path(__file__).parent.absolute()
    project_root = docs_dir.parent
    
    # Create template directories if they don't exist
    templates_dir = docs_dir / '_templates'
    os.makedirs(templates_dir, exist_ok=True)
    
    # Create template files
    create_template(templates_dir / 'module.rst', module_template)
    create_template(templates_dir / 'class.rst', class_template)
    
    # Clean _autosummary directory to force regeneration
    autosummary_dir = docs_dir / '_autosummary'
    if autosummary_dir.exists():
        print(f"Cleaning {autosummary_dir}...")
        shutil.rmtree(autosummary_dir)
    os.makedirs(autosummary_dir, exist_ok=True)
    
    # Run sphinx-build with verbose output to see what's happening
    print("Building documentation...")
    os.chdir(docs_dir)
    
    # Create a simple test file to verify import
    test_file = project_root / "import_test.py"
    with open(test_file, "w") as f:
        f.write("import sys\nimport os\n")
        f.write("print(f'Python path: {sys.path}')\n")
        f.write("print(f'Current working directory: {os.getcwd()}')\n")
        f.write("try:\n")
        f.write("    import optrade\n")
        f.write("    print(f'optrade package found at: {optrade.__file__}')\n")
        f.write("    print(f'optrade package contents: {dir(optrade)}')\n")
        f.write("except ImportError as e:\n")
        f.write("    print(f'Import error: {e}')\n")
    
    subprocess.run([sys.executable, str(test_file)])
    
    # Build docs with verbose output
    subprocess.run([
        "sphinx-build", 
        "-b", "html", 
        "-d", "_build/doctrees",
        "-v",  # Verbose output
        ".", 
        "_build/html"
    ])
    
    return 0

def create_template(path, content):
    """Create a template file if it doesn't exist."""
    if not path.exists():
        print(f"Creating template {path}...")
        with open(path, "w") as f:
            f.write(content)
        print(f"Created {path}")
    else:
        print(f"Template {path} already exists")

# Template content
module_template = """{{ fullname | escape | underline }}

.. automodule:: {{ fullname }}
   
   {% block attributes %}
   {% if attributes %}
   .. rubric:: Module Attributes

   {% for item in attributes %}
   .. autoattribute:: {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block functions %}
   {% if functions %}
   .. rubric:: Functions

   {% for item in functions %}
   .. autofunction:: {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block classes %}
   {% if classes %}
   .. rubric:: Classes

   {% for item in classes %}
   .. autoclass:: {{ item }}
      :members:
      :undoc-members:
      :show-inheritance:
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block exceptions %}
   {% if exceptions %}
   .. rubric:: Exceptions

   {% for item in exceptions %}
   .. autoexception:: {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

{% block modules %}
{% if modules %}
.. rubric:: Modules

.. autosummary::
   :toctree:
   :recursive:
   {% for item in modules %}
   {{ item }}
   {%- endfor %}
{% endif %}
{% endblock %}
"""

class_template = """{{ fullname | escape | underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

   {% block methods %}
   {% if methods %}
   .. rubric:: {{ _('Methods') }}

   {% for item in methods %}
   .. automethod:: {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block attributes %}
   {% if attributes %}
   .. rubric:: {{ _('Attributes') }}

   {% for item in attributes %}
   .. autoattribute:: {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}
"""

if __name__ == "__main__":
    sys.exit(main())