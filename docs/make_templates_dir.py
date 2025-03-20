#!/usr/bin/env python
import os
import sys
from pathlib import Path

def main():
    # Get the docs directory
    docs_dir = Path(__file__).parent.absolute()
    
    # Create _templates directory if it doesn't exist
    templates_dir = docs_dir / '_templates'
    os.makedirs(templates_dir, exist_ok=True)
    
    # Create _static directory if it doesn't exist
    static_dir = docs_dir / '_static'
    os.makedirs(static_dir, exist_ok=True)
    
    # Create _autosummary directory if it doesn't exist
    autosummary_dir = docs_dir / '_autosummary'
    os.makedirs(autosummary_dir, exist_ok=True)
    
    print("Created required directories for Sphinx documentation.")
    return 0

if __name__ == "__main__":
    sys.exit(main())