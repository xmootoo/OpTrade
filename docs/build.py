#!/usr/bin/env python
import os
import subprocess
import sys

def main():
    # Get the project root directory (parent of docs directory)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    # Change to the project root directory
    os.chdir(project_root)

    # Install the project using poetry
    print("Installing project with Poetry...")
    subprocess.check_call(["poetry", "install", "--with=docs", "--no-interaction"])

    # Note: make_templates_dir.py has been removed as requested

    # We're using autosummary instead of sphinx-apidoc, so we don't need to run generate_apidoc.py
    # The autosummary extension will handle generating documentation for all modules automatically

    return 0

if __name__ == "__main__":
    sys.exit(main())
