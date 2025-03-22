#!/bin/bash
# Sphinx-apidoc helper script with exclusion lists

# Configuration
OUTPUT_DIR="docs"
MODULE_PATH="optrade"

# List of files and folders to exclude (relative to current directory)
EXCLUSIONS=(
    "optrade/config"
    "optrade/analysis"
    "optrade/assets"
    "optrade/examples"
    "optrade/jobs"
    "optrade/logs"
    "optrade/figures"
    "optrade/personal"
    "optrade/notebooks"
    "optrade/utils/personal"
    "optrade/main.py"
)

# Print configuration
echo "Generating Sphinx API documentation"
echo "Output directory: $OUTPUT_DIR"
echo "Module path: $MODULE_PATH"
echo "Excluded items:"
for item in "${EXCLUSIONS[@]}"; do
    echo "  - $item"
done
echo ""

# Make sure output directory exists
mkdir -p "$OUTPUT_DIR"

# Build the sphinx-apidoc command with exclusions
CMD="sphinx-apidoc -f -o $OUTPUT_DIR $MODULE_PATH"

# Add exclusions to command
for exclusion in "${EXCLUSIONS[@]}"; do
    CMD="$CMD $exclusion"
done

# Execute the command
echo "Executing: $CMD"
echo ""
eval "$CMD"

# Check if command was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "Documentation successfully generated in $OUTPUT_DIR"
    echo "To build the HTML documentation, run: cd $OUTPUT_DIR && make html"
else
    echo ""
    echo "Error generating documentation. Please check the output for details."
    exit 1
fi
