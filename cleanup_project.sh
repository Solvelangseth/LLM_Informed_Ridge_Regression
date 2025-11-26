#!/bin/bash
set -e

echo "=== Removing backup notebook files ==="
find notebooks -type f \( -name "*.bak" -o -name "*.bak2" -o -name "*.bak_importfix" \) -delete

echo "=== Removing macOS .DS_Store files ==="
find . -type f -name ".DS_Store" -delete

echo "=== Removing Jupyter checkpoint folders ==="
find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} +

echo "=== Removing old maintenance scripts ==="
rm -f reorganize_notebooks.sh reorganize_project.sh cleanup_old_structure.sh fix_imports.py fix_notebook_imports.py

echo "=== Cleaning empty directories ==="
find . -type d -empty -delete

echo "Done!"
