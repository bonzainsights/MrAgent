#!/usr/bin/env bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "📦 Preparing to publish bonza-mragent to PyPI..."

# Step 1: Clean old builds
echo "🧹 Cleaning old build files..."
rm -rf build/ dist/ bonza_mragent.egg-info/

# Step 2: Build the package
echo "🏗️ Building the package..."
python -m build

# Step 3: Check with Twine
echo "🔍 Checking the build with Twine..."
twine check dist/*

# Step 4: Upload to PyPI
echo "🚀 Uploading to PyPI..."
echo "Please enter your PyPI API token when prompted (username is __token__)."
twine upload dist/*

echo "✅ Successfully published to PyPI!"
