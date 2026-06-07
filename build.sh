#!/bin/bash

# build.sh - Build script for Render.com deployment
# This script is used to prepare the application for deployment

set -e  # Exit on error

echo "Python version check:"
python --version

echo "Upgrading pip, setuptools, and wheel..."
python -m pip install --no-cache-dir --upgrade pip setuptools wheel

echo "Installing Python dependencies..."
pip install --no-cache-dir -r requirements.txt

echo "Creating necessary directories..."
mkdir -p static/uploads
mkdir -p static/results
mkdir -p instance
mkdir -p model_cache

echo "Model will be downloaded from Hugging Face Hub at runtime..."
echo "Build complete!"
