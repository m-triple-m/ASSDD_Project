#!/bin/bash

# build.sh - Build script for Render.com deployment
# This script is used to prepare the application for deployment

echo "Upgrading pip, setuptools, and wheel..."
pip install --upgrade pip setuptools wheel

echo "Installing Python dependencies..."
pip install -r requirements.txt

echo "Creating necessary directories..."
mkdir -p static/uploads
mkdir -p static/results
mkdir -p instance
mkdir -p model_cache

echo "Model will be downloaded from Hugging Face Hub at runtime..."
echo "Build complete!"
