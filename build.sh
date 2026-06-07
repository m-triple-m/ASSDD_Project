#!/bin/bash

# build.sh - Build script for Render.com deployment
# This script is used to prepare the application for deployment

echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "Creating necessary directories..."
mkdir -p static/uploads
mkdir -p static/results
mkdir -p instance

echo "Model file check..."
# The model file should be downloaded separately or committed
# If you need to download it from a source, add the download command here
# Example:
# echo "Downloading model file..."
# wget https://your-storage-url/resnetunet_aerial.pth -O resnetunet_aerial.pth

if [ ! -f "resnetunet_aerial.pth" ]; then
    echo "⚠️  Warning: Model file (resnetunet_aerial.pth) not found!"
    echo "The application will start but predictions will not work."
    echo "Please configure model download in this script or upload via Render dashboard."
fi

echo "Build complete!"
