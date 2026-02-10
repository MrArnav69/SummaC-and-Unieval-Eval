#!/bin/bash
# RunPod Quick Start Script for Evaluation Setup

echo "🚀 RUNPOD QUICK START - Evaluation Setup"
echo "=========================================="

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 not found. Please install Python3 first."
    exit 1
fi

echo "✅ Python3 found"

# Run the setup script
echo "📥 Downloading repositories and models..."
python3 setup_repos.py

# Check if setup was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "🎯 SETUP SUCCESSFUL!"
    echo ""
    echo "📋 Next Steps:"
    echo "1. Upload your dataset JSON files to this directory"
    echo "2. Follow RUNPOD_SETUP.md for environment setup"
    echo "3. Run evaluation scripts"
    echo ""
    echo "📁 Directory structure ready:"
    ls -la
else
    echo "❌ Setup failed. Please check the error messages above."
    exit 1
fi
