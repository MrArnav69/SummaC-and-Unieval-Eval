#!/usr/bin/env python3
"""
Setup Script for RunPod - Download and Setup Evaluation Libraries
Downloads AlignScore, BARTScore, and UniEval from GitHub
"""

import os
import subprocess
import sys
from pathlib import Path
from urllib.request import urlretrieve
import zipfile
import shutil

# Repository URLs
REPOS = {
    'AlignScore': {
        'url': 'https://github.com/yuh-zha/AlignScore/archive/refs/heads/main.zip',
        'extract_name': 'AlignScore-main',
        'target_name': 'AlignScore'
    },
    'BARTScore': {
        'url': 'https://github.com/neulab/BARTScore/archive/refs/heads/main.zip',
        'extract_name': 'BARTScore-main', 
        'target_name': 'BARTScore'
    },
    'UniEval': {
        'url': 'https://github.com/maszhongming/UniEval/archive/refs/heads/main.zip',
        'extract_name': 'UniEval-main',
        'target_name': 'UniEval'
    }
}

# Model URLs
MODELS = {
    'AlignScore-large': {
        'url': 'https://huggingface.co/nyu-mll/AlignScore-large/resolve/main/AlignScore-large.ckpt',
        'filename': 'AlignScore-large.ckpt'
    }
}

def download_file(url, filename, description="File"):
    """Download file with progress bar."""
    print(f"📥 Downloading {description}...")
    try:
        urlretrieve(url, filename)
        print(f"✅ {description} downloaded successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to download {description}: {e}")
        return False

def extract_zip(zip_path, extract_to):
    """Extract zip file."""
    print(f"📦 Extracting {zip_path}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"✅ Extracted successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to extract {zip_path}: {e}")
        return False

def setup_repository(repo_name, repo_info):
    """Setup a single repository."""
    print(f"\n🔧 Setting up {repo_name}...")
    
    # Download
    zip_filename = f"{repo_name}.zip"
    if not download_file(repo_info['url'], zip_filename, repo_name):
        return False
    
    # Extract
    if not extract_zip(zip_filename, "."):
        return False
    
    # Rename/move to target directory
    extract_path = Path(repo_info['extract_name'])
    target_path = Path(repo_info['target_name'])
    
    if extract_path.exists():
        if target_path.exists():
            shutil.rmtree(target_path)
        shutil.move(str(extract_path), str(target_path))
        print(f"✅ {repo_name} setup complete")
        
        # Clean up zip file
        os.remove(zip_filename)
        return True
    else:
        print(f"❌ Extracted directory {extract_path} not found")
        return False

def setup_models():
    """Setup required models."""
    print(f"\n🤖 Setting up models...")
    
    for model_name, model_info in MODELS.items():
        if not download_file(model_info['url'], model_info['filename'], model_name):
            print(f"⚠️  Failed to download {model_name}")
            print(f"💡 You may need to download it manually from HuggingFace")
        else:
            print(f"✅ {model_name} downloaded successfully")

def check_dependencies():
    """Check if required dependencies are available."""
    print(f"\n🔍 Checking dependencies...")
    
    required_packages = ['requests', 'zipfile', 'shutil', 'pathlib']
    missing = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"❌ Missing packages: {missing}")
        print("💡 Install with: pip install " + " ".join(missing))
        return False
    
    print("✅ All dependencies available")
    return True

def create_requirements_files():
    """Create requirements files for both environments."""
    print(f"\n📋 Creating requirements files...")
    
    # UniEval + BARTScore requirements
    unieval_req = """# Requirements for UniEval + BARTScore Evaluation
# Compatible with torch>=2.0 (A40 GPU environment)

# Core ML/AI libraries
torch>=2.0.0
transformers>=4.20.0
datasets>=2.0.0
tokenizers>=0.13.0

# UniEval dependencies
nltk>=3.7
scikit-learn>=1.1.0
scipy>=1.8.0

# BARTScore dependencies
sentence-transformers>=2.2.0

# Data processing
pandas>=1.5.0
numpy>=1.21.0
tqdm>=4.64.0

# Text processing
spacy>=3.4.0

# Utilities
pathlib2>=2.3.0
python-dateutil>=2.8.0

# Setup script dependencies
requests>=2.25.0
"""
    
    # AlignScore requirements
    alignscore_req = """# Requirements for AlignScore Evaluation
# Compatible with torch<2.0 (separate environment for AlignScore)

# Core ML/AI libraries - EXACT versions required for AlignScore
torch==1.12.1
torchvision==0.13.1
torchaudio==0.12.1

# AlignScore specific dependencies
protobuf<=3.20
pytorch-lightning<2,>=1.7.7
datasets<3,>=2.3.2
jsonlines<3,>=2.0.0

# Text processing - compatible versions
spacy>=3.4.0,<3.8.0
nltk>=3.7,<4.0

# Data processing
pandas>=1.5.0
numpy>=1.21.0,<2.0.0
tqdm>=4.64.0

# Scientific computing
scikit-learn>=1.1.0,<2.0
scipy>=1.8.0,<2.0

# Utilities
pathlib2>=2.3.0
python-dateutil>=2.8.0

# Setup script dependencies
requests>=2.25.0
"""
    
    try:
        with open('requirements_unieval_bartscore.txt', 'w') as f:
            f.write(unieval_req)
        
        with open('requirements_alignscore.txt', 'w') as f:
            f.write(alignscore_req)
        
        print("✅ Requirements files created")
        return True
    except Exception as e:
        print(f"❌ Failed to create requirements files: {e}")
        return False

def create_setup_instructions():
    """Create setup instructions for RunPod."""
    instructions = """# RunPod Setup Instructions

## Quick Setup (RunPod)

### 1. Run Setup Script
```bash
python setup_repos.py
```

### 2. Environment Setup

#### Environment 1: UniEval + BARTScore
```bash
# Create environment
conda create -n unieval_bartscore python=3.9
conda activate unieval_bartscore

# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install -r requirements_unieval_bartscore.txt

# Install spaCy model
python -m spacy download en_core_web_sm
```

#### Environment 2: AlignScore
```bash
# Create environment
conda create -n alignscore python=3.8
conda activate alignscore

# Install specific torch version
pip install torch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install -r requirements_alignscore.txt

# Install spaCy model
python -m spacy download en_core_web_sm

# Install AlignScore
cd AlignScore && pip install . && cd ..
```

### 3. Upload Your Data
Upload your JSON dataset files to the same directory:
- summaries_flat_1024.json
- summaries_flat_overlap.json  
- summaries_treesum_pt1_first_500.json
- summaries_treesum_pt2_last_500.json

### 4. Run Evaluation
```bash
# UniEval Fluency (Environment 1)
python evaluation_complete.py

# AlignScore (Environment 2)
python alignscore_only.py

# Generate Report
python generate_report.py
```

## Directory Structure After Setup
```
/home/user/
├── AlignScore/              # AlignScore library
├── BARTScore/              # BARTScore library
├── UniEval/                # UniEval library
├── AlignScore-large.ckpt    # AlignScore model
├── evaluation_complete.py    # UniEval evaluation script
├── alignscore_only.py       # AlignScore evaluation script
├── generate_report.py       # Report generator
├── requirements_unieval_bartscore.txt
├── requirements_alignscore.txt
├── results/               # Output directory
└── [your datasets].json   # Your data files
```

## Notes
- The setup script downloads all required libraries and models
- AlignScore-large.ckpt is ~4.6GB, ensure sufficient disk space
- RunPod typically has fast internet, downloads should complete quickly
- Use GPU-enabled instances for best performance
"""
    
    try:
        with open('RUNPOD_SETUP.md', 'w') as f:
            f.write(instructions)
        print("✅ RunPod setup instructions created")
        return True
    except Exception as e:
        print(f"❌ Failed to create setup instructions: {e}")
        return False

def main():
    """Main setup function."""
    print("🚀 RUNPOD EVALUATION SETUP")
    print("="*50)
    print("📥 Downloading AlignScore, BARTScore, and UniEval from GitHub")
    print("🤖 Downloading required models")
    print("📋 Creating requirements and setup files")
    
    # Check dependencies
    if not check_dependencies():
        return False
    
    # Setup repositories
    success_count = 0
    for repo_name, repo_info in REPOS.items():
        if setup_repository(repo_name, repo_info):
            success_count += 1
    
    print(f"\n📊 Repositories setup: {success_count}/{len(REPOS)}")
    
    # Setup models
    setup_models()
    
    # Create requirements files
    create_requirements_files()
    
    # Create setup instructions
    create_setup_instructions()
    
    # Final status
    print(f"\n🎯 SETUP COMPLETE!")
    print("="*30)
    print("✅ Downloaded repositories:")
    for repo_name in REPOS.keys():
        if Path(repo_name).exists():
            print(f"   📁 {repo_name}")
    
    print("✅ Created files:")
    print("   📄 requirements_unieval_bartscore.txt")
    print("   📄 requirements_alignscore.txt") 
    print("   📄 RUNPOD_SETUP.md")
    
    print("\n📋 Next Steps:")
    print("1. Upload your dataset JSON files")
    print("2. Follow RUNPOD_SETUP.md for environment setup")
    print("3. Run evaluation scripts")
    
    return True

if __name__ == "__main__":
    main()
