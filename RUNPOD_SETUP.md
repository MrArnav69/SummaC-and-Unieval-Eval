# RunPod Setup Instructions

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
