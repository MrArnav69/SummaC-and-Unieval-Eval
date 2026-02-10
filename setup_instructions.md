# Environment Setup Instructions

## Overview
Two separate environments are required due to torch version conflicts:
- **Environment 1**: UniEval + BARTScore (torch>=2.0)
- **Environment 2**: AlignScore (torch<2.0)

---

## Environment 1: UniEval + BARTScore (A40 GPU)

### Setup Commands:
```bash
# Create environment
conda create -n unieval_bartscore python=3.9
conda activate unieval_bartscore

# Install PyTorch with CUDA support (for A40 GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install -r requirements_unieval_bartscore.txt

# Install spaCy model
python -m spacy download en_core_web_sm

# Install UniEval (if not already installed)
cd UniEval && pip install . && cd ..

# Verify installation
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

### Files to Run:
```bash
# UniEval Fluency evaluation
python evaluation_complete.py

# Generate report (after both metrics are done)
python generate_report.py
```

---

## Environment 2: AlignScore (CPU/GPU)

### Setup Commands:
```bash
# Create environment
conda create -n alignscore python=3.8
conda activate alignscore

# Install specific torch version
pip install torch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 --index-url https://download.pytorch.org/whl/cpu

# Install spaCy and model
pip install spacy==3.7.5
python -m spacy download en_core_web_sm

# Install dependencies
pip install -r requirements_alignscore.txt

# Install AlignScore
cd AlignScore && pip install . && cd ..

# Verify installation
python -c "import torch; print('PyTorch:', torch.__version__); from alignscore import AlignScore; print('AlignScore imported successfully')"
```

### Files to Run:
```bash
# AlignScore evaluation
python alignscore_only.py

# Generate report (after both metrics are done)
python generate_report.py
```

---

## Quick Reference

### Environment 1 (UniEval + BARTScore):
- **Python**: 3.9
- **PyTorch**: >=2.0
- **GPU**: CUDA enabled
- **Main script**: `evaluation_complete.py`

### Environment 2 (AlignScore):
- **Python**: 3.8
- **PyTorch**: 1.12.1
- **GPU**: Optional (CPU works)
- **Main script**: `alignscore_only.py`

### Common:
- **Report generator**: `generate_report.py`
- **Results directory**: `results/`
- **Both evaluate**: 1000 samples across 3 datasets

---

## Troubleshooting

### Common Issues:
1. **CUDA out of memory**: Scripts have automatic OOM protection
2. **Torch version conflicts**: Use separate environments as specified
3. **spaCy model missing**: Run `python -m spacy download en_core_web_sm`
4. **UniEval import error**: Ensure you're in the correct directory with UniEval folder

### Performance Tips:
- Use A40 GPU for best performance
- Scripts automatically optimize batch sizes
- OOM protection ensures completion even with memory constraints
- Results are saved with timestamps to avoid overwrites
