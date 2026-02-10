# Kaggle Evaluation Setup for SummaC-ZS + UniEval

## 🚀 Complete Kaggle-ready evaluation system with automatic setup

### 📁 Files Created:
1. **`kaggle_setup.py`** - Automatic setup script
2. **`kaggle_kernel.py`** - Complete notebook-ready script  
3. **`README_KAGGLE.md`** - This instruction file

---

## 🎯 Quick Start (Choose one option)

### Option 1: Automatic Setup (Recommended)
```bash
# Upload kaggle_setup.py to Kaggle and run:
python kaggle_setup.py
# Then run:
python kaggle_evaluation.py
```

### Option 2: Copy-Paste Kernel
```bash
# Copy the entire contents of kaggle_kernel.py
# Paste into a Kaggle notebook cell and run
```

---

## 📋 Required Dataset Files

Upload these JSON files to `/kaggle/input/`:

1. **`summaries_flat_1024.json`**
2. **`summaries_flat_overlap.json`**  
3. **`summaries_treesum_pt1_first_500.json`**
4. **`summaries_treesum_pt2_last_500.json`**

---

## 🛠️ What the Setup Does Automatically:

### ✅ Package Installation:
- `torch` (GPU/CPU compatible)
- `transformers>=4.35.0`
- `summac` (SummaC-ZS)
- `UniEval` (cloned from GitHub)
- `datasets`, `numpy`, `pandas`, `tqdm`

### ✅ Environment Setup:
- GPU detection and optimization
- Directory structure creation
- Path configuration
- Model initialization

### ✅ No Truncation:
- Full document processing
- Maximum accuracy evaluation
- Research-quality results

---

## 📊 Evaluation Metrics:

### SummaC-ZS:
- **Consistency** - Factual consistency with source documents
- **No token limits** - Processes full documents

### UniEval (Source-Independent):
- **Fluency** - Language quality and readability
- **Naturalness** - How natural the text sounds
- **Understandability** - Comprehensibility

---

## ⏱️ Expected Time on Kaggle GPUs:

### GPU (P100/T4):
- **100 samples**: ~5-8 minutes
- **1000 samples**: ~45-70 minutes

### CPU (if no GPU):
- **100 samples**: ~30-45 minutes  
- **1000 samples**: ~5-8 hours

---

## 📁 Directory Structure After Setup:

```
/kaggle/
├── input/
│   ├── summaries_flat_1024.json          # ← Upload your files here
│   ├── summaries_flat_overlap.json
│   ├── summaries_treesum_pt1_first_500.json
│   └── summaries_treesum_pt2_last_500.json
├── working/
│   ├── detailed_evaluation_results.csv    # ← Results appear here
│   ├── evaluation_summary.csv
│   ├── evaluation_results.json
│   └── UniEval/                          # Auto-downloaded
└── tmp/                                   # Kaggle temp files
```

---

## 🎯 Step-by-Step Instructions:

### 1. Upload Files to Kaggle
1. Create a new Kaggle notebook
2. Upload your 4 JSON files to the notebook
3. They will appear in `/kaggle/input/`

### 2. Run Evaluation (Option A - Setup Script)
```python
# Run this in first cell:
!python kaggle_setup.py

# Run this in second cell:
!python kaggle_evaluation.py
```

### 3. Run Evaluation (Option B - Direct Kernel)
```python
# Copy entire contents of kaggle_kernel.py
# Paste into notebook cell and run
```

### 4. Download Results
Results will be saved to `/kaggle/working/`:
- `detailed_evaluation_results.csv`
- `evaluation_summary.csv`  
- `evaluation_results.json`

---

## 🔧 Troubleshooting:

### ❌ "Missing datasets" error:
- **Solution**: Ensure JSON files are uploaded to `/kaggle/input/`
- **Check**: File names match exactly

### ❌ "CUDA out of memory" error:
- **Solution**: Reduce batch sizes in the script
- **Change**: `batch_size=16` → `batch_size=8`

### ❌ "Import error" for UniEval:
- **Solution**: Run setup script first
- **Manual**: `!git clone https://github.com/maszhongming/UniEval.git`

### ❌ "No GPU available":
- **Solution**: Script will automatically use CPU
- **Expect**: Much slower execution

---

## 📈 Output Format:

### detailed_evaluation_results.csv:
| sample_id | flat_1024_summac | flat_1024_fluency | ... | treesum_understandability |
|-----------|------------------|-------------------|-----|---------------------------|
| 3         | 0.52             | 0.87              | ... | 0.91                      |
| 4         | 0.48             | 0.82              | ... | 0.88                      |

### evaluation_summary.csv:
| Dataset     | Metric        | Mean  | Std   | Min   | Max   | Median |
|-------------|---------------|-------|-------|-------|-------|--------|
| flat_1024   | summac        | 0.51  | 0.12  | 0.25  | 0.78  | 0.50   |
| flat_1024   | fluency       | 0.85  | 0.08  | 0.65  | 0.96  | 0.86   |

---

## 🚀 Performance Tips:

### For Faster Execution:
1. **Use GPU instances** (P100, T4, A100)
2. **Increase batch sizes** if you have more GPU memory
3. **Process fewer samples** for testing first

### For Maximum Accuracy:
1. **No truncation** - already configured
2. **Full document processing** - already enabled  
3. **Research-quality metrics** - SummaC-ZS + UniEval

---

## 🎓 Research Quality:

✅ **SOTA Metrics**: SummaC-ZS + UniEval  
✅ **No Artificial Limits**: Full document processing  
✅ **Proper Methodology**: Source-dependent + independent evaluation  
✅ **Reproducible**: Deterministic setup  
✅ **Publication Ready**: Comprehensive results and statistics

---

## 📞 Support:

If you encounter issues:
1. Check GPU availability: `!nvidia-smi`
2. Verify file locations: `!ls /kaggle/input/`
3. Check memory usage: `!free -h`
4. Review error messages carefully

---

**🎉 Ready for research-quality evaluation on Kaggle!**
