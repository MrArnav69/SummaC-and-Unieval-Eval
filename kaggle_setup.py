#!/usr/bin/env python3
"""
Kaggle Setup Script for SummaC-ZS and UniEval Evaluation
Automatically downloads and configures all dependencies
"""

import os
import sys
import subprocess
import json
from pathlib import Path

def install_dependencies():
    """Install all required packages."""
    print("📦 Installing required packages...")
    
    packages = [
        "torch",
        "transformers>=4.35.0", 
        "datasets",
        "numpy",
        "pandas",
        "tqdm",
        "summac",
        "sentence-transformers",
        "huggingface_hub"
    ]
    
    for package in packages:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package, "-q"])
    
    print("✅ All packages installed successfully!")

def setup_unieval():
    """Download and setup UniEval."""
    print("🔧 Setting up UniEval...")
    
    # Clone UniEval if not exists
    if not os.path.exists("UniEval"):
        subprocess.check_call(["git", "clone", "https://github.com/maszhongming/UniEval.git"])
        print("✅ UniEval cloned successfully!")
    else:
        print("✅ UniEval already exists!")
    
    # Install UniEval requirements
    unieval_req_path = os.path.join("UniEval", "requirements.txt")
    if os.path.exists(unieval_req_path):
        print("Installing UniEval requirements...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", unieval_req_path, "-q"])
    
    print("✅ UniEval setup complete!")

def setup_directory_structure():
    """Create necessary directories."""
    print("📁 Setting up directory structure...")
    
    directories = [
        "input",
        "output", 
        "working",
        "models"
    ]
    
    for dir_name in directories:
        os.makedirs(dir_name, exist_ok=True)
        print(f"✅ Created {dir_name}/ directory")
    
    print("✅ Directory structure ready!")

def check_gpu():
    """Check GPU availability and print info."""
    print("🔍 Checking GPU availability...")
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name()
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"✅ GPU detected: {gpu_name}")
            print(f"✅ GPU Memory: {gpu_memory:.1f} GB")
            return True
        else:
            print("⚠️  No GPU detected. Using CPU (much slower!)")
            return False
    except Exception as e:
        print(f"❌ Error checking GPU: {e}")
        return False

def create_kaggle_evaluation_script():
    """Create the main evaluation script for Kaggle."""
    script_content = '''#!/usr/bin/env python3
"""
Kaggle Evaluation Script for SummaC-ZS and UniEval Metrics
Optimized for GPU with no truncation limits
"""

import json
import sys
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

# Add paths
sys.path.append('/kaggle/working/UniEval')
sys.path.append('/kaggle/input')

try:
    from summac.model_summac import SummaCZS
    from UniEval.utils import convert_to_json
    from UniEval.metric.evaluator import get_evaluator
except ImportError as e:
    print(f"Import error: {e}")
    print("Please run setup.py first!")
    sys.exit(1)

def load_datasets():
    """Load datasets from Kaggle input directory."""
    print("Loading datasets...")
    
    base_path = Path("/kaggle/input")
    
    datasets = {
        'flat_1024': None,
        'flat_overlap': None, 
        'treesum_pt1': None,
        'treesum_pt2': None
    }
    
    # Try different possible locations
    possible_paths = [
        base_path,
        base_path / "evaluation-data",
        base_path / "summaries",
        Path("/kaggle/working")
    ]
    
    for dataset_name in datasets.keys():
        filename = f"summaries_{dataset_name.replace('_', '')}.json" if dataset_name.startswith('treesum') else f"summaries_{dataset_name}.json"
        
        for path in possible_paths:
            file_path = path / filename
            if file_path.exists():
                with open(file_path, 'r') as f:
                    datasets[dataset_name] = json.load(f)
                print(f"✅ Loaded {filename} from {path}")
                break
        
        if datasets[dataset_name] is None:
            print(f"❌ Could not find {filename}")
            print(f"🔍 Searched in: {[str(p) for p in possible_paths]}")
    
    return datasets

def create_sample_mappings(flat_1024, flat_overlap, treesum_pt1, treesum_pt2):
    """Create matched samples across datasets."""
    print("Creating matched samples...")
    
    # Concatenate treesum datasets
    treesum_combined = treesum_pt1 + treesum_pt2
    
    # Create mappings
    flat_1024_map = {item['sample_idx']: item for item in flat_1024}
    flat_overlap_map = {item['sample_idx']: item for item in flat_overlap}
    treesum_map = {item['sample_id']: item for item in treesum_combined}
    
    # Find common IDs
    common_ids = set(flat_1024_map.keys()) & set(flat_overlap_map.keys()) & set(treesum_map.keys())
    print(f"Found {len(common_ids)} common samples")
    
    # Create matched samples
    matched_samples = []
    for sample_id in sorted(common_ids):
        matched_samples.append({
            'sample_id': sample_id,
            'flat_1024': flat_1024_map[sample_id],
            'flat_overlap': flat_overlap_map[sample_id],
            'treesum': treesum_map[sample_id]
        })
    
    return matched_samples

def initialize_summac():
    """Initialize SummaC-ZS model."""
    print("Initializing SummaC-ZS...")
    import torch
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SummaCZS(models=['vitc'], granularity="sentence", device=device, batch_size=8)
    return model

def initialize_unieval():
    """Initialize UniEval evaluator."""
    print("Initializing UniEval...")
    import torch
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    evaluator = get_evaluator('summarization', device=device, max_length=4096)
    return evaluator

def evaluate_summac_batch(model, documents, summaries, batch_size=16):
    """Evaluate SummaC-ZS scores with no limits."""
    scores = []
    
    for i in tqdm(range(0, len(documents), batch_size), desc="SummaC-ZS"):
        batch_docs = documents[i:i+batch_size]
        batch_sums = summaries[i:i+batch_size]
        
        try:
            valid_docs = [doc for doc in batch_docs if doc.strip()]
            valid_sums = [summary for summary in batch_sums if summary.strip()]
            
            if not valid_docs:
                scores.extend([0.0] * len(batch_docs))
                continue
            
            result = model.score(valid_docs, valid_sums)
            
            if result and 'scores' in result and len(result['scores']) > 0:
                scores.extend(result['scores'])
            else:
                scores.extend([0.0] * len(valid_docs))
                
        except Exception as e:
            print(f"Error in SummaC-ZS: {e}")
            scores.extend([0.0] * len(batch_docs))
    
    return scores

def evaluate_unieval_batch(evaluator, output_list, dimensions=['fluency', 'naturalness', 'understandability'], batch_size=32):
    """Evaluate UniEval scores with no limits."""
    all_scores = {dim: [] for dim in dimensions}
    
    for i in tqdm(range(0, len(output_list), batch_size), desc="UniEval"):
        batch_out = output_list[i:i+batch_size]
        
        try:
            filtered_out = [out for out in batch_out if out.strip()]
            
            if not filtered_out:
                for dim in dimensions:
                    all_scores[dim].extend([0.0] * len(batch_out))
                continue
            
            data = convert_to_json(output_list=filtered_out)
            eval_scores = evaluator.evaluate(data, dims=dimensions, overall=False, print_result=False)
            
            if isinstance(eval_scores, list) and len(eval_scores) > 0:
                for dim in dimensions:
                    dim_scores = [sample_scores.get(dim, 0.0) for sample_scores in eval_scores]
                    all_scores[dim].extend(dim_scores)
            else:
                for dim in dimensions:
                    all_scores[dim].extend([0.0] * len(filtered_out))
                    
        except Exception as e:
            print(f"Error in UniEval: {e}")
            for dim in dimensions:
                all_scores[dim].extend([0.0] * len(batch_out))
    
    return all_scores

def main():
    """Main evaluation function."""
    print("🚀 Starting Kaggle Evaluation (NO TRUNCATION)...")
    
    # Check GPU
    import torch
    if torch.cuda.is_available():
        print(f"✅ GPU: {torch.cuda.get_device_name()}")
        print(f"✅ Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        torch.backends.cudnn.benchmark = True
    else:
        print("⚠️  Using CPU - this will be very slow!")
    
    # Load datasets
    datasets = load_datasets()
    if None in datasets.values():
        print("❌ Missing datasets. Please check file locations.")
        return
    
    # Create matched samples
    matched_samples = create_sample_mappings(
        datasets['flat_1024'], 
        datasets['flat_overlap'],
        datasets['treesum_pt1'],
        datasets['treesum_pt2']
    )
    
    # Use all samples for Kaggle
    print(f"Processing all {len(matched_samples)} samples...")
    
    # Initialize models
    summac_model = initialize_summac()
    unieval_evaluator = initialize_unieval()
    
    # Create document mapping
    treesum_combined = datasets['treesum_pt1'] + datasets['treesum_pt2']
    document_map = {sample['sample_id']: sample['document'] for sample in treesum_combined}
    
    # Prepare data
    datasets_info = {
        'flat_1024': {
            'docs': [document_map[sample['sample_id']] for sample in matched_samples],
            'generated': [sample['flat_1024']['generated_summary'] for sample in matched_samples],
            'reference': [sample['flat_1024']['reference_summary'] for sample in matched_samples]
        },
        'flat_overlap': {
            'docs': [document_map[sample['sample_id']] for sample in matched_samples],
            'generated': [sample['flat_overlap']['generated_summary'] for sample in matched_samples],
            'reference': [sample['flat_overlap']['reference_summary'] for sample in matched_samples]
        },
        'treesum': {
            'docs': [sample['treesum']['document'] for sample in matched_samples],
            'generated': [sample['treesum']['generated_summary'] for sample in matched_samples],
            'reference': [sample['treesum']['reference_summary'] for sample in matched_samples]
        }
    }
    
    # Evaluate
    results = {}
    
    for dataset_name, data in datasets_info.items():
        print(f"\\n📊 Evaluating {dataset_name}...")
        
        summac_scores = evaluate_summac_batch(summac_model, data['docs'], data['generated'])
        unieval_scores = evaluate_unieval_batch(unieval_evaluator, data['generated'])
        
        results[dataset_name] = {
            'summac_scores': summac_scores,
            'unieval_fluency': unieval_scores['fluency'],
            'unieval_naturalness': unieval_scores['naturalness'],
            'unieval_understandability': unieval_scores['understandability']
        }
        
        print(f"✅ {dataset_name} completed!")
    
    # Save results
    output_dir = Path("/kaggle/working")
    
    # Detailed results
    results_data = []
    for i, sample in enumerate(matched_samples):
        row = {
            'sample_id': sample['sample_id'],
            'flat_1024_summac': results['flat_1024']['summac_scores'][i],
            'flat_1024_fluency': results['flat_1024']['unieval_fluency'][i],
            'flat_1024_naturalness': results['flat_1024']['unieval_naturalness'][i],
            'flat_1024_understandability': results['flat_1024']['unieval_understandability'][i],
            'flat_overlap_summac': results['flat_overlap']['summac_scores'][i],
            'flat_overlap_fluency': results['flat_overlap']['unieval_fluency'][i],
            'flat_overlap_naturalness': results['flat_overlap']['unieval_naturalness'][i],
            'flat_overlap_understandability': results['flat_overlap']['unieval_understandability'][i],
            'treesum_summac': results['treesum']['summac_scores'][i],
            'treesum_fluency': results['treesum']['unieval_fluency'][i],
            'treesum_naturalness': results['treesum']['unieval_naturalness'][i],
            'treesum_understandability': results['treesum']['unieval_understandability'][i]
        }
        results_data.append(row)
    
    df = pd.DataFrame(results_data)
    df.to_csv(output_dir / "detailed_evaluation_results.csv", index=False)
    
    # Summary statistics
    summary_stats = []
    metrics = ['summac', 'fluency', 'naturalness', 'understandability']
    datasets_list = ['flat_1024', 'flat_overlap', 'treesum']
    
    for metric in metrics:
        for dataset in datasets_list:
            if metric == 'summac':
                scores = results[dataset]['summac_scores']
            elif metric == 'fluency':
                scores = results[dataset]['unieval_fluency']
            elif metric == 'naturalness':
                scores = results[dataset]['unieval_naturalness']
            else:  # understandability
                scores = results[dataset]['unieval_understandability']
            
            summary_stats.append({
                'Dataset': dataset,
                'Metric': metric,
                'Mean': np.mean(scores),
                'Std': np.std(scores),
                'Min': np.min(scores),
                'Max': np.max(scores),
                'Median': np.median(scores)
            })
    
    summary_df = pd.DataFrame(summary_stats)
    summary_df.to_csv(output_dir / "evaluation_summary.csv", index=False)
    
    # JSON results
    with open(output_dir / "evaluation_results.json", 'w') as f:
        json.dump({
            'detailed_results': results_data,
            'summary_statistics': summary_stats,
            'num_samples': len(matched_samples)
        }, f, indent=2)
    
    print(f"\\n🎉 Evaluation completed!")
    print(f"📊 Processed {len(matched_samples)} samples")
    print(f"💾 Results saved to /kaggle/working/")
    print(f"📄 Files created:")
    print(f"   - detailed_evaluation_results.csv")
    print(f"   - evaluation_summary.csv")
    print(f"   - evaluation_results.json")

if __name__ == "__main__":
    main()
'''
    
    with open("kaggle_evaluation.py", "w") as f:
        f.write(script_content)
    
    print("✅ Created kaggle_evaluation.py")

def main():
    """Main setup function."""
    print("🚀 Kaggle Setup for SummaC-ZS + UniEval Evaluation")
    print("=" * 50)
    
    # Setup steps
    setup_directory_structure()
    install_dependencies()
    setup_unieval()
    create_kaggle_evaluation_script()
    
    # Check GPU
    has_gpu = check_gpu()
    
    print("\n" + "=" * 50)
    print("✅ SETUP COMPLETE!")
    print("=" * 50)
    print("📁 Directory Structure:")
    print("   /kaggle/working/     - Output files")
    print("   /kaggle/input/       - Your dataset files")
    print("   /kaggle/working/UniEval/ - UniEval library")
    print("\n📋 Next Steps:")
    print("1. Upload your JSON files to /kaggle/input/")
    print("2. Run: python kaggle_evaluation.py")
    print("\n📊 Expected Files:")
    print("   - summaries_flat_1024.json")
    print("   - summaries_flat_overlap.json") 
    print("   - summaries_treesum_pt1_first_500.json")
    print("   - summaries_treesum_pt2_last_500.json")
    
    if has_gpu:
        print("🚀 GPU detected - evaluation will be fast!")
    else:
        print("⚠️  No GPU - evaluation will be slow but should work")

if __name__ == "__main__":
    main()
