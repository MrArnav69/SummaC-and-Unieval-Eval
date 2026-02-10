#!/usr/bin/env python3
"""
Kaggle Kernel Script - Final Working Version
Uses Kaggle's pre-installed packages to avoid conflicts
"""

# Cell 1: Setup and Imports
print("🚀 Setting up Kaggle Environment...")

import os
import sys
import subprocess
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

# Only install what's missing, use existing packages
print("📦 Installing missing packages...")
missing_packages = [
    "summac",  # Only install what we need
]

for package in missing_packages:
    print(f"Installing {package}...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", package, "-q"])

# Clone UniEval if not exists
if not os.path.exists("UniEval"):
    subprocess.check_call(["git", "clone", "https://github.com/maszhongming/UniEval.git"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "UniEval/requirements.txt", "-q"])

print("✅ Setup complete!")

# Cell 2: Imports and GPU Check
print("🔍 Checking environment...")

# Add paths
sys.path.append('/kaggle/working/UniEval')

import torch
from summac.model_summac import SummaCZS
from UniEval.utils import convert_to_json
from UniEval.metric.evaluator import get_evaluator

# GPU Check
if torch.cuda.is_available():
    print(f"✅ GPU: {torch.cuda.get_device_name()}")
    print(f"✅ Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    torch.backends.cudnn.benchmark = True
    device = "cuda"
else:
    print("⚠️  Using CPU - this will be slow!")
    device = "cpu"

# Cell 3: Data Loading Functions
def find_dataset_files():
    """Find dataset files in common Kaggle locations."""
    base_paths = [
        Path("/kaggle/input"),
        Path("/kaggle/working"),
        Path(".")
    ]
    
    datasets = {}
    file_patterns = {
        'flat_1024': 'summaries_flat_1024.json',
        'flat_overlap': 'summaries_flat_overlap.json',
        'treesum_pt1': 'summaries_treesum_pt1_first_500.json',
        'treesum_pt2': 'summaries_treesum_pt2_last_500.json'
    }
    
    for name, filename in file_patterns.items():
        for base_path in base_paths:
            file_path = base_path / filename
            if file_path.exists():
                with open(file_path, 'r') as f:
                    datasets[name] = json.load(f)
                print(f"✅ Found {filename} at {file_path}")
                break
        else:
            print(f"❌ Missing {filename}")
    
    return datasets

def create_matched_samples(datasets):
    """Create matched samples across datasets."""
    print("🔗 Creating matched samples...")
    
    # Concatenate treesum datasets
    treesum_combined = datasets['treesum_pt1'] + datasets['treesum_pt2']
    
    # Create mappings
    flat_1024_map = {item['sample_idx']: item for item in datasets['flat_1024']}
    flat_overlap_map = {item['sample_idx']: item for item in datasets['flat_overlap']}
    treesum_map = {item['sample_id']: item for item in treesum_combined}
    
    # Find common IDs
    common_ids = set(flat_1024_map.keys()) & set(flat_overlap_map.keys()) & set(treesum_map.keys())
    print(f"✅ Found {len(common_ids)} matching samples")
    
    # Create matched samples
    matched_samples = []
    for sample_id in sorted(common_ids):
        matched_samples.append({
            'sample_id': sample_id,
            'flat_1024': flat_1024_map[sample_id],
            'flat_overlap': flat_overlap_map[sample_id],
            'treesum': treesum_map[sample_id]
        })
    
    return matched_samples, treesum_combined

# Cell 4: Model Initialization
print("🤖 Initializing models...")

def initialize_models():
    """Initialize SummaC-ZS and UniEval models."""
    print("Initializing SummaC-ZS...")
    summac_model = SummaCZS(models=['vitc'], granularity="sentence", device=device, batch_size=8)
    
    print("Initializing UniEval...")
    unieval_evaluator = get_evaluator('summarization', device=device, max_length=4096)
    
    print("✅ Models ready!")
    return summac_model, unieval_evaluator

# Cell 5: Evaluation Functions
def evaluate_summac_batch(model, documents, summaries, batch_size=16):
    """Evaluate SummaC-ZS scores."""
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
    """Evaluate UniEval scores - Simplified version."""
    all_scores = {dim: [] for dim in dimensions}
    
    for i in tqdm(range(0, len(output_list), batch_size), desc="UniEval"):
        batch_out = output_list[i:i+batch_size]
        
        try:
            filtered_out = [out for out in batch_out if out.strip()]
            
            if not filtered_out:
                for dim in dimensions:
                    all_scores[dim].extend([0.0] * len(batch_out))
                continue
            
            # Simplified: Use basic data format
            data = convert_to_json(output_list=filtered_out)
            
            # Try to evaluate all dimensions at once first
            try:
                eval_scores = evaluator.evaluate(data, dims=dimensions, overall=False, print_result=False)
                
                if isinstance(eval_scores, list) and len(eval_scores) > 0:
                    for dim in dimensions:
                        dim_scores = [sample_scores.get(dim, 0.0) for sample_scores in eval_scores]
                        all_scores[dim].extend(dim_scores)
                else:
                    for dim in dimensions:
                        all_scores[dim].extend([0.0] * len(filtered_out))
                        
            except Exception as e:
                print(f"UniEval batch error, trying individual dims: {e}")
                # Fallback: evaluate each dimension separately
                for dim in dimensions:
                    try:
                        eval_scores = evaluator.evaluate(data, dims=[dim], overall=False, print_result=False)
                        
                        if isinstance(eval_scores, list) and len(eval_scores) > 0:
                            dim_scores = [sample_scores.get(dim, 0.0) for sample_scores in eval_scores]
                            all_scores[dim].extend(dim_scores)
                        else:
                            all_scores[dim].extend([0.0] * len(filtered_out))
                    except Exception as e2:
                        print(f"Error in UniEval {dim}: {e2}")
                        all_scores[dim].extend([0.0] * len(filtered_out))
                    
        except Exception as e:
            print(f"Error in UniEval batch: {e}")
            for dim in dimensions:
                all_scores[dim].extend([0.0] * len(batch_out))
    
    return all_scores

# Cell 6: Main Evaluation
print("🚀 Starting evaluation...")

# Load datasets
datasets = find_dataset_files()
if None in datasets.values():
    print("❌ Missing datasets. Please upload your JSON files to /kaggle/input/")
    print("Required files:")
    print("- summaries_flat_1024.json")
    print("- summaries_flat_overlap.json") 
    print("- summaries_treesum_pt1_first_500.json")
    print("- summaries_treesum_pt2_last_500.json")
else:
    # Create matched samples
    matched_samples, treesum_combined = create_matched_samples(datasets)
    
    # Initialize models
    summac_model, unieval_evaluator = initialize_models()
    
    # Create document mapping
    document_map = {sample['sample_id']: sample['document'] for sample in treesum_combined}
    
    # Prepare data for all datasets
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
    
    # Evaluate all datasets
    results = {}
    
    for dataset_name, data in datasets_info.items():
        print(f"\n📊 Evaluating {dataset_name} ({len(data['docs'])} samples)...")
        
        # SummaC-ZS evaluation
        summac_scores = evaluate_summac_batch(summac_model, data['docs'], data['generated'])
        
        # UniEval evaluation
        unieval_scores = evaluate_unieval_batch(unieval_evaluator, data['generated'])
        
        results[dataset_name] = {
            'summac_scores': summac_scores,
            'unieval_fluency': unieval_scores['fluency'],
            'unieval_naturalness': unieval_scores['naturalness'],
            'unieval_understandability': unieval_scores['understandability']
        }
        
        print(f"✅ {dataset_name} completed!")
    
    # Cell 7: Save Results
    print("\n💾 Saving results...")
    
    # Create detailed results DataFrame
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
    df.to_csv('/kaggle/working/detailed_evaluation_results.csv', index=False)
    
    # Create summary statistics
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
    summary_df.to_csv('/kaggle/working/evaluation_summary.csv', index=False)
    
    # Save JSON results
    with open('/kaggle/working/evaluation_results.json', 'w') as f:
        json.dump({
            'detailed_results': results_data,
            'summary_statistics': summary_stats,
            'num_samples': len(matched_samples)
        }, f, indent=2)
    
    # Display summary
    print("\n" + "="*80)
    print("🎉 EVALUATION COMPLETE!")
    print("="*80)
    print(f"📊 Processed {len(matched_samples)} samples")
    print(f"🚀 Device used: {device}")
    print(f"💾 Results saved to /kaggle/working/")
    
    print("\n📈 SUMMARY STATISTICS:")
    print(summary_df.to_string(index=False))
    
    print(f"\n📄 Files created:")
    print(f"   - detailed_evaluation_results.csv")
    print(f"   - evaluation_summary.csv")
    print(f"   - evaluation_results.json")
    
    print("\n✅ Ready to download results from Kaggle!")
