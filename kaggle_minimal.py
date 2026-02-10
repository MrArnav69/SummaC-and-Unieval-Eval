#!/usr/bin/env python3
"""
Kaggle Minimal Script - Bypasses all dependency conflicts
Uses only essential packages and manual UniEval setup
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

# Only install summac - let Kaggle handle the rest
print("📦 Installing summac only...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "summac", "-q"])

print("✅ Setup complete!")

# Cell 2: Imports and GPU Check
print("🔍 Checking environment...")

import torch
from summac.model_summac import SummaCZS

# GPU Check
if torch.cuda.is_available():
    print(f"✅ GPU: {torch.cuda.get_device_name()}")
    print(f"✅ Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    torch.backends.cudnn.benchmark = True
    device = "cuda"
else:
    print("⚠️  Using CPU - this will be slow!")
    device = "cpu"

# Cell 3: Manual UniEval Setup (bypass pip conflicts)
print("🔧 Setting up UniEval manually...")

# Clone UniEval
if not os.path.exists("UniEval"):
    subprocess.check_call(["git", "clone", "https://github.com/maszhongming/UniEval.git"])

# Add to path
sys.path.append('/kaggle/working/UniEval')
sys.path.append('/kaggle/working')

# Import UniEval components manually
try:
    from UniEval.metric.evaluator import get_evaluator
    from UniEval.utils import convert_to_json
    print("✅ UniEval imported successfully!")
except Exception as e:
    print(f"⚠️  UniEval import error: {e}")
    print("🔄 Using fallback evaluation...")
    
    # Fallback: Simple rule-based evaluation
    def fallback_fluency(text):
        """Simple fluency evaluation based on text properties."""
        if not text or not text.strip():
            return 0.0
        
        words = text.split()
        sentences = text.split('.') + text.split('!') + text.split('?')
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Basic fluency metrics
        avg_word_length = np.mean([len(word) for word in words]) if words else 0
        sentence_count = len(sentences)
        word_count = len(words)
        
        # Normalize scores (0-1 range)
        fluency_score = min(1.0, (sentence_count / max(1, word_count/10)) * (1.0 / max(1.0, avg_word_length/5)))
        return fluency_score
    
    def fallback_naturalness(text):
        """Simple naturalness evaluation."""
        if not text or not text.strip():
            return 0.0
        
        # Check for unnatural patterns
        unnatural_patterns = ['the the', 'very very', 'really really', 'much much']
        pattern_count = sum(1 for pattern in unnatural_patterns if pattern in text.lower())
        
        # Penalize repetitive patterns
        words = text.lower().split()
        unique_words = set(words)
        repetition_ratio = 1 - (len(unique_words) / max(1, len(words)))
        
        naturalness = max(0.0, 1.0 - (pattern_count * 0.1) - (repetition_ratio * 0.2))
        return naturalness
    
    def fallback_understandability(text):
        """Simple understandability evaluation."""
        if not text or not text.strip():
            return 0.0
        
        # Basic readability metrics
        words = text.split()
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        
        avg_sentence_length = np.mean([len(s.split()) for s in sentences]) if sentences else 0
        complex_words = sum(1 for word in words if len(word) > 6)
        word_count = len(words)
        
        # Score based on sentence complexity and word length
        complexity_penalty = min(0.5, complex_words / max(1, word_count))
        length_penalty = min(0.3, avg_sentence_length / 20 if sentences else 0)
        
        understandability = max(0.0, 1.0 - complexity_penalty - length_penalty)
        return understandability

# Cell 4: Data Loading Functions
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

# Cell 5: Model Initialization
print("🤖 Initializing models...")

def initialize_models():
    """Initialize SummaC-ZS and UniEval models."""
    print("Initializing SummaC-ZS...")
    summac_model = SummaCZS(models=['vitc'], granularity="sentence", device=device, batch_size=8)
    
    # Try to initialize UniEval
    try:
        unieval_evaluator = get_evaluator('summarization', device=device, max_length=4096)
        print("✅ UniEval initialized successfully!")
        use_fallback = False
    except Exception as e:
        print(f"⚠️  UniEval failed to initialize: {e}")
        print("🔄 Using fallback evaluation methods...")
        unieval_evaluator = None
        use_fallback = True
    
    print("✅ Models ready!")
    return summac_model, unieval_evaluator, use_fallback

# Cell 6: Evaluation Functions
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

def evaluate_unieval_batch(evaluator, output_list, use_fallback, batch_size=32):
    """Evaluate UniEval scores with fallback support."""
    dimensions = ['fluency', 'naturalness', 'understandability']
    all_scores = {dim: [] for dim in dimensions}
    
    if use_fallback:
        print("🔄 Using fallback evaluation methods...")
        # Use fallback functions
        for i in tqdm(range(0, len(output_list)), desc="Fallback Eval"):
            text = output_list[i]
            if text.strip():
                all_scores['fluency'].append(fallback_fluency(text))
                all_scores['naturalness'].append(fallback_naturalness(text))
                all_scores['understandability'].append(fallback_understandability(text))
            else:
                for dim in dimensions:
                    all_scores[dim].append(0.0)
    else:
        print("📊 Using UniEval evaluation...")
        # Use UniEval
        for i in tqdm(range(0, len(output_list), batch_size), desc="UniEval"):
            batch_out = output_list[i:i+batch_size]
            
            try:
                filtered_out = [out for out in batch_out if out.strip()]
                
                if not filtered_out:
                    for dim in dimensions:
                        all_scores[dim].extend([0.0] * len(batch_out))
                    continue
                
                data = convert_to_json(output_list=filtered_out)
                
                # Try each dimension separately
                for dim in dimensions:
                    try:
                        eval_scores = evaluator.evaluate(data, dims=[dim], overall=False, print_result=False)
                        
                        if isinstance(eval_scores, list) and len(eval_scores) > 0:
                            dim_scores = [sample_scores.get(dim, 0.0) for sample_scores in eval_scores]
                            all_scores[dim].extend(dim_scores)
                        else:
                            all_scores[dim].extend([0.0] * len(filtered_out))
                            
                    except Exception as e:
                        print(f"Error in UniEval {dim}: {e}")
                        all_scores[dim].extend([0.0] * len(filtered_out))
                        
            except Exception as e:
                print(f"Error in UniEval batch: {e}")
                for dim in dimensions:
                    all_scores[dim].extend([0.0] * len(batch_out))
    
    return all_scores

# Cell 7: Main Evaluation
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
    summac_model, unieval_evaluator, use_fallback = initialize_models()
    
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
        unieval_scores = evaluate_unieval_batch(unieval_evaluator, data['generated'], use_fallback)
        
        results[dataset_name] = {
            'summac_scores': summac_scores,
            'unieval_fluency': unieval_scores['fluency'],
            'unieval_naturalness': unieval_scores['naturalness'],
            'unieval_understandability': unieval_scores['understandability']
        }
        
        print(f"✅ {dataset_name} completed!")
    
    # Cell 8: Save Results
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
            'num_samples': len(matched_samples),
            'use_fallback': use_fallback
        }, f, indent=2)
    
    # Display summary
    print("\n" + "="*80)
    print("🎉 EVALUATION COMPLETE!")
    print("="*80)
    print(f"📊 Processed {len(matched_samples)} samples")
    print(f"🚀 Device used: {device}")
    print(f"🔄 Used fallback: {use_fallback}")
    print(f"💾 Results saved to /kaggle/working/")
    
    print("\n📈 SUMMARY STATISTICS:")
    print(summary_df.to_string(index=False))
    
    print(f"\n📄 Files created:")
    print(f"   - detailed_evaluation_results.csv")
    print(f"   - evaluation_summary.csv")
    print(f"   - evaluation_results.json")
    
    if use_fallback:
        print("\n⚠️  NOTE: Fallback evaluation methods were used")
        print("   - Results are approximate estimates")
        print("   - For full accuracy, fix UniEval dependencies")
    else:
        print("\n✅ Full UniEval evaluation completed!")
    
    print("\n✅ Ready to download results from Kaggle!")
