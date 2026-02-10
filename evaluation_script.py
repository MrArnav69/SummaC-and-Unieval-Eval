#!/usr/bin/env python3
"""
Evaluation script for SummaC and UniEval metrics on summary datasets.
Evaluates Coherence and Fluency dimensions on 1000 matched samples across 3 datasets.
"""

import json
import sys
import os
import numpy as np
from typing import Dict, List, Tuple
import pandas as pd
from tqdm import tqdm

# Add UniEval to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'UniEval'))

from summac.model_summac import SummaCZS
from UniEval.utils import convert_to_json
from UniEval.metric.evaluator import get_evaluator

def load_datasets():
    """Load and prepare all datasets."""
    print("Loading datasets...")
    
    # Load flat datasets
    with open('summaries_flat_1024.json', 'r') as f:
        flat_1024 = json.load(f)
    
    with open('summaries_flat_overlap.json', 'r') as f:
        flat_overlap = json.load(f)
    
    # Load treesum datasets
    with open('summaries_treesum_pt1_first_500.json', 'r') as f:
        treesum_pt1 = json.load(f)
    
    with open('summaries_treesum_pt2_last_500.json', 'r') as f:
        treesum_pt2 = json.load(f)
    
    # Concatenate treesum datasets
    treesum_combined = treesum_pt1 + treesum_pt2
    
    print(f"Loaded {len(flat_1024)} samples from flat_1024")
    print(f"Loaded {len(flat_overlap)} samples from flat_overlap")
    print(f"Loaded {len(treesum_combined)} samples from treesum_combined")
    
    return flat_1024, flat_overlap, treesum_combined

def create_sample_mappings(flat_1024, flat_overlap, treesum_combined):
    """Create mappings between sample IDs across datasets."""
    # Create ID to sample mappings
    flat_1024_map = {item['sample_idx']: item for item in flat_1024}
    flat_overlap_map = {item['sample_idx']: item for item in flat_overlap}
    treesum_map = {item['sample_id']: item for item in treesum_combined}
    
    # Find common IDs across all datasets
    common_ids = set(flat_1024_map.keys()) & set(flat_overlap_map.keys()) & set(treesum_map.keys())
    
    print(f"Found {len(common_ids)} common samples across all datasets")
    
    # Create matched samples list
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
    """Initialize SummaC-ZS model optimized for P100 GPU with no limits."""
    print("Initializing SummaC-ZS model for P100 GPU (no truncation limits)...")
    model = SummaCZS(models=['vitc'], granularity="sentence", device="cuda", batch_size=8)  # Smaller batch for P100
    return model

def initialize_unieval():
    """Initialize UniEval evaluator optimized for P100 GPU with no limits."""
    print("Initializing UniEval evaluator for P100 GPU (no truncation limits)...")
    evaluator = get_evaluator('summarization', device='cuda', max_length=4096)  # Higher limit for P100
    return evaluator

def evaluate_summac_batch(model, documents, summaries, batch_size=16):
    """Evaluate SummaC-ZS scores in batches optimized for P100 GPU with NO limits."""
    scores = []
    
    # Process in batches optimized for P100
    for i in tqdm(range(0, len(documents), batch_size), desc="SummaC-ZS evaluation"):
        batch_docs = documents[i:i+batch_size]
        batch_sums = summaries[i:i+batch_size]
        
        try:
            # Filter valid pairs - NO TRUNCATION
            valid_docs = []
            valid_sums = []
            for doc, summary in zip(batch_docs, batch_sums):
                if doc.strip() and summary.strip():
                    # NO LENGTH LIMITS - use full documents
                    valid_docs.append(doc)
                    valid_sums.append(summary)
            
            if not valid_docs:
                scores.extend([0.0] * len(batch_docs))
                continue
            
            # GPU batch processing with full documents
            result = model.score(valid_docs, valid_sums)
            
            if result and 'scores' in result and len(result['scores']) > 0:
                scores.extend(result['scores'])
            else:
                scores.extend([0.0] * len(valid_docs))
                
        except Exception as e:
            print(f"Error in SummaC-ZS evaluation: {e}")
            scores.extend([0.0] * len(batch_docs))
    
    return scores

def evaluate_unieval_batch(evaluator, output_list, dimensions=['fluency', 'naturalness', 'understandability'], batch_size=32):
    """Evaluate UniEval scores optimized for P100 GPU with NO limits."""
    all_scores = {dim: [] for dim in dimensions}
    
    # Process in batches optimized for P100
    for i in tqdm(range(0, len(output_list), batch_size), desc="UniEval evaluation"):
        batch_out = output_list[i:i+batch_size]
        
        try:
            # Filter and prepare batch - NO TRUNCATION
            filtered_out = []
            for out in batch_out:
                if out.strip():
                    # NO LENGTH LIMITS - use full summaries
                    filtered_out.append(out)
                else:
                    filtered_out.append("Empty summary")
            
            if not filtered_out:
                for dim in dimensions:
                    all_scores[dim].extend([0.0] * len(batch_out))
                continue
            
            # GPU batch processing with full texts
            data = convert_to_json(output_list=filtered_out)
            eval_scores = evaluator.evaluate(data, dims=dimensions, overall=False, print_result=False)
            
            # Process results
            if isinstance(eval_scores, list) and len(eval_scores) > 0:
                for dim in dimensions:
                    dim_scores = []
                    for sample_scores in eval_scores:
                        if isinstance(sample_scores, dict) and dim in sample_scores:
                            dim_scores.append(sample_scores[dim])
                        else:
                            dim_scores.append(0.0)
                    all_scores[dim].extend(dim_scores)
            else:
                for dim in dimensions:
                    all_scores[dim].extend([0.0] * len(filtered_out))
                    
        except Exception as e:
            print(f"Error in UniEval evaluation: {e}")
            for dim in dimensions:
                all_scores[dim].extend([0.0] * len(batch_out))
    
    return all_scores

def main():
    """Main evaluation function optimized for P100 GPU with NO limits."""
    print("Starting GPU-optimized evaluation of SummaC-ZS and UniEval metrics (NO TRUNCATION)...")
    
    # Set up GPU optimization for P100
    import torch
    if torch.cuda.is_available():
        print(f"GPU detected: {torch.cuda.get_device_name()}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
        torch.backends.cudnn.deterministic = False  # Allow non-deterministic for speed
        print("⚠️  NO TRUNCATION - Processing full documents for maximum accuracy")
    
    # Load datasets
    flat_1024, flat_overlap, treesum_combined = load_datasets()
    
    # Create matched samples
    matched_samples = create_sample_mappings(flat_1024, flat_overlap, treesum_combined)
    
    # For P100 GPU, moderate batch size but no truncation
    test_samples = matched_samples[:100]  # Start with 100, can scale to 1000
    print(f"Testing with {len(test_samples)} samples on P100 GPU (NO LIMITS)...")
    
    # Initialize models
    summac_model = initialize_summac()
    unieval_evaluator = initialize_unieval()
    
    # Create a mapping of sample_id to document from treesum (since it has the documents)
    document_map = {sample['sample_id']: sample['document'] for sample in treesum_combined}
    
    # Prepare data for each dataset (using test samples)
    datasets_info = {
        'flat_1024': {
            'docs': [document_map[sample['sample_id']] for sample in test_samples],
            'generated': [sample['flat_1024']['generated_summary'] for sample in test_samples],
            'reference': [sample['flat_1024']['reference_summary'] for sample in test_samples]
        },
        'flat_overlap': {
            'docs': [document_map[sample['sample_id']] for sample in test_samples],
            'generated': [sample['flat_overlap']['generated_summary'] for sample in test_samples],
            'reference': [sample['flat_overlap']['reference_summary'] for sample in test_samples]
        },
        'treesum': {
            'docs': [sample['treesum']['document'] for sample in test_samples],
            'generated': [sample['treesum']['generated_summary'] for sample in test_samples],
            'reference': [sample['treesum']['reference_summary'] for sample in test_samples]
        }
    }
    
    # Results storage
    results = {}
    
    # Evaluate each dataset
    for dataset_name, data in datasets_info.items():
        print(f"\nEvaluating {dataset_name} dataset...")
        
        # SummaC-ZS evaluation
        print("Running SummaC-ZS evaluation...")
        summac_scores = evaluate_summac_batch(
            summac_model, 
            data['docs'], 
            data['generated']
        )
        
        # UniEval evaluation (source-independent dimensions only)
        print("Running UniEval evaluation...")
        unieval_scores = evaluate_unieval_batch(
            unieval_evaluator,
            data['generated'],
            dimensions=['fluency', 'naturalness', 'understandability']
        )
        
        # Store results
        results[dataset_name] = {
            'summac_scores': summac_scores,
            'unieval_fluency': unieval_scores['fluency'],
            'unieval_naturalness': unieval_scores['naturalness'],
            'unieval_understandability': unieval_scores['understandability']
        }
        
        # Print basic statistics
        print(f"SummaC-ZS - Mean: {np.mean(summac_scores):.4f}, Std: {np.std(summac_scores):.4f}")
        print(f"UniEval Fluency - Mean: {np.mean(unieval_scores['fluency']):.4f}, Std: {np.std(unieval_scores['fluency']):.4f}")
        print(f"UniEval Naturalness - Mean: {np.mean(unieval_scores['naturalness']):.4f}, Std: {np.std(unieval_scores['naturalness']):.4f}")
        print(f"UniEval Understandability - Mean: {np.mean(unieval_scores['understandability']):.4f}, Std: {np.std(unieval_scores['understandability']):.4f}")
    
    # Create comprehensive results DataFrame
    print("\nCreating comprehensive results...")
    
    # Prepare data for DataFrame
    results_data = []
    for i, sample in enumerate(test_samples):
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
    
    # Save detailed results
    df.to_csv('test_evaluation_results.csv', index=False)
    print("Test results saved to 'test_evaluation_results.csv'")
    
    # Generate summary statistics
    summary_stats = []
    metrics = ['summac', 'fluency', 'naturalness', 'understandability']
    datasets = ['flat_1024', 'flat_overlap', 'treesum']
    
    for metric in metrics:
        for dataset in datasets:
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
    summary_df.to_csv('test_evaluation_summary.csv', index=False)
    print("Test summary statistics saved to 'test_evaluation_summary.csv'")
    
    # Print summary table
    print("\n" + "="*80)
    print("TEST EVALUATION SUMMARY")
    print("="*80)
    print(summary_df.to_string(index=False))
    
    # Save results as JSON
    with open('test_evaluation_results.json', 'w') as f:
        json.dump({
            'detailed_results': results_data,
            'summary_statistics': summary_stats,
            'num_samples': len(test_samples)
        }, f, indent=2)
    
    print(f"\nTest evaluation completed for {len(test_samples)} samples!")
    print("If results look good, you can run the full evaluation on all 1000 samples.")
    print("Results saved to:")
    print("- test_evaluation_results.csv")
    print("- test_evaluation_summary.csv") 
    print("- test_evaluation_results.json")

if __name__ == "__main__":
    main()
