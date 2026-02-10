#!/usr/bin/env python3
"""
A40 GPU Optimized Evaluation Script - High Performance SOTA Metrics
Uses SummaC-ZS and UniEval with maximum efficiency and robustness
"""

import json
import sys
import os
import numpy as np
from typing import Dict, List, Tuple
import pandas as pd
from tqdm import tqdm
import torch
import time
from pathlib import Path

# Add UniEval to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'UniEval'))

from summac.model_summac import SummaCZS
from UniEval.utils import convert_to_json
from UniEval.metric.evaluator import get_evaluator

class A40Optimizer:
    """A40 GPU optimization settings and utilities."""
    
    def __init__(self):
        self.device = "cuda"
        self.setup_gpu()
        
    def setup_gpu(self):
        """Configure A40 GPU for maximum performance."""
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available - A40 GPU required")
            
        print(f"🚀 A40 GPU Detected: {torch.cuda.get_device_name()}")
        print(f"💾 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        # A40 optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.cuda.empty_cache()
        
        # Enable mixed precision for faster processing
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        print("✅ A40 optimizations enabled")
        
    def get_optimal_batch_sizes(self):
        """Return optimal batch sizes for A40 GPU."""
        return {
            'summac': 64,      # A40 can handle 64+ samples
            'unieval': 128      # A40 can handle 128+ samples
        }

def find_datasets():
    """Robust dataset discovery in current directory."""
    print("🔍 Discovering datasets...")
    
    base_dir = Path(".")
    required_files = {
        'flat_1024': 'summaries_flat_1024.json',
        'flat_overlap': 'summaries_flat_overlap.json',
        'treesum_pt1': 'summaries_treesum_pt1_first_500.json',
        'treesum_pt2': 'summaries_treesum_pt2_last_500.json'
    }
    
    datasets = {}
    for name, filename in required_files.items():
        file_path = base_dir / filename
        if file_path.exists():
            with open(file_path, 'r') as f:
                datasets[name] = json.load(f)
            print(f"✅ Found {filename}")
        else:
            raise FileNotFoundError(f"❌ Missing required file: {filename}")
    
    return datasets

def create_matched_samples(datasets):
    """Create matched samples with robust error handling."""
    print("🔗 Creating matched samples...")
    
    # Concatenate treesum datasets
    treesum_combined = datasets['treesum_pt1'] + datasets['treesum_pt2']
    
    # Create mappings with proper key handling
    flat_1024_map = {item['sample_idx']: item for item in datasets['flat_1024']}
    flat_overlap_map = {item['sample_idx']: item for item in datasets['flat_overlap']}
    treesum_map = {item['sample_id']: item for item in treesum_combined}  # treesum uses 'sample_id'
    
    # Find common IDs - convert all to same type for comparison
    flat_1024_ids = set(flat_1024_map.keys())
    flat_overlap_ids = set(flat_overlap_map.keys())
    treesum_ids = set(treesum_map.keys())
    
    common_ids = flat_1024_ids & flat_overlap_ids & treesum_ids
    
    if len(common_ids) == 0:
        raise ValueError("No common sample IDs found across datasets")
    
    print(f"✅ Found {len(common_ids)} matching samples")
    
    # Create matched samples with validation
    matched_samples = []
    for sample_id in sorted(common_ids):
        sample = {
            'sample_id': sample_id,
            'flat_1024': flat_1024_map[sample_id],
            'flat_overlap': flat_overlap_map[sample_id],
            'treesum': treesum_map[sample_id]
        }
        
        # Validate data integrity - check all required keys exist
        required_flat_keys = ['generated_summary', 'reference_summary']
        required_treesum_keys = ['document', 'generated_summary', 'reference_summary']
        
        valid = True
        for dataset_name in ['flat_1024', 'flat_overlap']:
            for key in required_flat_keys:
                if key not in sample[dataset_name]:
                    print(f"⚠️  Sample {sample_id} missing '{key}' in {dataset_name}")
                    valid = False
        
        for key in required_treesum_keys:
            if key not in sample['treesum']:
                print(f"⚠️  Sample {sample_id} missing '{key}' in treesum")
                valid = False
        
        if valid:
            matched_samples.append(sample)
        else:
            print(f"⚠️  Skipping sample {sample_id} - incomplete data")
    
    print(f"✅ {len(matched_samples)} valid samples after validation")
    return matched_samples, treesum_combined

class SOTAEvaluator:
    """State-of-the-Art evaluator with robust error handling."""
    
    def __init__(self, optimizer: A40Optimizer):
        self.optimizer = optimizer
        self.batch_sizes = optimizer.get_optimal_batch_sizes()
        self.summac_model = None
        self.unieval_evaluator = None
        
    def initialize_models(self):
        """Initialize SOTA models with A40 optimization."""
        print("🤖 Initializing SOTA models...")
        
        # Initialize SummaC-ZS (SOTA factual consistency)
        print("📊 Initializing SummaC-ZS (SOTA factual consistency)...")
        self.summac_model = SummaCZS(
            models=['vitc'], 
            granularity="sentence", 
            device=self.optimizer.device, 
            batch_size=self.batch_sizes['summac']
        )
        
        # Initialize UniEval (SOTA multi-dimensional evaluation)
        print("🎯 Initializing UniEval (SOTA multi-dimensional evaluation)...")
        self.unieval_evaluator = get_evaluator(
            'summarization', 
            device=self.optimizer.device, 
            max_length=8192  # A40 can handle longer sequences
        )
        
        print("✅ SOTA models initialized")
        
    def evaluate_summac_batch(self, documents: List[str], summaries: List[str]) -> List[float]:
        """High-performance SummaC-ZS evaluation."""
        scores = []
        batch_size = self.batch_sizes['summac']
        
        for i in tqdm(range(0, len(documents), batch_size), desc="📊 SummaC-ZS"):
            batch_docs = documents[i:i+batch_size]
            batch_sums = summaries[i:i+batch_size]
            
            try:
                # Filter valid pairs
                valid_pairs = [(doc, summary) for doc, summary in zip(batch_docs, batch_sums) 
                             if doc.strip() and summary.strip()]
                
                if not valid_pairs:
                    scores.extend([0.0] * len(batch_docs))
                    continue
                
                valid_docs, valid_sums = zip(*valid_pairs)
                
                # GPU batch processing
                result = self.summac_model.score(list(valid_docs), list(valid_sums))
                
                if result and 'scores' in result and len(result['scores']) > 0:
                    scores.extend(result['scores'])
                else:
                    scores.extend([0.0] * len(valid_docs))
                    
            except Exception as e:
                print(f"⚠️  SummaC-ZS batch error: {e}")
                scores.extend([0.0] * len(batch_docs))
        
        return scores
    
    def evaluate_unieval_batch(self, summaries: List[str], 
                            dimensions: List[str] = ['fluency', 'naturalness', 'understandability']) -> Dict[str, List[float]]:
        """High-performance UniEval evaluation with robust error handling."""
        all_scores = {dim: [] for dim in dimensions}
        batch_size = self.batch_sizes['unieval']
        
        for i in tqdm(range(0, len(summaries), batch_size), desc="🎯 UniEval"):
            batch_sums = summaries[i:i+batch_size]
            
            try:
                # Filter and prepare batch
                filtered_sums = [s if s.strip() else "Empty summary" for s in batch_sums]
                
                # Prepare data for UniEval
                data = convert_to_json(output_list=filtered_sums)
                
                # Evaluate each dimension separately for robustness
                for dim in dimensions:
                    try:
                        eval_scores = self.unieval_evaluator.evaluate(
                            data, dims=[dim], overall=False, print_result=False
                        )
                        
                        if isinstance(eval_scores, list) and len(eval_scores) > 0:
                            dim_scores = [sample_scores.get(dim, 0.0) for sample_scores in eval_scores]
                            all_scores[dim].extend(dim_scores)
                        else:
                            all_scores[dim].extend([0.0] * len(filtered_sums))
                            
                    except Exception as e:
                        print(f"⚠️  UniEval {dim} error: {e}")
                        all_scores[dim].extend([0.0] * len(filtered_sums))
                        
            except Exception as e:
                print(f"⚠️  UniEval batch error: {e}")
                for dim in dimensions:
                    all_scores[dim].extend([0.0] * len(batch_sums))
        
        return all_scores

def run_evaluation():
    """Main evaluation function optimized for A40 GPU."""
    print("="*80)
    print("🚀 A40 GPU OPTIMIZED SOTA EVALUATION")
    print("="*80)
    
    start_time = time.time()
    
    try:
        # Initialize A40 optimizer
        optimizer = A40Optimizer()
        
        # Find and load datasets
        datasets = find_datasets()
        
        # Create matched samples
        matched_samples, treesum_combined = create_matched_samples(datasets)
        
        # Initialize SOTA evaluator
        evaluator = SOTAEvaluator(optimizer)
        evaluator.initialize_models()
        
        # Prepare data for all datasets
        document_map = {sample['sample_id']: sample['document'] for sample in treesum_combined}
        
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
            summac_scores = evaluator.evaluate_summac_batch(data['docs'], data['generated'])
            
            # UniEval evaluation
            unieval_scores = evaluator.evaluate_unieval_batch(data['generated'])
            
            results[dataset_name] = {
                'summac_scores': summac_scores,
                'unieval_fluency': unieval_scores['fluency'],
                'unieval_naturalness': unieval_scores['naturalness'],
                'unieval_understandability': unieval_scores['understandability']
            }
            
            # Print quick stats
            print(f"   ✅ SummaC-ZS: {np.mean(summac_scores):.4f} ± {np.std(summac_scores):.4f}")
            print(f"   ✅ Fluency: {np.mean(unieval_scores['fluency']):.4f} ± {np.std(unieval_scores['fluency']):.4f}")
            print(f"   ✅ Naturalness: {np.mean(unieval_scores['naturalness']):.4f} ± {np.std(unieval_scores['naturalness']):.4f}")
            print(f"   ✅ Understandability: {np.mean(unieval_scores['understandability']):.4f} ± {np.std(unieval_scores['understandability']):.4f}")
        
        # Save comprehensive results
        save_results(results, matched_samples, time.time() - start_time)
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        raise

def save_results(results: Dict, matched_samples: List[Dict], execution_time: float):
    """Save comprehensive results with multiple formats."""
    print("\n💾 Saving comprehensive results...")
    
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
    
    # Save detailed results
    df.to_csv('a40_evaluation_results.csv', index=False)
    
    # Generate and save summary statistics
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
                'Median': np.median(scores),
                'Q1': np.percentile(scores, 25),
                'Q3': np.percentile(scores, 75)
            })
    
    summary_df = pd.DataFrame(summary_stats)
    summary_df.to_csv('a40_evaluation_summary.csv', index=False)
    
    # Save JSON results
    with open('a40_evaluation_results.json', 'w') as f:
        json.dump({
            'detailed_results': results_data,
            'summary_statistics': summary_stats,
            'metadata': {
                'num_samples': len(matched_samples),
                'execution_time_seconds': execution_time,
                'gpu': 'A40',
                'models': ['SummaC-ZS', 'UniEval'],
                'metrics': ['factual_consistency', 'fluency', 'naturalness', 'understandability']
            }
        }, f, indent=2)
    
    # Display final summary
    print("\n" + "="*80)
    print("🎉 A40 GPU SOTA EVALUATION COMPLETE!")
    print("="*80)
    print(f"📊 Processed {len(matched_samples)} samples")
    print(f"⚡ Execution time: {execution_time:.2f} seconds")
    print(f"🚀 GPU: A40 (48GB VRAM)")
    print(f"💾 Results saved:")
    print(f"   - a40_evaluation_results.csv (detailed scores)")
    print(f"   - a40_evaluation_summary.csv (summary statistics)")
    print(f"   - a40_evaluation_results.json (complete data)")
    
    print("\n📈 SUMMARY STATISTICS:")
    print(summary_df.to_string(index=False))
    
    print(f"\n✅ Ready for analysis!")

if __name__ == "__main__":
    run_evaluation()
