#!/usr/bin/env python3
"""
Report Generator for Evaluation Results
Generates comprehensive reports comparing metrics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import glob

# Create results directory
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

def load_latest_results(metric_name):
    """Load the latest results for a given metric."""
    pattern = f"{metric_name}_summary_*.csv"
    files = glob.glob(str(RESULTS_DIR / pattern))
    
    if not files:
        return None
    
    # Get the latest file
    latest_file = max(files, key=lambda x: x.split('_')[-1].replace('.csv', ''))
    return pd.read_csv(latest_file), latest_file

def load_detailed_results(metric_name):
    """Load the latest detailed results for a given metric."""
    pattern = f"{metric_name}_detailed_*.csv"
    files = glob.glob(str(RESULTS_DIR / pattern))
    
    if not files:
        return None
    
    # Get the latest file
    latest_file = max(files, key=lambda x: x.split('_')[-1].replace('.csv', ''))
    return pd.read_csv(latest_file), latest_file

def generate_comparison_report():
    """Generate comprehensive comparison report."""
    print("📊 GENERATING COMPREHENSIVE EVALUATION REPORT")
    print("="*60)
    
    # Load available metrics
    metrics = {}
    metric_files = {}
    
    # Try to load UniEval Fluency
    unieval_data = load_latest_results("unieval_fluency")
    if unieval_data:
        metrics['UniEval Fluency'] = unieval_data[0]
        metric_files['UniEval Fluency'] = unieval_data[1]
        print("✅ UniEval Fluency results loaded")
    
    # Try to load AlignScore
    alignscore_data = load_latest_results("alignscore")
    if alignscore_data:
        metrics['AlignScore'] = alignscore_data[0]
        metric_files['AlignScore'] = alignscore_data[1]
        print("✅ AlignScore results loaded")
    
    if not metrics:
        print("❌ No evaluation results found!")
        print("💡 Please run evaluation scripts first")
        return
    
    # Generate timestamped report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = RESULTS_DIR / f"evaluation_report_{timestamp}.md"
    
    with open(report_file, 'w') as f:
        f.write("# Evaluation Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Datasets:** flat_1024, flat_overlap, treesum\n")
        f.write(f"**Total Samples:** 1000\n\n")
        
        # Summary table
        f.write("## Summary Statistics\n\n")
        f.write("| Metric | Dataset | Mean | Std | Min | Max | Median |\n")
        f.write("|--------|---------|------|-----|-----|-----|-------|\n")
        
        for metric_name, data in metrics.items():
            for dataset in data.index:
                row = data.loc[dataset]
                f.write(f"| {metric_name} | {dataset} | {row.iloc[0]:.4f} | {row.iloc[1]:.4f} | {row.iloc[2]:.4f} | {row.iloc[3]:.4f} | {row.iloc[4]:.4f} |\n")
        
        # Detailed analysis
        f.write("\n## Detailed Analysis\n\n")
        
        for metric_name, data in metrics.items():
            f.write(f"### {metric_name}\n\n")
            
            # Dataset comparison
            f.write("#### Dataset Performance\n\n")
            best_dataset = data.iloc[:, 0].idxmax()  # Dataset with highest mean
            worst_dataset = data.iloc[:, 0].idxmin()  # Dataset with lowest mean
            
            f.write(f"- **Best performing:** {best_dataset} (Mean: {data.loc[best_dataset].iloc[0]:.4f})\n")
            f.write(f"- **Worst performing:** {worst_dataset} (Mean: {data.loc[worst_dataset].iloc[0]:.4f})\n")
            
            # Variability analysis
            f.write(f"\n#### Variability Analysis\n\n")
            most_variable = data.iloc[:, 1].idxmax()  # Dataset with highest std
            least_variable = data.iloc[:, 1].idxmin()  # Dataset with lowest std
            
            f.write(f"- **Most variable:** {most_variable} (Std: {data.loc[most_variable].iloc[1]:.4f})\n")
            f.write(f"- **Least variable:** {least_variable} (Std: {data.loc[least_variable].iloc[1]:.4f})\n")
            
            f.write("\n")
        
        # Cross-metric comparison (if multiple metrics)
        if len(metrics) > 1:
            f.write("## Cross-Metric Comparison\n\n")
            f.write("### Dataset Rankings\n\n")
            
            # Create ranking table
            rankings = {}
            for metric_name, data in metrics.items():
                rankings[metric_name] = data.iloc[:, 0].rank(ascending=False).to_dict()
            
            f.write("| Dataset | " + " | ".join(metrics.keys()) + " | Average Rank |\n")
            f.write("|---------|" + "|".join(["-" * len(name) for name in metrics.keys()]) + "|-------------|\n")
            
            all_datasets = set()
            for metric_data in metrics.values():
                all_datasets.update(metric_data.index)
            
            for dataset in sorted(all_datasets):
                ranks = []
                for metric_name in metrics.keys():
                    rank = rankings[metric_name].get(dataset, "N/A")
                    ranks.append(str(rank) if rank != "N/A" else "N/A")
                
                # Calculate average rank (only for numeric ranks)
                numeric_ranks = [rankings[metric_name].get(dataset, float('inf')) 
                               for metric_name in metrics.keys() 
                               if rankings[metric_name].get(dataset, float('inf')) != float('inf')]
                avg_rank = np.mean(numeric_ranks) if numeric_ranks else "N/A"
                
                f.write(f"| {dataset} | " + " | ".join(ranks) + f" | {avg_rank:.1f} |\n")
        
        # Data files information
        f.write("\n## Data Files\n\n")
        f.write("### Source Files\n")
        f.write(f"- UniEval Fluency: `{metric_files.get('UniEval Fluency', 'N/A')}`\n")
        f.write(f"- AlignScore: `{metric_files.get('AlignScore', 'N/A')}`\n")
        
        f.write("\n### Generated Files\n")
        f.write(f"- This report: `{report_file.name}`\n")
        
        # Interpretation guide
        f.write("\n## Interpretation Guide\n\n")
        
        if 'UniEval Fluency' in metrics:
            f.write("### UniEval Fluency\n")
            f.write("- **Range:** 0.0 to 1.0\n")
            f.write("- **Higher values:** Better fluency\n")
            f.write("- **Interpretation:** Text quality, readability, naturalness\n\n")
        
        if 'AlignScore' in metrics:
            f.write("### AlignScore\n")
            f.write("- **Range:** 0.0 to 1.0\n")
            f.write("- **Higher values:** Better factual consistency\n")
            f.write("- **Interpretation:** How well the summary aligns with the source document\n\n")
        
        f.write("---\n")
        f.write("*Report generated automatically*")
    
    print(f"✅ Report generated: {report_file}")
    
    # Generate summary statistics file
    summary_file = RESULTS_DIR / f"combined_summary_{timestamp}.csv"
    
    # Combine all metrics into one summary
    combined_summary = pd.DataFrame()
    
    for metric_name, data in metrics.items():
        data_copy = data.copy()
        data_copy.columns = [f"{metric_name}_{col}" for col in data_copy.columns]
        combined_summary = pd.concat([combined_summary, data_copy], axis=1)
    
    combined_summary.to_csv(summary_file)
    print(f"✅ Combined summary: {summary_file}")
    
    # Print summary to console
    print(f"\n📊 EXECUTIVE SUMMARY:")
    print(f"=" * 50)
    
    for metric_name, data in metrics.items():
        print(f"\n{metric_name}:")
        best = data.iloc[:, 0].idxmax()
        worst = data.iloc[:, 0].idxmin()
        print(f"  Best: {best} ({data.loc[best].iloc[0]:.4f})")
        print(f"  Worst: {worst} ({data.loc[worst].iloc[0]:.4f})")
    
    print(f"\n📁 Files created in '{RESULTS_DIR}' directory:")
    print(f"   📄 {report_file.name} (detailed report)")
    print(f"   📊 {summary_file.name} (combined summary)")

def main():
    """Main report generator."""
    generate_comparison_report()

if __name__ == "__main__":
    main()
