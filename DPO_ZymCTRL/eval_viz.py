"""
ZymCTRL Evaluation Visualizer
============================

This module provides visualization and analysis tools for ZymCTRL evaluation results.
It works in conjunction with eval_harness.py to create plots and statistical reports.

Usage
-----
After running evaluations with eval_harness.py, analyze the results:
```
python eval_viz.py \
    --results_dir "path/to/results" \
    --output_dir "analysis_results"
```

The script expects the results directory to contain:
- performance_results.json (from performance evaluation)
- controllability_results.json (from controllability evaluation)
- base_performance_results.json (optional, for model preservation comparison)

Output
------
Generates:
1. Performance Analysis
   - Comparison plots between base and finetuned models
   - Statistical tests for perplexity and stability
   - Detailed performance report

2. Controllability Analysis
   - Distribution plots for different stability levels
   - Statistical comparisons between stability tags
   - Comprehensive controllability report
"""

import os
from typing import Dict, Any, List
import json
from pathlib import Path

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

def _safe_stat(values: List[float]) -> Dict[str, float]:
    """Return basic stats without crashing on empty lists."""
    if not values:
        return {k: float("nan") for k in ("mean", "std", "min", "max")}
    
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }

class PerformanceVisualizer:
    """Visualizes and analyzes model performance comparisons."""
    
    def __init__(self, base_results: Dict[str, Any], finetuned_results: Dict[str, Any]):
        self.base = base_results
        self.finetuned = finetuned_results
    
    def plot_comparisons(self, output_dir: str) -> None:
        """Create comparison plots between base and finetuned models."""
        os.makedirs(output_dir, exist_ok=True)
        
        plt.figure(figsize=(12, 6))
        
        # Perplexity comparison
        plt.subplot(1, 2, 1)
        data = [
            self.base["metrics"]["perplexity"],
            self.finetuned["metrics"]["perplexity"]
        ]
        sns.boxplot(data=data)
        plt.xticks([0, 1], ["Base Model", "Finetuned Model"])
        plt.title("Baseline Perplexity Comparison")
        plt.ylabel("Perplexity")
        
        # Stability comparison
        plt.subplot(1, 2, 2)
        data = [
            self.base["metrics"]["stability"],
            self.finetuned["metrics"]["stability"]
        ]
        sns.boxplot(data=data)
        plt.xticks([0, 1], ["Base Model", "Finetuned Model"])
        plt.title("Baseline Stability Comparison")
        plt.ylabel("ΔG (kcal/mol)")
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "performance_comparison.png"))
        plt.close()
    
    def generate_report(self, output_dir: str) -> None:
        """Generate statistical analysis report."""
        os.makedirs(output_dir, exist_ok=True)
        
        with open(os.path.join(output_dir, "performance_report.txt"), "w") as f:
            f.write("Model Performance Comparison\n")
            f.write("=" * 50 + "\n\n")
            
            # Perplexity comparison
            base_ppl = self.base["metrics"]["perplexity"]
            ft_ppl = self.finetuned["metrics"]["perplexity"]
            t_stat, p_val = stats.ttest_ind(base_ppl, ft_ppl, equal_var=False)
            
            f.write("Perplexity Analysis:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Base Model:      mean={np.mean(base_ppl):.2f}, std={np.std(base_ppl):.2f}\n")
            f.write(f"Finetuned Model: mean={np.mean(ft_ppl):.2f}, std={np.std(ft_ppl):.2f}\n")
            f.write(f"T-test p-value:  {p_val:.4f}\n\n")
            
            # Stability comparison
            base_stab = self.base["metrics"]["stability"]
            ft_stab = self.finetuned["metrics"]["stability"]
            t_stat, p_val = stats.ttest_ind(base_stab, ft_stab, equal_var=False)
            
            f.write("Stability Analysis:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Base Model:      mean={np.mean(base_stab):.2f}, std={np.std(base_stab):.2f}\n")
            f.write(f"Finetuned Model: mean={np.mean(ft_stab):.2f}, std={np.std(ft_stab):.2f}\n")
            f.write(f"T-test p-value:  {p_val:.4f}\n")

class ControllabilityVisualizer:
    """Visualizes and analyzes controllability results."""
    
    def __init__(self, results: Dict[str, Any]):
        self.results = results
    
    def plot_distributions(self, output_dir: str) -> None:
        """Create distribution plots for different stability levels."""
        os.makedirs(output_dir, exist_ok=True)
        
        plt.figure(figsize=(12, 6))
        
        # Stability distribution
        plt.subplot(1, 2, 1)
        data = [self.results[tag]["metrics"]["stability"] for tag in ["high", "medium", "low"]]
        sns.boxplot(data=data)
        plt.xticks(range(3), ["High", "Medium", "Low"])
        plt.title("Stability Distribution by Control Level")
        plt.ylabel("ΔG (kcal/mol)")
        
        # Perplexity distribution
        plt.subplot(1, 2, 2)
        data = [self.results[tag]["metrics"]["perplexity"] for tag in ["high", "medium", "low"]]
        sns.boxplot(data=data)
        plt.xticks(range(3), ["High", "Medium", "Low"])
        plt.title("Perplexity Distribution by Control Level")
        plt.ylabel("Perplexity")
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "controllability_distribution.png"))
        plt.close()
    
    def generate_report(self, output_dir: str) -> None:
        """Generate statistical analysis report."""
        os.makedirs(output_dir, exist_ok=True)
        
        with open(os.path.join(output_dir, "controllability_report.txt"), "w") as f:
            f.write("Controllability Analysis\n")
            f.write("=" * 50 + "\n\n")
            
            # Statistics for each level
            for tag in ["high", "medium", "low"]:
                f.write(f"\n{tag.capitalize()} Stability Level:\n")
                f.write("-" * 30 + "\n")
                
                metrics = self.results[tag]["metrics"]
                stab_stats = _safe_stat(metrics["stability"])
                ppl_stats = _safe_stat(metrics["perplexity"])
                
                f.write("Stability:\n")
                f.write(f"  Mean: {stab_stats['mean']:.2f}\n")
                f.write(f"  Std:  {stab_stats['std']:.2f}\n")
                f.write(f"  Range: [{stab_stats['min']:.2f}, {stab_stats['max']:.2f}]\n\n")
                
                f.write("Perplexity:\n")
                f.write(f"  Mean: {ppl_stats['mean']:.2f}\n")
                f.write(f"  Std:  {ppl_stats['std']:.2f}\n")
                f.write(f"  Range: [{ppl_stats['min']:.2f}, {ppl_stats['max']:.2f}]\n")
            
            # Pairwise comparisons
            f.write("\nPairwise Comparisons:\n")
            f.write("=" * 50 + "\n")
            
            for i, tag1 in enumerate(["high", "medium", "low"]):
                for tag2 in ["high", "medium", "low"][i+1:]:
                    f.write(f"\n{tag1.capitalize()} vs {tag2.capitalize()}:\n")
                    f.write("-" * 30 + "\n")
                    
                    # Stability comparison
                    stab1 = self.results[tag1]["metrics"]["stability"]
                    stab2 = self.results[tag2]["metrics"]["stability"]
                    t_stat, p_val = stats.ttest_ind(stab1, stab2, equal_var=False)
                    
                    f.write("Stability:\n")
                    f.write(f"  Mean difference: {np.mean(stab1) - np.mean(stab2):.2f}\n")
                    f.write(f"  T-test p-value: {p_val:.4f}\n")
                    
                    # Perplexity comparison
                    ppl1 = self.results[tag1]["metrics"]["perplexity"]
                    ppl2 = self.results[tag2]["metrics"]["perplexity"]
                    t_stat, p_val = stats.ttest_ind(ppl1, ppl2, equal_var=False)
                    
                    f.write("\nPerplexity:\n")
                    f.write(f"  Mean difference: {np.mean(ppl1) - np.mean(ppl2):.2f}\n")
                    f.write(f"  T-test p-value: {p_val:.4f}\n")

def analyze_results(results_dir: str, output_dir: str) -> None:
    """Analyze evaluation results and generate visualizations/reports."""
    # Load results
    with open(os.path.join(results_dir, "performance_results.json")) as f:
        perf_results = json.load(f)
    with open(os.path.join(results_dir, "controllability_results.json")) as f:
        ctrl_results = json.load(f)
    
    # Performance analysis
    if "base_performance_results.json" in os.listdir(results_dir):
        with open(os.path.join(results_dir, "base_performance_results.json")) as f:
            base_results = json.load(f)
        
        perf_viz = PerformanceVisualizer(base_results, perf_results)
        perf_viz.plot_comparisons(output_dir)
        perf_viz.generate_report(output_dir)
    
    # Controllability analysis
    ctrl_viz = ControllabilityVisualizer(ctrl_results)
    ctrl_viz.plot_distributions(output_dir)
    ctrl_viz.generate_report(output_dir)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze and visualize evaluation results")
    parser.add_argument("--results_dir", required=True, help="Directory containing evaluation results")
    parser.add_argument("--output_dir", default="analysis_results", help="Directory to save analysis results")
    
    args = parser.parse_args()
    analyze_results(args.results_dir, args.output_dir) 