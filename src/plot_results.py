#!/usr/bin/env python3
"""
Plotting utilities for Monte Carlo simulation results.
Creates heatmaps and visualizations similar to MATLAB plotting capabilities.
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import argparse
from typing import Dict, List, Any, Tuple
import seaborn as sns

# Set matplotlib to use Times New Roman and smaller default sizes
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 10
plt.rcParams['figure.figsize'] = (6, 4.5)  # Smaller default size

def load_results(filename: str) -> List[Dict[str, Any]]:
    """Load results from JSON file."""
    with open(filename, 'r') as f:
        return json.load(f)

def create_parameter_heatmap(results: List[Dict[str, Any]], 
                           param1: str, param2: str, 
                           metric: str = 'success_rate',
                           title: str = None) -> plt.Figure:
    """Create a heatmap of results across two parameters."""
    
    # Extract unique parameter values
    param1_vals = sorted(list(set([r[param1] for r in results])))
    param2_vals = sorted(list(set([r[param2] for r in results])))
    
    # Create meshgrid
    P1, P2 = np.meshgrid(param1_vals, param2_vals)
    
    # Initialize result matrix
    metric_matrix = np.zeros_like(P1, dtype=float)
    
    # Fill in the results
    for result in results:
        i = param2_vals.index(result[param2])
        j = param1_vals.index(result[param1])
        
        if metric == 'success_rate':
            metric_matrix[i, j] = result['full_order_success_rate']['mean']
        elif metric == 'stepwise_accuracy':
            metric_matrix[i, j] = result['stepwise_accuracy']['mean']
        elif metric == 'kendall_tau':
            metric_matrix[i, j] = result['kendall_tau']['mean']
        elif metric == 'first_error_index':
            metric_matrix[i, j] = result['first_error_index']['mean']
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create heatmap
    im = ax.contourf(P1, P2, metric_matrix, levels=20, cmap='jet')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(metric.replace('_', ' ').title())
    
    # Labels and title
    ax.set_xlabel(param1.replace('_', ' ').title())
    ax.set_ylabel(param2.replace('_', ' ').title())
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f'{metric.replace("_", " ").title()} Heatmap')
    
    # Grid
    ax.grid(True, alpha=0.3)
    
    return fig

def create_eps_vs_performance_heatmap(results: List[Dict[str, Any]], 
                                    n_vals: List[int], 
                                    eps_vals: List[float],
                                    metric: str = 'success_rate') -> plt.Figure:
    """Create heatmap of epsilon vs n for a specific metric."""
    
    # Create meshgrid
    N, Eps = np.meshgrid(n_vals, eps_vals)
    metric_matrix = np.zeros_like(N, dtype=float)
    
    # Fill in results
    for result in results:
        if result['dist_mode'] != 'noisy':
            continue
            
        n_idx = n_vals.index(result['n'])
        eps_idx = eps_vals.index(result['eps'])
        
        if metric == 'success_rate':
            metric_matrix[eps_idx, n_idx] = result['full_order_success_rate']['mean']
        elif metric == 'stepwise_accuracy':
            metric_matrix[eps_idx, n_idx] = result['stepwise_accuracy']['mean']
        elif metric == 'kendall_tau':
            metric_matrix[eps_idx, n_idx] = result['kendall_tau']['mean']
        elif metric == 'first_error_index':
            metric_matrix[eps_idx, n_idx] = result['first_error_index']['mean']
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Heatmap
    im = ax.contourf(N, Eps, metric_matrix, levels=20, cmap='jet')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(metric.replace('_', ' ').title())
    
    # Labels
    ax.set_xlabel('Graph Size (n)')
    ax.set_ylabel('Epsilon (ε)')
    ax.set_title(f'{metric.replace("_", " ").title()} vs Graph Size and Epsilon')
    
    # Grid
    ax.grid(True, alpha=0.3)
    
    return fig

def create_distance_mode_comparison(results: List[Dict[str, Any]], 
                                  n_vals: List[int],
                                  metric: str = 'success_rate') -> plt.Figure:
    """Compare different distance modes across graph sizes."""
    
    modes = ['noisy', 'hamming', 'edgecount']
    metric_matrix = np.zeros((len(modes), len(n_vals)))
    
    # Fill in results
    for result in results:
        if result['n'] not in n_vals:
            continue
            
        mode_idx = modes.index(result['dist_mode'])
        n_idx = n_vals.index(result['n'])
        
        if metric == 'success_rate':
            metric_matrix[mode_idx, n_idx] = result['full_order_success_rate']['mean']
        elif metric == 'stepwise_accuracy':
            metric_matrix[mode_idx, n_idx] = result['stepwise_accuracy']['mean']
        elif metric == 'kendall_tau':
            metric_matrix[mode_idx, n_idx] = result['kendall_tau']['mean']
        elif metric == 'first_error_index':
            metric_matrix[mode_idx, n_idx] = result['first_error_index']['mean']
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Heatmap
    im = ax.imshow(metric_matrix, cmap='jet', aspect='auto')
    
    # Set ticks
    ax.set_xticks(range(len(n_vals)))
    ax.set_xticklabels(n_vals)
    ax.set_yticks(range(len(modes)))
    ax.set_yticklabels(modes)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(metric.replace('_', ' ').title())
    
    # Labels
    ax.set_xlabel('Graph Size (n)')
    ax.set_ylabel('Distance Mode')
    ax.set_title(f'{metric.replace("_", " ").title()} by Distance Mode and Graph Size')
    
    return fig

def create_first_error_analysis(results: List[Dict[str, Any]], 
                              n_vals: List[int]) -> plt.Figure:
    """Create analysis of first error index vs graph size."""
    
    # Extract first error data
    first_errors = {}
    for result in results:
        n = result['n']
        if n not in first_errors:
            first_errors[n] = []
        first_errors[n].append(result['first_error_index']['mean'])
    
    # Calculate means and stds
    n_sorted = sorted(first_errors.keys())
    means = [np.mean(first_errors[n]) for n in n_sorted]
    stds = [np.std(first_errors[n]) for n in n_sorted]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Error bars
    ax.errorbar(n_sorted, means, yerr=stds, marker='o', capsize=5, capthick=2)
    
    # Add theoretical curves
    n_array = np.array(n_sorted)
    ax.plot(n_sorted, 3 * np.log(n_array), 'r--', label='3×ln(n)', linewidth=2)
    ax.plot(n_sorted, 4 * np.log(n_array), 'g--', label='4×ln(n)', linewidth=2)
    ax.plot(n_sorted, 5 * np.log(n_array), 'b--', label='5×ln(n)', linewidth=2)
    
    # Labels
    ax.set_xlabel('Graph Size (n)')
    ax.set_ylabel('First Error Index')
    ax.set_title('First Error Index vs Graph Size')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig

def main():
    parser = argparse.ArgumentParser(description="Create heatmaps from Monte Carlo results")
    parser.add_argument("input_file", help="JSON file with results")
    parser.add_argument("--metric", default="success_rate", 
                       choices=["success_rate", "stepwise_accuracy", "kendall_tau", "first_error_index"],
                       help="Metric to plot")
    parser.add_argument("--output", help="Output file (default: display)")
    parser.add_argument("--dpi", type=int, default=300, help="DPI for saved figures")
    
    args = parser.parse_args()
    
    # Load results
    results = load_results(args.input_file)
    
    # Create different types of plots
    print("Creating heatmaps...")
    
    # 1. Epsilon vs performance (if noisy mode results exist)
    noisy_results = [r for r in results if r['dist_mode'] == 'noisy']
    if noisy_results:
        n_vals = sorted(list(set([r['n'] for r in noisy_results])))
        eps_vals = sorted(list(set([r['eps'] for r in noisy_results])))
        
        if len(n_vals) > 1 and len(eps_vals) > 1:
            fig1 = create_eps_vs_performance_heatmap(noisy_results, n_vals, eps_vals, args.metric)
            if args.output:
                fig1.savefig(f"{args.output}_eps_vs_n_{args.metric}.png", dpi=args.dpi, bbox_inches='tight')
            plt.show()
    
    # 2. Distance mode comparison
    n_vals = sorted(list(set([r['n'] for r in results])))
    if len(n_vals) > 1:
        fig2 = create_distance_mode_comparison(results, n_vals, args.metric)
        if args.output:
            fig2.savefig(f"{args.output}_distance_modes_{args.metric}.png", dpi=args.dpi, bbox_inches='tight')
        plt.show()
    
    # 3. First error analysis
    if args.metric == "first_error_index":
        fig3 = create_first_error_analysis(results, n_vals)
        if args.output:
            fig3.savefig(f"{args.output}_first_error_analysis.png", dpi=args.dpi, bbox_inches='tight')
        plt.show()
    
    print("Heatmaps created!")

if __name__ == "__main__":
    main()
