#!/usr/bin/env python3
"""
Create T vs p heatmap similar to MATLAB style.
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import math

# Set matplotlib to use Times New Roman and smaller default sizes
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 10
plt.rcParams['figure.figsize'] = (6, 4.5)

def load_results(filename: str):
    """Load results from JSON file."""
    with open(filename, 'r') as f:
        return json.load(f)

def create_t_p_heatmap(results, metric='success_rate'):
    """Create T vs p heatmap."""
    
    # Extract unique T and p values
    T_vals = sorted(list(set([r['T'] for r in results])))
    p_vals = sorted(list(set([r['p0'] for r in results])))
    
    print(f"T values: {T_vals}")
    print(f"p values: {p_vals}")
    
    # Create meshgrid
    T_grid, P_grid = np.meshgrid(T_vals, p_vals)
    
    # Initialize result matrix
    metric_matrix = np.zeros_like(T_grid, dtype=float)
    
    # Fill in the results
    for result in results:
        T_idx = T_vals.index(result['T'])
        p_idx = p_vals.index(result['p0'])
        
        if metric == 'success_rate':
            metric_matrix[p_idx, T_idx] = result['full_order_success_rate']['mean']
        elif metric == 'stepwise_accuracy':
            metric_matrix[p_idx, T_idx] = result['stepwise_accuracy']['mean']
        elif metric == 'kendall_tau':
            metric_matrix[p_idx, T_idx] = result['kendall_tau']['mean']
        elif metric == 'first_error_index':
            metric_matrix[p_idx, T_idx] = result['first_error_index']['mean']
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create heatmap with contourf
    levels = np.linspace(0, 1, 21) if metric in ['success_rate', 'stepwise_accuracy', 'kendall_tau'] else None
    im = ax.contourf(T_grid, P_grid, metric_matrix, levels=levels, cmap='jet')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(metric.replace('_', ' ').title())
    
    # Labels and title
    ax.set_xlabel('Sequence Length (T)')
    ax.set_ylabel('Initial Edge Probability (p0)')
    ax.set_title(f'{metric.replace("_", " ").title()} vs T and p0 (n=200, q01=0.06, q10=0.14)')
    
    # Grid
    ax.grid(True, alpha=0.3)
    
    # Theoretical boundary plotting removed
    
    return fig

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Create T vs p heatmap")
    parser.add_argument("input_file", default="t_p_heatmap_data.json", 
                       help="JSON file with results")
    parser.add_argument("--metric", default="success_rate", 
                       choices=["success_rate", "stepwise_accuracy", "kendall_tau", "first_error_index"],
                       help="Metric to plot")
    parser.add_argument("--output", help="Output file (default: display)")
    parser.add_argument("--dpi", type=int, default=300, help="DPI for saved figures")
    
    args = parser.parse_args()
    
    # Load results
    results = load_results(args.input_file)
    print(f"Loaded {len(results)} results")
    
    # Create heatmap
    fig = create_t_p_heatmap(results, args.metric)
    
    if args.output:
        fig.savefig(f"{args.output}_{args.metric}.png", dpi=args.dpi, bbox_inches='tight')
        print(f"Saved to {args.output}_{args.metric}.png")
    else:
        plt.show()

if __name__ == "__main__":
    main()
