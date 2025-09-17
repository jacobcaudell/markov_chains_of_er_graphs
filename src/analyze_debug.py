#!/usr/bin/env python3
"""
Read and display results from various JSON output files.
Supports simulation results, comparison results, and debug output.
"""

import json
import sys
import argparse
from typing import Dict, Any, List

def read_simulation_results(data: List[Dict[str, Any]], filename: str):
    """Read and display simulation results."""
    print(f"=== SIMULATION RESULTS: {filename} ===")
    print(f"Number of experiments: {len(data)}")
    print()
    
    for i, result in enumerate(data):
        print(f"Experiment {i+1}:")
        print(f"  Distance mode: {result['dist_mode']}")
        print(f"  Epsilon: {result['eps']}")
        print(f"  Success rate: {result['full_order_success_rate']:.3f}")
        print(f"  Stepwise accuracy: {result['mean_stepwise_accuracy']:.3f}")
        print(f"  Kendall tau: {result['mean_kendall_tau']:.3f}")
        print(f"  Avg evaluations: {result['avg_candidate_evals']:.1f}")
        print(f"  Time per trial: {result['sec_per_trial']:.3f}s")
        print(f"  Memory usage: {result['memory_usage_mb']:.1f}MB")
        print()

def read_debug_results(data: Dict[str, Any], filename: str):
    """Read and display debug results."""
    print(f"=== DEBUG ANALYSIS: {filename} ===")
    
    # Check if it's a list (old format) or dict (new format)
    if isinstance(data, list):
        debug_info = data[0]
    else:
        debug_info = data
    
    # Basic info
    if 'sequence_stats' in debug_info:
        seq_stats = debug_info['sequence_stats']
        print(f"Mode: {seq_stats['dist_mode']}")
        print(f"Epsilon: {seq_stats['eps']}")
        print(f"Nodes: {seq_stats['n']}, Time steps: {seq_stats['T']}")
        print(f"Edge density: {seq_stats['edge_density']:.3f}")
        print()
    
    # Results
    if 'final_summary' in debug_info:
        summary = debug_info['final_summary']
        print("=== RECONSTRUCTION RESULTS ===")
        print(f"Perfect reconstruction: {summary['is_perfect']}")
        print(f"Stepwise accuracy: {summary['stepwise_accuracy']:.3f}")
        print(f"Kendall tau: {summary['kendall_tau']:.3f}")
        print(f"Correct steps: {summary['correct_steps']}/{summary['total_steps']}")
        print()
    
    # Step-by-step breakdown
    if 'steps' in debug_info and debug_info['steps']:
        print("=== STEP-BY-STEP BREAKDOWN ===")
        for i, step in enumerate(debug_info['steps']):
            status = "✓" if step['is_correct'] else "✗"
            print(f"Step {i}: Current={step['current_node']}, Chosen={step['chosen_node']} {status}")
            print(f"  Distances: min={step['min_distance']:.1f}, max={step['max_distance']:.1f}, mean={step['mean_distance']:.1f}")
            
            # Show mode-specific info
            if 'permutation_info' in step:
                perm_info = step['permutation_info']
                print(f"  Permutations: {perm_info['num_identity_perms']} identity, {perm_info['num_random_perms']} random")
            elif 'hamming_info' in step:
                hamming_info = step['hamming_info']
                print(f"  Edge counts: current={hamming_info['current_edges']}, chosen={hamming_info['chosen_edges']}")
            print()
    
    # Sequence analysis
    if 'sequence_analysis' in debug_info:
        seq_analysis = debug_info['sequence_analysis']
        print("=== SEQUENCE ANALYSIS ===")
        print(f"True order: {seq_analysis['true_order']}")
        print(f"Reconstructed: {seq_analysis['reconstructed_order']}")
        if 'edge_evolution' in seq_analysis:
            print(f"Edge evolution: {seq_analysis['edge_evolution']}")
        print()

def read_results_file(filename: str):
    """Read and display results from a JSON file."""
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
        
        # Determine file type based on structure
        if isinstance(data, list):
            if len(data) > 0 and 'dist_mode' in data[0] and 'full_order_success_rate' in data[0]:
                # Simulation results
                read_simulation_results(data, filename)
            elif len(data) > 0 and 'steps' in data[0]:
                # Debug results (old format)
                read_debug_results(data, filename)
            else:
                print(f"Unknown list format in {filename}")
        elif isinstance(data, dict):
            if 'steps' in data or 'sequence_stats' in data:
                # Debug results (new format)
                read_debug_results(data, filename)
            else:
                print(f"Unknown dict format in {filename}")
        else:
            print(f"Unknown data format in {filename}")
            
    except Exception as e:
        print(f"Error reading {filename}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Read and display results from JSON files")
    parser.add_argument("files", nargs="+", help="JSON files to read")
    parser.add_argument("--summary", action="store_true", help="Show only summary information")
    args = parser.parse_args()
    
    for filename in args.files:
        read_results_file(filename)
        print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    main()
