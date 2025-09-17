#!/usr/bin/env python3
"""
Generate heatmap data for T vs p with fixed n=200, q01=0.06, q10=0.14
"""

import subprocess
import json
import numpy as np
import math
from typing import List, Dict, Any

def run_simulation(T: int, p0: float, trials: int = 20) -> Dict[str, Any]:
    """Run a single simulation with fixed parameters."""
    
    cmd = [
        "python", "src/greedy_reorder_mc.py",
        "--n", "200",
        "--T", str(T), 
        "--trials", str(trials),
        "--dist", "edgecount",  # Use edge count distance
        "--p0", str(p0),
        "--q01", "0.06",
        "--q10", "0.14",
        "--ci-level", "0.95",
        "--output", f"temp_result_{T}_{p0:.3f}.json"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error running simulation T={T}, p0={p0}: {result.stderr}")
        return None
    
    # Read the JSON file that was created
    import os
    temp_file = f"temp_result_{T}_{p0:.3f}.json"
    try:
        if os.path.exists(temp_file):
            with open(temp_file, 'r') as f:
                data = json.load(f)[0]
            data['T'] = T
            data['p0'] = p0
            # Clean up temp file
            os.remove(temp_file)
            return data
        else:
            print(f"Output file {temp_file} not found")
            return None
    except (json.JSONDecodeError, IndexError) as e:
        print(f"Error parsing JSON: {e}")
        return None

def generate_t_p_heatmap_data():
    """Generate data for T vs p heatmap."""
    
    n = 200
    T_min = 3
    T_max = 30
    
    # Create parameter ranges
    T_vals = list(range(T_min, T_max + 1))  # 3 to 30
    p_vals = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]  # 7 specific values
    
    print(f"T range: {T_min} to {T_max} ({len(T_vals)} values)")
    print(f"p range: {p_vals[0]} to {p_vals[-1]} ({len(p_vals)} values)")
    print(f"Total simulations: {len(T_vals) * len(p_vals)}")
    
    results = []
    total_sims = len(T_vals) * len(p_vals)
    current_sim = 0
    
    for T in T_vals:
        for p0 in p_vals:
            current_sim += 1
            print(f"Running simulation {current_sim}/{total_sims}: T={T}, p0={p0:.3f}")
            
            result = run_simulation(T, p0, trials=200)
            if result:
                results.append(result)
    
    return results

def main():
    print("Generating T vs p heatmap data...")
    print("Parameters: n=200, q01=0.06, q10=0.14")
    print("T range: 3 to 30")
    print("p range: 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0")
    print()
    
    results = generate_t_p_heatmap_data()
    
    # Save results
    output_file = "t_p_heatmap_data.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    print(f"Total simulations completed: {len(results)}")
    
    # Print summary statistics
    if results:
        success_rates = [r['full_order_success_rate']['mean'] for r in results]
        print(f"Success rate range: {min(success_rates):.3f} to {max(success_rates):.3f}")
        
        stepwise_accs = [r['stepwise_accuracy']['mean'] for r in results]
        print(f"Stepwise accuracy range: {min(stepwise_accs):.3f} to {max(stepwise_accs):.3f}")

if __name__ == "__main__":
    main()
