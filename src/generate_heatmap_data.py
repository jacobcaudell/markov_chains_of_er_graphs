#!/usr/bin/env python3
"""
Generate data for heatmap analysis by running simulations across parameter ranges.
"""

import subprocess
import json
import argparse
import numpy as np
from typing import List, Dict, Any

def run_simulation(n: int, T: int, trials: int, dist_mode: str, 
                  eps: float = None, p0: float = 0.8, q01: float = 0.06, q10: float = 0.14) -> Dict[str, Any]:
    """Run a single simulation and return results."""
    
    cmd = [
        "python", "src/greedy_reorder_mc.py",
        "--n", str(n),
        "--T", str(T), 
        "--trials", str(trials),
        "--dist", dist_mode,
        "--p0", str(p0),
        "--q01", str(q01),
        "--q10", str(q10),
        "--ci-level", "0.95",
        "--quiet"
    ]
    
    if eps is not None and dist_mode == "noisy":
        cmd.extend(["--eps", str(eps)])
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error running simulation: {result.stderr}")
        return None
    
    # Parse the JSON output from stdout
    try:
        # The output should be JSON, but we need to extract it
        lines = result.stdout.strip().split('\n')
        json_line = None
        for line in lines:
            if line.startswith('[') or line.startswith('{'):
                json_line = line
                break
        
        if json_line:
            return json.loads(json_line)[0]  # Get first (and only) result
        else:
            print("No JSON output found")
            return None
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        return None

def generate_eps_vs_n_data(n_vals: List[int], eps_vals: List[float], 
                          T: int = 20, trials: int = 10) -> List[Dict[str, Any]]:
    """Generate data for epsilon vs n heatmap."""
    
    results = []
    total_sims = len(n_vals) * len(eps_vals)
    current_sim = 0
    
    for n in n_vals:
        for eps in eps_vals:
            current_sim += 1
            print(f"Running simulation {current_sim}/{total_sims}: n={n}, eps={eps:.2f}")
            
            result = run_simulation(n, T, trials, "noisy", eps)
            if result:
                result['n'] = n
                result['eps'] = eps
                results.append(result)
    
    return results

def generate_distance_mode_data(n_vals: List[int], T: int = 20, trials: int = 10) -> List[Dict[str, Any]]:
    """Generate data for distance mode comparison."""
    
    results = []
    modes = ["noisy", "hamming", "edgecount"]
    total_sims = len(n_vals) * len(modes)
    current_sim = 0
    
    for n in n_vals:
        for mode in modes:
            current_sim += 1
            print(f"Running simulation {current_sim}/{total_sims}: n={n}, mode={mode}")
            
            eps = 0.2 if mode == "noisy" else None
            result = run_simulation(n, T, trials, mode, eps)
            if result:
                result['n'] = n
                result['dist_mode'] = mode
                results.append(result)
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Generate data for heatmap analysis")
    parser.add_argument("--type", choices=["eps_vs_n", "distance_modes"], 
                       default="eps_vs_n", help="Type of analysis to generate")
    parser.add_argument("--n-range", nargs=2, type=int, default=[100, 1000], 
                       help="Range of n values (min max)")
    parser.add_argument("--eps-range", nargs=2, type=float, default=[0.0, 1.0],
                       help="Range of epsilon values (min max)")
    parser.add_argument("--n-points", type=int, default=5, help="Number of n points")
    parser.add_argument("--eps-points", type=int, default=6, help="Number of eps points")
    parser.add_argument("--T", type=int, default=20, help="Sequence length")
    parser.add_argument("--trials", type=int, default=10, help="Number of trials per simulation")
    parser.add_argument("--output", default="heatmap_data.json", help="Output file")
    
    args = parser.parse_args()
    
    if args.type == "eps_vs_n":
        # Generate epsilon vs n data
        n_vals = np.linspace(args.n_range[0], args.n_range[1], args.n_points, dtype=int)
        eps_vals = np.linspace(args.eps_range[0], args.eps_range[1], args.eps_points)
        
        print(f"Generating epsilon vs n data...")
        print(f"n values: {n_vals}")
        print(f"eps values: {eps_vals}")
        
        results = generate_eps_vs_n_data(n_vals, eps_vals, args.T, args.trials)
        
    elif args.type == "distance_modes":
        # Generate distance mode comparison data
        n_vals = np.linspace(args.n_range[0], args.n_range[1], args.n_points, dtype=int)
        
        print(f"Generating distance mode comparison data...")
        print(f"n values: {n_vals}")
        
        results = generate_distance_mode_data(n_vals, args.T, args.trials)
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {args.output}")
    print(f"Total simulations: {len(results)}")

if __name__ == "__main__":
    main()
