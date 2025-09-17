#!/usr/bin/env python3
"""
Utility to recompute confidence intervals from saved JSON results.
Allows post-processing of trial data with different confidence levels.
"""

import json
import argparse
import numpy as np
from typing import List, Tuple, Dict, Any

def bootstrap_confidence_interval(data: List[float], n_bootstrap: int = 1000, 
                                ci_level: float = 0.95) -> Tuple[float, float]:
    """Calculate bootstrap confidence interval for a list of values."""
    if len(data) < 2:
        return float(data[0]) if data else 0.0, float(data[0]) if data else 0.0
    
    data_array = np.array(data)
    bootstrap_means = []
    
    for _ in range(n_bootstrap):
        # Resample with replacement
        resample = np.random.choice(data_array, size=len(data_array), replace=True)
        bootstrap_means.append(np.mean(resample))
    
    # Calculate confidence interval bounds
    alpha = 1 - ci_level
    lower = np.percentile(bootstrap_means, 100 * alpha / 2)
    upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))
    
    return float(lower), float(upper)

def recompute_confidence_intervals(results: List[Dict[str, Any]], 
                                 ci_level: float = 0.95) -> List[Dict[str, Any]]:
    """Recompute confidence intervals for all results with new confidence level."""
    updated_results = []
    
    for result in results:
        if "raw_trial_data" not in result:
            print(f"Warning: No raw trial data found for {result.get('dist_mode', 'unknown')} mode")
            updated_results.append(result)
            continue
            
        raw_data = result["raw_trial_data"]
        
        # Recompute confidence intervals
        success_ci = bootstrap_confidence_interval(raw_data["success_trials"], ci_level=ci_level)
        stepwise_ci = bootstrap_confidence_interval(raw_data["stepwise_trials"], ci_level=ci_level)
        kendall_ci = bootstrap_confidence_interval(raw_data["kendall_trials"], ci_level=ci_level)
        evals_ci = bootstrap_confidence_interval(raw_data["eval_trials"], ci_level=ci_level)
        
        # Update the result with new confidence intervals
        updated_result = result.copy()
        updated_result["full_order_success_rate"].update({
            "ci_lower": success_ci[0],
            "ci_upper": success_ci[1],
            "ci_level": ci_level
        })
        updated_result["stepwise_accuracy"].update({
            "ci_lower": stepwise_ci[0],
            "ci_upper": stepwise_ci[1],
            "ci_level": ci_level
        })
        updated_result["kendall_tau"].update({
            "ci_lower": kendall_ci[0],
            "ci_upper": kendall_ci[1],
            "ci_level": ci_level
        })
        updated_result["candidate_evals"].update({
            "ci_lower": evals_ci[0],
            "ci_upper": evals_ci[1],
            "ci_level": ci_level
        })
        
        updated_results.append(updated_result)
    
    return updated_results

def print_results_summary(results: List[Dict[str, Any]]):
    """Print a summary of the results with confidence intervals."""
    for result in results:
        dist_mode = result["dist_mode"]
        eps = result["eps"]
        
        success = result["full_order_success_rate"]
        stepwise = result["stepwise_accuracy"]
        kendall = result["kendall_tau"]
        evals = result["candidate_evals"]
        
        print(f"Dist={dist_mode}, Eps={eps:.1f}: "
              f"Success={success['mean']:.3f} [{success['ci_lower']:.3f}, {success['ci_upper']:.3f}], "
              f"StepAcc={stepwise['mean']:.3f} [{stepwise['ci_lower']:.3f}, {stepwise['ci_upper']:.3f}], "
              f"Kendall={kendall['mean']:.3f} [{kendall['ci_lower']:.3f}, {kendall['ci_upper']:.3f}], "
              f"Evals={evals['mean']:.1f} [{evals['ci_lower']:.1f}, {evals['ci_upper']:.1f}]")

def main():
    parser = argparse.ArgumentParser(description="Recompute confidence intervals from saved JSON results")
    parser.add_argument("input_file", help="Input JSON file with raw trial data")
    parser.add_argument("--ci-level", type=float, default=0.95, 
                       help="Confidence level for recomputed intervals (0.0-1.0)")
    parser.add_argument("--output", type=str, help="Output JSON file (default: print to stdout)")
    parser.add_argument("--summary", action="store_true", help="Print summary to console")
    
    args = parser.parse_args()
    
    # Load results
    with open(args.input_file, 'r') as f:
        results = json.load(f)
    
    # Recompute confidence intervals
    updated_results = recompute_confidence_intervals(results, args.ci_level)
    
    # Print summary if requested
    if args.summary:
        print(f"Results with {args.ci_level*100:.0f}% confidence intervals:")
        print_results_summary(updated_results)
    
    # Save or print results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(updated_results, f, indent=2)
        print(f"Updated results saved to {args.output}")
    else:
        print(json.dumps(updated_results, indent=2))

if __name__ == "__main__":
    main()
