#!/usr/bin/env python3
"""
Comparison script for running all distance modes on the same seeded trials.
"""

import argparse
import json
import logging
import time
from typing import List, Dict, Any, Optional

import torch

from greedy_reorder_mc import (
    validate_parameters, get_device, build_upper_lut, build_popcnt_table,
    gen_sequence_flip_upper, greedy_reconstruct, full_order_success,
    stepwise_accuracy, kendall_tau, memory_usage_mb, run_trials_for_eps
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@torch.no_grad()
def run_comparison(n: int = 100, T: int = 10, p0: float = 0.8, q01: float = 0.06, q10: float = 0.14, 
                  eps_list: List[float] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0), trials: int = 50, 
                  device: Optional[str] = None, seed: int = 123, bitpack: bool = True, 
                  compile_greedy: bool = False, verbose: bool = True, 
                  save_state: bool = False) -> List[Dict[str, Any]]:
    """Run comparison across all distance modes using the same seeded trials."""
    # Validate parameters
    validate_parameters(n, T, p0, q01, q10, eps_list, trials)
    
    device = get_device(device)
    torch.manual_seed(seed)
    g = torch.Generator(device=device).manual_seed(seed)

    if verbose:
        logger.info(f"[device={device}] n={n}, T={T}, trials={trials}, bitpack={bitpack}, compile={compile_greedy}")
        logger.info(f"Memory usage before: {memory_usage_mb():.1f}MB")
        logger.info("Running comparison across all distance modes with same seeded trials")

    # Precompute structures shared across trials/eps
    iu, ju, m, lut = build_upper_lut(n, device)
    POPCNT = build_popcnt_table(device) if bitpack else None

    # Generate all sequences upfront with the same seed
    sequences = []
    for trial in range(trials):
        # Reset generator to same state for each trial
        g_trial = torch.Generator(device=device).manual_seed(seed + trial)
        Useq, _, _, _ = gen_sequence_flip_upper(n, T, p0, q01, q10, device, g_trial)
        sequences.append(Useq)

    def _one_trial_comparison(Useq: torch.Tensor, dist_mode: str, eps: float) -> tuple:
        """Run one trial with a specific distance mode."""
        order, evals = greedy_reconstruct(Useq, n, eps, iu, ju, lut, device, bitpack, POPCNT, g, dist_mode)
        return order, evals

    compiled = None
    if compile_greedy and hasattr(torch, "compile"):
        compiled = torch.compile(_one_trial_comparison)
        if verbose:
            logger.info("Compiled greedy path enabled.")

    results = []
    distance_modes = ["noisy", "hamming", "edgecount"]
    
    for dist_mode in distance_modes:
        if verbose:
            logger.info(f"Running {dist_mode} mode...")
        
        # Choose eps values to iterate (single 0.0 for non-noisy modes)
        eps_iter = eps_list if dist_mode == "noisy" else [0.0]
        
        for eps in eps_iter:
            def trial_func():
                Useq = sequences[trial]  # Use the same sequence for all distance modes and eps values
                if compiled is not None:
                    return compiled(Useq, dist_mode, eps)
                else:
                    return _one_trial_comparison(Useq, dist_mode, eps)
            
            stats = run_trials_for_eps(trials, trial_func, eps, dist_mode, verbose)
            results.append(stats)
            
            if verbose:
                logger.info(f"Dist={dist_mode}, Eps={eps:.1f}: Success={stats['full_order_success_rate']:.3f}, "
                           f"StepAcc={stats['mean_stepwise_accuracy']:.3f}, "
                           f"Kendall={stats['mean_kendall_tau']:.3f}, "
                           f"Time={stats['sec_per_trial']:.3f}s/trial")
    
    if save_state:
        state_file = f"random_state_seed_{seed}.pt"
        torch.save(g.get_state(), state_file)
        logger.info(f"Random state saved to {state_file}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Compare all distance modes on same seeded trials")
    parser.add_argument("--n", type=int, default=100, help="Number of nodes")
    parser.add_argument("--T", type=int, default=10, help="Sequence length")
    parser.add_argument("--p0", type=float, default=0.8, help="Initial edge probability")
    parser.add_argument("--q01", type=float, default=0.06, help="0->1 transition probability")
    parser.add_argument("--q10", type=float, default=0.14, help="1->0 transition probability")
    parser.add_argument("--eps", nargs="+", type=float, default=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0], help="Epsilon values")
    parser.add_argument("--trials", type=int, default=50, help="Number of trials per epsilon")
    parser.add_argument("--device", type=str, default=None, help="Device (auto-detect if None)")
    parser.add_argument("--seed", type=int, default=123, help="Random seed")
    parser.add_argument("--no-bitpack", action="store_true", help="Disable bit-packing optimization")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile for greedy reconstruction")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output")
    parser.add_argument("--save-state", action="store_true", help="Save random state for reproducibility")
    parser.add_argument("--output", type=str, default="comparison_results.json", help="Output JSON file")
    
    args = parser.parse_args()
    
    try:
        results = run_comparison(
            n=args.n, T=args.T, p0=args.p0, q01=args.q01, q10=args.q10,
            eps_list=args.eps, trials=args.trials, device=args.device, seed=args.seed,
            bitpack=not args.no_bitpack, compile_greedy=args.compile,
            verbose=not args.quiet, save_state=args.save_state
        )
        
        # Save results
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        
        if not args.quiet:
            logger.info(f"Results saved to {args.output}")
            logger.info(f"Total results: {len(results)}")
            
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
