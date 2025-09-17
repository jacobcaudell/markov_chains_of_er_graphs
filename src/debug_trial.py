#!/usr/bin/env python3
"""
Debug script for running single trials with detailed step-by-step information.
"""

import argparse
import json
import logging
from typing import Dict, Any, Optional

import torch

from greedy_reorder_mc import (
    validate_parameters, get_device, build_upper_lut, build_popcnt_table,
    gen_sequence_flip_upper, greedy_reconstruct, full_order_success,
    stepwise_accuracy, kendall_tau, memory_usage_mb
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@torch.no_grad()
def run_debug_trial(n: int = 100, T: int = 10, p0: float = 0.8, q01: float = 0.06, q10: float = 0.14, 
                   eps: float = 0.0, dist_mode: str = "noisy", device: Optional[str] = None, 
                   seed: int = 123, bitpack: bool = True) -> Dict[str, Any]:
    """Run a single trial with detailed debug information."""
    # Validate parameters
    validate_parameters(n, T, p0, q01, q10, [eps], 1)
    
    device = get_device(device)
    torch.manual_seed(seed)
    g = torch.Generator(device=device).manual_seed(seed)

    logger.info(f"[device={device}] n={n}, T={T}, eps={eps}, dist={dist_mode}, bitpack={bitpack}")
    logger.info(f"Memory usage before: {memory_usage_mb():.1f}MB")

    # Generate sequence
    Useq, iu, ju, m = gen_sequence_flip_upper(n, T, p0, q01, q10, device, g)
    logger.info(f"Generated sequence: {Useq.shape}")

    # Precompute structures
    lut = build_upper_lut(n, device)[3]
    POPCNT = build_popcnt_table(device) if bitpack else None

    # Run greedy reconstruction with debug collection
    order, evals, debug_info = greedy_reconstruct(Useq, n, eps, iu, ju, lut, device, 
                                                 bitpack, POPCNT, g, dist_mode, debug=True)
    
    # Add sequence analysis to debug info
    if debug_info is not None:
        debug_info["sequence_analysis"] = {
            "true_order": list(range(T)),
            "reconstructed_order": order,
            "edge_evolution": Useq.sum(dim=1).tolist() if T <= 20 else f"Too large (T={T})",
            "is_perfect": full_order_success(order),
            "stepwise_accuracy": stepwise_accuracy(order),
            "kendall_tau": kendall_tau(order)
        }
        
        # Add memory usage
        debug_info["memory_usage_mb"] = memory_usage_mb()
        
        logger.info(f"Debug info collected: {len(debug_info)} sections")
        logger.info(f"Perfect reconstruction: {debug_info['sequence_analysis']['is_perfect']}")
        logger.info(f"Stepwise accuracy: {debug_info['sequence_analysis']['stepwise_accuracy']:.3f}")
        logger.info(f"Kendall tau: {debug_info['sequence_analysis']['kendall_tau']:.3f}")
    
    return debug_info

def main():
    parser = argparse.ArgumentParser(description="Debug single trial with detailed step-by-step information")
    parser.add_argument("--n", type=int, default=100, help="Number of nodes")
    parser.add_argument("--T", type=int, default=10, help="Sequence length")
    parser.add_argument("--p0", type=float, default=0.8, help="Initial edge probability")
    parser.add_argument("--q01", type=float, default=0.06, help="0->1 transition probability")
    parser.add_argument("--q10", type=float, default=0.14, help="1->0 transition probability")
    parser.add_argument("--eps", type=float, default=0.0, help="Epsilon value")
    parser.add_argument("--dist", type=str, default="noisy", 
                       choices=["noisy", "hamming", "edgecount"], help="Distance mode")
    parser.add_argument("--device", type=str, default=None, help="Device (auto-detect if None)")
    parser.add_argument("--seed", type=int, default=123, help="Random seed")
    parser.add_argument("--no-bitpack", action="store_true", help="Disable bit-packing optimization")
    parser.add_argument("--output", type=str, default="debug_trial.json", help="Output JSON file")
    
    args = parser.parse_args()
    
    try:
        debug_info = run_debug_trial(
            n=args.n, T=args.T, p0=args.p0, q01=args.q01, q10=args.q10,
            eps=args.eps, dist_mode=args.dist, device=args.device,
            seed=args.seed, bitpack=not args.no_bitpack
        )
        
        # Save debug info
        with open(args.output, 'w') as f:
            json.dump(debug_info, f, indent=2)
        
        logger.info(f"Debug information saved to {args.output}")
        logger.info("Use analyze_debug.py to view the results in a readable format")
            
    except Exception as e:
        logger.error(f"Debug trial failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
