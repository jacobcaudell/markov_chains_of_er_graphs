#!/usr/bin/env python3
# greedy_reorder_mc.py
# Monte-Carlo greedy reordering under noisy pairwise permutations
# Works on Apple MPS (Metal) and CUDA

import argparse
import logging
import time
import warnings
from typing import List, Tuple, Optional, Dict, Any
import torch
import torch.nn.functional as F

# ----------------------------- Configuration -----------------------------

# Constants
DEFAULT_MEMORY_WARNING_GB = 8.0

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ----------------------------- Utilities -----------------------------

def run_trials_for_eps(trials: int, trial_func, eps: float, dist_mode: str, verbose: bool) -> Dict[str, Any]:
    """Run trials for a single epsilon value and collect statistics."""
    ok = step_acc_sum = kendall_sum = total_evals = 0
    t0 = time.time()
    
    for trial in range(trials):
        order, evals = trial_func()
        ok += int(full_order_success(order))
        step_acc_sum += stepwise_accuracy(order)
        kendall_sum += kendall_tau(order)
        total_evals += evals
        
        if verbose and (trial + 1) % max(1, trials // 10) == 0:
            logger.info(f"Dist={dist_mode}, Eps={eps:.1f}, Trial {trial+1}/{trials}")
    
    dt = time.time() - t0
    return {
        "dist_mode": dist_mode,
        "eps": float(eps),
        "full_order_success_rate": ok / trials,
        "mean_stepwise_accuracy": step_acc_sum / trials,
        "mean_kendall_tau": kendall_sum / trials,
        "avg_candidate_evals": total_evals / trials,
        "sec": dt,
        "sec_per_trial": dt / trials,
        "memory_usage_mb": memory_usage_mb()
    }

def validate_parameters(n: int, T: int, p0: float, q01: float, q10: float, 
                       eps_list: List[float], trials: int) -> None:
    """Validate input parameters."""
    if n <= 0 or T <= 0 or trials <= 0:
        raise ValueError(f"n, T, trials must be positive: {n}, {T}, {trials}")
    if not all(0 <= p <= 1 for p in [p0, q01, q10]):
        raise ValueError(f"Probabilities must be in [0,1]: p0={p0}, q01={q01}, q10={q10}")
    if not all(0 <= eps <= 1 for eps in eps_list):
        raise ValueError(f"All eps values must be in [0,1]: {eps_list}")
    
    # Memory warning
    memory_gb = (n * (n - 1) // 2 * T) / (1024**3)
    if memory_gb > DEFAULT_MEMORY_WARNING_GB:
        warnings.warn(f"Estimated memory: {memory_gb:.2f}GB. Consider reducing n or T.")

def get_device(prefer: Optional[str] = None) -> str:
    """Get the best available device."""
    if prefer:
        if prefer not in ["cuda", "mps", "cpu"]:
            raise ValueError(f"Device must be cuda/mps/cpu, got {prefer}")
        if (prefer == "cuda" and torch.cuda.is_available()) or \
           (prefer == "mps" and torch.backends.mps.is_available()) or \
           prefer == "cpu":
            return prefer
        logger.warning(f"{prefer} not available, falling back")
    
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def upper_tri_indices(n: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Get upper triangular indices for n x n matrix."""
    iu, ju = torch.triu_indices(n, n, offset=1, device=device)
    return iu, ju, iu.numel()  # m = #upper-tri edges

def build_upper_lut(n: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, int, torch.Tensor]:
    """Build lookup table for upper triangular edge indexing."""
    iu, ju, m = upper_tri_indices(n, device)
    lut = torch.full((n, n), -1, dtype=torch.long, device=device)
    idx = torch.arange(m, device=device)
    lut[iu, ju] = idx
    lut[ju, iu] = idx  # symmetric lookup; avoids branching on (min,max)
    return iu, ju, m, lut

# --------------------- Bit-packing (optional fast path) ---------------------

def build_popcnt_table(device: torch.device) -> torch.Tensor:
    """Build population count lookup table for bit-packing optimization."""
    return torch.tensor([bin(x).count("1") for x in range(256)],
                        dtype=torch.int16, device=device)

def pack_bits_bool_to_u8(U_bool: torch.Tensor) -> torch.Tensor:
    """Pack boolean tensor to uint8 for memory efficiency."""
    m = U_bool.shape[-1]
    pad = (-m) % 8
    if pad:
        U_bool = F.pad(U_bool, (0, pad))
    bits = U_bool.view(*U_bool.shape[:-1], -1, 8).to(torch.uint8)
    weights = (1 << torch.arange(8, device=U_bool.device, dtype=torch.uint8))
    return (bits * weights).sum(dim=-1)  # (..., m/8)

def hamming_u8(A_u8: torch.Tensor, B_u8: torch.Tensor, POPCNT: torch.Tensor) -> torch.Tensor:
    """Compute Hamming distance using bit-packed representation."""
    X = torch.bitwise_xor(A_u8, B_u8)
    return POPCNT[X].sum(dim=1).to(torch.int32)  # batch popcount

# --------------------------- Sequence generator ---------------------------

@torch.no_grad()
def gen_sequence_flip_upper(n: int, T: int, p0: float, q01: float, q10: float, 
                           device: torch.device, generator: Optional[torch.Generator] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """
    Generate XOR-evolved sequence on the upper-tri edge vector (bool).
    
    This implements a Markov chain where each edge (i,j) flips independently:
    - 0->1 with probability q01
    - 1->0 with probability q10
    
    Args:
        n: Number of nodes
        T: Sequence length
        p0: Initial probability of edge existence
        q01: Transition probability 0->1
        q10: Transition probability 1->0
        device: PyTorch device
        generator: Optional random number generator for reproducibility
        
    Returns:
        Tuple of (sequence, row_indices, col_indices, num_edges)
    """
    iu, ju, m = upper_tri_indices(n, device)
    U0 = (torch.rand(m, device=device, generator=generator) < p0)  # bool
    seq = [U0.clone()]
    
    for _ in range(1, T):
        U = seq[-1]
        ones = U
        zeros = ~U
        flips = torch.zeros_like(U)
        
        if ones.any():
            ones_indices = torch.where(ones)[0]
            n_ones = len(ones_indices)
            flips[ones_indices] = (torch.rand(n_ones, device=device, generator=generator) < q10)
        if zeros.any():
            zeros_indices = torch.where(zeros)[0]
            n_zeros = len(zeros_indices)
            flips[zeros_indices] = (torch.rand(n_zeros, device=device, generator=generator) < q01)
            
        seq.append(U ^ flips)
    
    Useq = torch.stack(seq, dim=0)  # (T, m) bool
    return Useq, iu, ju, m

# ------------------------- Permutations & relabeling ------------------------

@torch.no_grad()
def sample_perms_epsilon(B: int, n: int, eps: float, device: torch.device, 
                        g: Optional[torch.Generator] = None) -> torch.Tensor:
    """Sample permutations with epsilon mixture: identity w.p. (1-eps), random w.p. eps."""
    # Random perms
    if g is None:
        perms = torch.stack([torch.randperm(n, device=device) for _ in range(B)], dim=0)
    else:
        perms = torch.stack([torch.randperm(n, generator=g, device=device) for _ in range(B)], dim=0)
    
    # Some identities
    use_id = (torch.rand(B, device=device, generator=g) >= eps)
    if use_id.any():
        perms[use_id] = torch.arange(n, device=device).expand(use_id.sum(), n)
    return perms

@torch.no_grad()
def permute_upper_vec_batch(U_batch: torch.Tensor, perms: torch.Tensor, 
                           iu: torch.Tensor, ju: torch.Tensor, lut: torch.Tensor) -> torch.Tensor:
    """Apply node relabeling to a batch of upper-tri vectors."""
    pi = perms[:, iu]  # (B, m)
    pj = perms[:, ju]  # (B, m)
    a = torch.minimum(pi, pj)
    b = torch.maximum(pi, pj)
    new_idx = lut[a, b]  # (B, m)
    return torch.take_along_dim(U_batch, new_idx, dim=1)

# --------------------------- Greedy reconstruction --------------------------

@torch.no_grad()
def greedy_reconstruct(Useq: torch.Tensor, n: int, eps: float, iu: torch.Tensor, 
                      ju: torch.Tensor, lut: torch.Tensor, device: torch.device,
                      bitpack: bool = False, POPCNT: Optional[torch.Tensor] = None,
                      generator: Optional[torch.Generator] = None,
                      dist_mode: str = "noisy", debug: bool = False) -> Tuple[List[int], int, Optional[Dict]]:
    """Greedy reconstruction with multiple distance modes: 'noisy', 'hamming', 'edgecount'."""
    T, m = Useq.shape
    remaining = torch.arange(T, device=device).tolist()
    order = [remaining.pop(0)]  # Always start with the true first graph
    evals = 0

    # Precompute helpers based on distance mode
    Useq_u8 = pack_bits_bool_to_u8(Useq) if (bitpack and dist_mode in ("noisy", "hamming")) else None
    edge_counts = Useq.sum(dim=1) if dist_mode == "edgecount" else None

    # Debug information collection
    debug_info = {
        "steps": [],
        "sequence_stats": {
            "n": n, "T": T, "dist_mode": dist_mode, "eps": eps,
            "edge_density": float(Useq.float().mean())
        }
    } if debug else None

    step = 0
    while remaining:
        i = order[-1]
        R = len(remaining)
        evals += R

        if dist_mode == "noisy":
            # Original approach: apply random permutations then Hamming distance
            Ui = Useq[i:i+1] if not bitpack else Useq_u8[i:i+1]
            Urem = Useq[remaining]
            perms = sample_perms_epsilon(R, n, eps, device, generator)
            Uperm = permute_upper_vec_batch(Urem, perms, iu, ju, lut)
            
            if bitpack:
                Uperm_u8 = pack_bits_bool_to_u8(Uperm)
                d = hamming_u8(Uperm_u8, Ui.expand_as(Uperm_u8), POPCNT)
            else:
                d = (Uperm ^ Ui.expand_as(Uperm)).sum(dim=1)

        elif dist_mode == "hamming":
            # Direct Hamming distance without permutations
            Ui = Useq[i:i+1] if not bitpack else Useq_u8[i:i+1]
            Urem = Useq[remaining] if not bitpack else Useq_u8[remaining]
            
            if bitpack:
                d = hamming_u8(Urem, Ui.expand_as(Urem), POPCNT)
            else:
                d = (Urem ^ Ui.expand_as(Urem)).sum(dim=1)

        elif dist_mode == "edgecount":
            # Simple edge count difference
            ci = edge_counts[i]
            cj = edge_counts[remaining]
            d = (cj - ci).abs()

        else:
            raise ValueError(f"Unknown distance mode: {dist_mode}. Must be 'noisy', 'hamming', or 'edgecount'")

        k = int(torch.argmin(d))
        chosen = remaining.pop(k)
        order.append(chosen)

        # Collect debug information for this step
        if debug and debug_info is not None:
            step_debug = {
                "step": step,
                "current_node": i,
                "chosen_node": chosen,
                "is_correct": chosen == step + 1,
                "min_distance": float(d.min()),
                "max_distance": float(d.max()),
                "mean_distance": float(d.float().mean())
            }
            
            if dist_mode == "noisy":
                step_debug["permutation_info"] = {
                    "num_identity_perms": int((perms == torch.arange(n, device=device)).all(dim=1).sum()),
                    "num_random_perms": R - int((perms == torch.arange(n, device=device)).all(dim=1).sum())
                }
            elif dist_mode in ["hamming", "edgecount"]:
                step_debug["hamming_info"] = {
                    "current_edges": int(Useq[i].sum()),
                    "chosen_edges": int(Useq[chosen].sum())
                }
            
            debug_info["steps"].append(step_debug)
        
        step += 1

    # Add final debug summary
    if debug and debug_info is not None:
        debug_info["final_summary"] = {
            "is_perfect": order == list(range(T)),
            "stepwise_accuracy": stepwise_accuracy(order),
            "kendall_tau": kendall_tau(order),
            "correct_steps": sum(1 for s in debug_info["steps"] if s["is_correct"]),
            "total_steps": len(debug_info["steps"])
        }

    return order, evals, debug_info

# --------------------------------- Metrics ----------------------------------

def full_order_success(order: List[int]) -> bool:
    """Check if the reconstructed order is exactly correct."""
    return order == list(range(len(order)))

def stepwise_accuracy(order: List[int]) -> float:
    """Compute fraction of consecutive pairs that are correctly ordered."""
    if len(order) <= 1:
        return 1.0
    return sum(1 for t in range(len(order)-1) if order[t+1] == order[t] + 1) / (len(order)-1)

def kendall_tau(order: List[int]) -> float:
    """Compute Kendall's tau correlation coefficient."""
    n = len(order)
    if n <= 1:
        return 1.0
    
    concordant = 0
    total = 0
    
    for i in range(n):
        for j in range(i + 1, n):
            total += 1
            # Check if relative order is preserved
            if (order[i] < order[j]) == (i < j):
                concordant += 1
    
    return (2 * concordant - total) / total if total > 0 else 1.0

def memory_usage_mb() -> float:
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 * 1024)
    return 0.0

# --------------------------------- Driver -----------------------------------

@torch.no_grad()
def run_mc(n: int = 100, T: int = 10, p0: float = 0.8, q01: float = 0.06, q10: float = 0.14, 
           eps_list: List[float] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0), trials: int = 50, 
           device: Optional[str] = None, seed: int = 123, bitpack: bool = True, 
           compile_greedy: bool = False, verbose: bool = True, 
           save_state: bool = False, dist_mode: str = "noisy") -> List[Dict[str, Any]]:
    """Run Monte Carlo simulation for greedy reordering with multiple distance modes."""
    # Validate parameters
    validate_parameters(n, T, p0, q01, q10, eps_list, trials)
    
    device = get_device(device)
    torch.manual_seed(seed)
    g = torch.Generator(device=device).manual_seed(seed)

    if verbose:
        logger.info(f"[device={device}] n={n}, T={T}, trials={trials}, bitpack={bitpack}, dist={dist_mode}, compile={compile_greedy}")
        logger.info(f"Memory usage before: {memory_usage_mb():.1f}MB")
        if dist_mode != "noisy":
            logger.info("(eps is ignored for 'hamming' and 'edgecount' modes)")

    # Precompute structures shared across trials/eps
    iu, ju, m, lut = build_upper_lut(n, device)
    POPCNT = build_popcnt_table(device) if (bitpack and dist_mode in ("noisy", "hamming")) else None

    # (Optional) compiled wrapper
    def _one_trial(eps: float) -> Tuple[List[int], int]:
        Useq, _, _, _ = gen_sequence_flip_upper(n, T, p0, q01, q10, device, g)
        order, evals, _ = greedy_reconstruct(Useq, n, eps, iu, ju, lut, device, bitpack, POPCNT, g, dist_mode)
        return order, evals

    compiled = None
    if compile_greedy and hasattr(torch, "compile"):
        compiled = torch.compile(_one_trial)
        if verbose:
            logger.info("Compiled greedy path enabled.")

    # Choose eps values to iterate (single 0.0 for non-noisy modes)
    eps_iter = eps_list if dist_mode == "noisy" else [0.0]
    
    results = []
    for eps in eps_iter:
        def trial_func():
            if compiled is not None:
                return compiled(eps)
            else:
                return _one_trial(eps)
        
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




# ----------------------------------- CLI ------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Monte-Carlo greedy reordering with multiple distance modes")
    ap.add_argument("--n", type=int, default=100, help="Number of nodes")
    ap.add_argument("--T", type=int, default=10, help="Sequence length")
    ap.add_argument("--p0", type=float, default=0.8, help="Initial edge probability")
    ap.add_argument("--q01", type=float, default=0.06, help="0->1 transition probability")
    ap.add_argument("--q10", type=float, default=0.14, help="1->0 transition probability")
    ap.add_argument("--eps", type=str, default="0.0,0.2,0.4,0.6,0.8,1.0",
                    help="comma-separated eps values (used only for dist=noisy)")
    ap.add_argument("--dist", type=str, default="noisy",
                    choices=["noisy", "hamming", "edgecount"],
                    help="distance mode for greedy selection")
    ap.add_argument("--trials", type=int, default=50, help="Number of trials per epsilon")
    ap.add_argument("--device", type=str, default=None,
                    help="force device: cuda|mps|cpu (default: auto)")
    ap.add_argument("--no-bitpack", action="store_true", help="disable uint8 bitpacking")
    ap.add_argument("--compile", action="store_true", help="torch.compile the inner trial")
    ap.add_argument("--seed", type=int, default=123, help="Random seed")
    ap.add_argument("--save-state", action="store_true", help="Save random state for reproducibility")
    ap.add_argument("--quiet", action="store_true", help="Reduce output verbosity")
    ap.add_argument("--output", type=str, help="Save results to JSON file")
    args = ap.parse_args()

    try:
        eps_list = [float(x) for x in args.eps.split(",") if x.strip() != ""]
        
        results = run_mc(n=args.n, T=args.T, p0=args.p0, q01=args.q01, q10=args.q10,
                        eps_list=eps_list, trials=args.trials, device=args.device,
                        seed=args.seed, bitpack=not args.no_bitpack, 
                        compile_greedy=args.compile, verbose=not args.quiet,
                        save_state=args.save_state, dist_mode=args.dist)
        
        if args.output:
            import json
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to {args.output}")
            
    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        raise

# ----------------------------------- Tests ------------------------------------

if __name__ == "__main__":
    main()