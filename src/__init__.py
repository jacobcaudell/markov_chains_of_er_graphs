"""
Markov Chain of ER Graphs - Greedy Reordering

A PyTorch-based implementation for studying greedy reconstruction algorithms 
on sequences of Erdős-Rényi graphs with noisy permutations.
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .greedy_reorder_mc import run_mc, greedy_reconstruct
from .compare_modes import run_comparison
from .debug_trial import run_debug_trial
from .analyze_debug import read_results_file

__all__ = [
    "run_mc",
    "greedy_reconstruct", 
    "run_comparison",
    "run_debug_trial",
    "read_results_file"
]
