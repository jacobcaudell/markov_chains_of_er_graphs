# Markov Chain of ER Graphs - Greedy Reordering

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Greedy reconstruction algorithms for sequences of Erdős-Rényi graphs with noisy permutations.

## Problem

Given a sequence of graphs `G₀, G₁, ..., Gₜ` that evolve according to a Markov chain, reconstruct the correct temporal ordering using only graph structures. The challenge is made harder by applying random permutations to node labels, simulating scenarios where node identities are unknown or corrupted.

## Features

- Multiple distance modes (noisy, hamming, edgecount)
- GPU acceleration (CUDA/MPS)
- Memory optimization with bit-packing
- Debug tools for step-by-step analysis

## Installation

```bash
git clone https://github.com/jacobcaudell/markov_chains_of_er_graphs.git
cd markov_chains_of_er_graphs
pip install -r requirements.txt
```

Requirements: `torch`, `numpy`

## Usage

```bash
# Basic simulation
python src/greedy_reorder_mc.py --n 100 --T 10 --trials 50

# Compare all distance modes
python src/compare_modes.py --n 100 --T 10 --trials 50

# Debug single trial
python src/debug_trial.py --n 50 --T 8 --dist noisy --eps 0.2

# View results
python src/analyze_debug.py results.json
```

## Algorithm

Greedy reconstruction algorithm:
1. Start with the first graph (ground truth)
2. For each remaining position, compute distance to all remaining graphs
3. Choose the graph with minimum distance
4. Repeat until all graphs are ordered

Distance computation varies by mode:
- **Noisy**: `min_H d_H(U_i, P^ε(U_j))` where P^ε is a random permutation
- **Hamming**: `d_H(U_i, U_j)` where d_H is Hamming distance  
- **Edge Count**: `|E_i - E_j|` where E_i is the edge count of graph i

## Performance

- **Bit-packing**: Reduces memory usage by ~8x for large graphs
- **GPU acceleration**: Automatic device detection (CUDA > MPS > CPU)
- **Compilation**: Use `--compile` for additional speedup

## Citation

```bibtex
@software{markov_chain_er_graphs,
  title={Markov Chain of ER Graphs - Greedy Reordering},
  author={Jacob Caudell},
  year={2024},
  url={https://github.com/jacobcaudell/markov_chains_of_er_graphs}
}
```

## License

MIT License - see LICENSE file for details.
