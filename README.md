# Markov Chain of ER Graphs - Greedy Reordering

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A PyTorch-based implementation for studying greedy reconstruction algorithms on sequences of Erdős-Rényi graphs with noisy permutations.

## Quick Start

```bash
# Clone and install
git clone https://github.com/jacobcaudell/markov_chains_of_er_graphs.git
cd markov_chains_of_er_graphs
pip install -r requirements.txt

# Run basic simulation
python src/greedy_reorder_mc.py --n 100 --T 10 --trials 50

# Compare all distance modes
python src/compare_modes.py --n 100 --T 10 --trials 50
```

## Problem Statement

Given a sequence of graphs `G₀, G₁, ..., Gₜ` that evolve according to a Markov chain, reconstruct the correct temporal ordering using only graph structures. The challenge is made harder by applying random permutations to node labels, simulating scenarios where node identities are unknown or corrupted.

## Algorithm

**Greedy Reconstruction:**
1. Start with the first graph (ground truth)
2. For each remaining position, compute distance to all remaining graphs
3. Choose the graph with minimum distance
4. Repeat until all graphs are ordered

**Distance Modes:**
- **Noisy**: `min_H d_H(U_i, P^ε(U_j))` where P^ε is a random permutation
- **Hamming**: `d_H(U_i, U_j)` where d_H is Hamming distance  
- **Edge Count**: `|E_i - E_j|` where E_i is the edge count of graph i

## Usage

### Command Line Tools

```bash
# Basic simulation with different parameters
python src/greedy_reorder_mc.py --n 100 --T 10 --trials 50 --dist noisy --eps 0.2

# Compare all distance modes on same trials
python src/compare_modes.py --n 100 --T 10 --trials 50

# Debug single trial with detailed output
python src/debug_trial.py --n 50 --T 8 --dist noisy --eps 0.2

# Analyze results from JSON output
python src/analyze_debug.py results.json
```

### Programmatic API

```python
from src import run_mc, run_comparison, run_debug_trial

# Run Monte Carlo simulation
results = run_mc(n=100, T=10, trials=50, dist_mode="noisy", eps_list=[0.1, 0.2])

# Compare all modes
comparison = run_comparison(n=100, T=10, trials=50)

# Debug single trial
debug_info = run_debug_trial(n=50, T=8, dist_mode="noisy", eps=0.2)
```

## Performance Features

- **GPU Acceleration**: Automatic device detection (CUDA > MPS > CPU)
- **Memory Optimization**: Bit-packing reduces memory usage by ~8x for large graphs
- **Compilation**: Use `--compile` for additional speedup with `torch.compile`
- **Batch Processing**: Efficient tensor operations for multiple trials

## Project Structure

```
├── src/                    # Main source code
│   ├── greedy_reorder_mc.py    # Core simulation driver
│   ├── compare_modes.py        # Multi-mode comparison
│   ├── debug_trial.py          # Single trial debugging
│   └── analyze_debug.py        # Results analysis
├── docs/                   # Documentation
├── examples/               # Usage examples
└── tests/                  # Test suite
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- NumPy

## Citation

```bibtex
@software{markov_chain_er_graphs,
  title={Markov Chain of ER Graphs - Greedy Reordering},
  author={Jacob Caudell},
  year={2025},
  url={https://github.com/jacobcaudell/markov_chains_of_er_graphs}
}
```

## License

MIT License - see [LICENSE](LICENSE) file for details.