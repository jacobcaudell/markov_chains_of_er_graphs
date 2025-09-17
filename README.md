# Markov Chain of ER Graphs - Greedy Reordering

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A PyTorch-based implementation for studying greedy reconstruction algorithms on sequences of Erdős-Rényi graphs with noisy permutations.

## Overview

This project implements a Monte Carlo simulation framework for evaluating greedy reconstruction algorithms on sequences of graphs that evolve according to a Markov chain. The key innovation is the use of noisy pairwise permutations to test the robustness of reconstruction algorithms under different distance metrics.

## Problem Statement

Given a sequence of graphs `G₀, G₁, ..., Gₜ` that evolve according to a Markov chain, we want to reconstruct the correct temporal ordering using only the graph structures. The challenge is made harder by applying random permutations to the node labels, simulating real-world scenarios where node identities might be unknown or corrupted.

## Key Features

- **Multiple Distance Modes**: Compare different distance metrics for graph comparison
- **GPU Acceleration**: Full PyTorch implementation with CUDA/MPS support
- **Memory Optimization**: Bit-packing for large-scale simulations
- **Comprehensive Analysis**: Multiple evaluation metrics and debug tools
- **Modular Design**: Clean package structure with focused modules
- **Research Ready**: Debug tools and detailed analysis capabilities

## Installation

### From Source

```bash
git clone https://github.com/yourusername/markov-chain-of-er-graphs.git
cd markov-chain-of-er-graphs
pip install -e .
```

### Requirements

```bash
pip install torch numpy
```

### Optional Dependencies

- **CUDA**: For NVIDIA GPU acceleration
- **MPS**: For Apple Silicon GPU acceleration (macOS)

## Quick Start

### Basic Simulation

Run a Monte Carlo simulation with default parameters:

```bash
python src/greedy_reorder_mc.py
```

### Custom Parameters

```bash
python src/greedy_reorder_mc.py --n 100 --T 10 --trials 50 --dist noisy --eps 0.0,0.2,0.4,0.6,0.8,1.0
```

### Compare All Distance Modes

```bash
python src/compare_modes.py --n 100 --T 10 --trials 50
```

### Debug Single Trial

```bash
python src/debug_trial.py --n 50 --T 8 --dist noisy --eps 0.2
```

### Run Examples

```bash
python examples/examples.py
```

## Package Structure

```
markov-chain-of-er-graphs/
├── src/                          # Core package
│   ├── __init__.py              # Package initialization
│   ├── greedy_reorder_mc.py     # Main simulation driver
│   ├── compare_modes.py         # Cross-mode comparison
│   ├── debug_trial.py           # Single trial debug analysis
│   └── analyze_debug.py         # Results reader
├── examples/                     # Example usage
│   └── examples.py              # Demonstration script
├── docs/                        # Documentation
│   ├── API.md                   # API reference
│   └── ALGORITHM.md             # Algorithm details
├── README.md                    # This file
├── LICENSE                      # MIT license
├── requirements.txt             # Dependencies
└── setup.py                     # Package setup
```

## Scripts Overview

### 1. `src/greedy_reorder_mc.py` - Main Simulation Driver

The core Monte Carlo simulation script for running experiments with different distance modes.

**Key Features:**
- Single distance mode per run
- Multiple epsilon values for noisy mode
- Comprehensive statistics collection
- JSON output support

**Usage:**
```bash
python src/greedy_reorder_mc.py [OPTIONS]
```

**Key Options:**
- `--n`: Number of nodes (default: 100)
- `--T`: Sequence length (default: 10)
- `--dist`: Distance mode (`noisy`, `hamming`, `edgecount`)
- `--eps`: Epsilon values for noisy mode (comma-separated)
- `--trials`: Number of trials per epsilon (default: 50)
- `--device`: Force device (`cuda`, `mps`, `cpu`)
- `--compile`: Use torch.compile for speedup
- `--output`: Save results to JSON file

### 2. `src/compare_modes.py` - Cross-Mode Comparison

Compare all distance modes on the same seeded trials for fair comparison.

**Key Features:**
- Same random seeds across all modes
- Fair comparison of different distance metrics
- Comprehensive performance analysis

**Usage:**
```bash
python src/compare_modes.py [OPTIONS]
```

### 3. `src/debug_trial.py` - Single Trial Analysis

Detailed analysis of a single trial with step-by-step reconstruction information.

**Key Features:**
- Step-by-step reconstruction details
- Edge evolution analysis
- Performance metrics per step
- JSON output for further analysis

**Usage:**
```bash
python src/debug_trial.py [OPTIONS]
```

### 4. `src/analyze_debug.py` - Results Reader

Parse and display results from JSON files in a human-readable format.

**Usage:**
```bash
python src/analyze_debug.py results.json
```

## Distance Modes

### 1. Noisy Mode (`--dist noisy`)
- **Description**: Original approach using random permutations
- **Process**: 
  1. Apply random permutation P^ε to each remaining graph
  2. Compute Hamming distance with current graph
  3. Choose graph with minimum distance
- **Parameters**: `eps` controls probability of random vs identity permutation

### 2. Hamming Mode (`--dist hamming`)
- **Description**: Direct Hamming distance without permutations
- **Process**: Compute Hamming distance directly between adjacency matrices
- **Use Case**: Baseline comparison for noisy mode

### 3. Edge Count Mode (`--dist edgecount`)
- **Description**: Simple edge count difference
- **Process**: Compare absolute difference in number of edges
- **Use Case**: Test if edge density alone is sufficient for reconstruction

## Algorithm Details

### Greedy Reconstruction Algorithm

1. **Initialize**: Start with the first graph (ground truth)
2. **Iterate**: For each remaining position:
   - Compute distance from current graph to all remaining graphs
   - Choose the graph with minimum distance
   - Add to reconstructed sequence
3. **Output**: Return the reconstructed ordering

### Distance Computation

The distance computation varies by mode:

- **Noisy**: `min_H d_H(U_i, P^ε(U_j))` where P^ε is a random permutation
- **Hamming**: `d_H(U_i, U_j)` where d_H is Hamming distance
- **Edge Count**: `|E_i - E_j|` where E_i is the edge count of graph i

## Performance Optimizations

### Bit-Packing
- Packs boolean adjacency matrices into uint8 for memory efficiency
- Reduces memory usage by ~8x for large graphs
- Enabled by default, disable with `--no-bitpack`

### GPU Acceleration
- Automatic device detection (CUDA > MPS > CPU)
- Force specific device with `--device`
- Significant speedup for large graphs

### Compilation
- Use `--compile` for torch.compile optimization
- Provides additional speedup for repeated computations

## Memory Considerations

### Memory Usage Estimation
- Base memory: ~O(n²T) for adjacency matrices
- Peak memory: ~O(n²T²) during reconstruction
- Large graphs (n > 5000) may require CPU or reduced T

### Recommended Parameters
- **Small**: n=100, T=10 (fast, good for testing)
- **Medium**: n=1000, T=20 (balanced performance)
- **Large**: n=5000, T=10 (memory-intensive)

## Output Format

### Simulation Results
```json
[
  {
    "dist_mode": "noisy",
    "eps": 0.2,
    "full_order_success_rate": 0.85,
    "mean_stepwise_accuracy": 0.92,
    "mean_kendall_tau": 0.88,
    "avg_candidate_evals": 45.2,
    "sec": 12.5,
    "sec_per_trial": 0.25,
    "memory_usage_mb": 256.7
  }
]
```

### Debug Output
```json
{
  "sequence_stats": {
    "n": 100,
    "T": 10,
    "dist_mode": "noisy",
    "eps": 0.2,
    "edge_density": 0.15
  },
  "sequence_analysis": {
    "true_order": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    "reconstructed_order": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    "is_perfect": true,
    "stepwise_accuracy": 1.0,
    "kendall_tau": 1.0
  }
}
```

## Examples

### Example 1: Basic Simulation
```bash
# Run 50 trials with noisy mode, epsilon from 0 to 1
python src/greedy_reorder_mc.py --n 100 --T 10 --trials 50 --dist noisy --eps 0.0,0.2,0.4,0.6,0.8,1.0 --output results.json
```

### Example 2: Compare All Modes
```bash
# Compare all distance modes on same trials
python src/compare_modes.py --n 200 --T 15 --trials 100 --output comparison.json
```

### Example 3: Debug Analysis
```bash
# Debug a single trial with detailed output
python src/debug_trial.py --n 50 --T 8 --dist noisy --eps 0.3 --output debug.json
python src/analyze_debug.py debug.json
```

### Example 4: Large Scale Simulation
```bash
# Large graph with GPU acceleration
python src/greedy_reorder_mc.py --n 1000 --T 20 --trials 100 --device cuda --compile --output large_results.json
```

### Example 5: Using as Python Package
```python
from src import run_mc, run_comparison, run_debug_trial

# Run simulation programmatically
results = run_mc(n=100, T=10, trials=50, dist_mode="noisy")

# Compare all modes
comparison = run_comparison(n=100, T=10, trials=50)

# Debug single trial
debug_info = run_debug_trial(n=50, T=8, dist_mode="noisy", eps=0.3)
```

## Troubleshooting

### Common Issues

1. **Memory Errors**: Reduce `n` or `T`, or use `--device cpu`
2. **Slow Performance**: Enable `--compile` or use GPU
3. **MPS Errors**: Use `--device cpu` on Apple Silicon if issues occur

### Performance Tips

1. Use bit-packing (default) for memory efficiency
2. Enable compilation for repeated runs
3. Use GPU for large graphs when available
4. Start with small parameters to test setup

## Citation

If you use this code in your research, please cite:

```bibtex
@software{markov_chain_er_graphs,
  title={Markov Chain of ER Graphs - Greedy Reordering},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/markov_chain_of_er_graphs}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- PyTorch team for the excellent deep learning framework
- The graph theory community for foundational algorithms
- Contributors and testers who helped improve this implementation
