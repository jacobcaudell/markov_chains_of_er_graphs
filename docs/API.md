# API Reference

## Core Functions

### `run_mc()`
Main Monte Carlo simulation function.

```python
from src import run_mc

results = run_mc(
    n=100,                    # Number of nodes
    T=10,                     # Sequence length  
    p0=0.8,                   # Initial edge probability
    q01=0.06,                 # 0->1 transition probability
    q10=0.14,                 # 1->0 transition probability
    eps_list=[0.0, 0.2, 0.4], # Epsilon values (noisy mode only)
    trials=50,                # Number of trials per epsilon
    device=None,              # Device (auto-detect if None)
    seed=123,                 # Random seed
    bitpack=True,             # Use bit-packing optimization
    compile_greedy=False,     # Use torch.compile
    verbose=True,             # Print progress
    save_state=False,         # Save random state
    dist_mode="noisy"         # Distance mode
)
```

### `greedy_reconstruct()`
Core greedy reconstruction algorithm.

```python
from src import greedy_reconstruct

order, evals, debug_info = greedy_reconstruct(
    Useq,                     # Graph sequence tensor
    n,                        # Number of nodes
    eps,                      # Epsilon value
    iu, ju, lut,             # Precomputed indices and lookup table
    device,                   # PyTorch device
    bitpack=True,             # Use bit-packing
    POPCNT=None,              # Population count table
    generator=None,           # Random generator
    dist_mode="noisy",        # Distance mode
    debug=False               # Collect debug info
)
```

### `run_comparison()`
Compare all distance modes on same seeded trials.

```python
from src import run_comparison

results = run_comparison(
    n=100, T=10, trials=50,
    eps_list=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    device=None, seed=123, bitpack=True,
    compile_greedy=False, verbose=True, save_state=False
)
```

### `run_debug_trial()`
Single trial with detailed debug information.

```python
from src import run_debug_trial

debug_info = run_debug_trial(
    n=100, T=10, p0=0.8, q01=0.06, q10=0.14,
    eps=0.0, dist_mode="noisy", device=None,
    seed=123, bitpack=True
)
```

### `read_results_file()`
Read and display results from JSON files.

```python
from src import read_results_file

read_results_file("results.json")
```

## Distance Modes

- **`"noisy"`**: Original approach with random permutations
- **`"hamming"`**: Direct Hamming distance without permutations  
- **`"edgecount"`**: Simple edge count difference

## Return Values

### Simulation Results
```python
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
```

### Debug Information
```python
{
    "steps": [...],           # Step-by-step reconstruction details
    "sequence_stats": {...},  # Basic sequence information
    "final_summary": {...},   # Overall performance metrics
    "sequence_analysis": {...} # Ordering and edge analysis
}
```
