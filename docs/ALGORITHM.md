# Algorithm Details

## Problem Statement

Given a sequence of graphs `G₀, G₁, ..., Gₜ` that evolve according to a Markov chain, we want to reconstruct the correct temporal ordering using only the graph structures. The challenge is made harder by applying random permutations to the node labels, simulating real-world scenarios where node identities might be unknown or corrupted.

## Markov Chain Model

The graph sequence follows a Markov chain where:
- `G₀` is an Erdős-Rényi graph with edge probability `p₀`
- Each `Gᵢ` is obtained from `Gᵢ₋₁` by flipping edges with probabilities:
  - `q₀₁`: probability of adding an edge (0→1)
  - `q₁₀`: probability of removing an edge (1→0)

## Greedy Reconstruction Algorithm

### Algorithm 1: Greedy Reconstruction
```
Input: Graph sequence U = [U₀, U₁, ..., Uₜ], distance mode, parameters
Output: Reconstructed ordering π = [π₀, π₁, ..., πₜ]

1. Initialize: remaining = [1, 2, ..., T], order = [0]
2. For k = 1 to T-1:
   a. i = order[k-1]  // Current graph
   b. For each j in remaining:
      - Compute distance d(i,j) based on mode
   c. j* = argmin_j d(i,j)  // Choose closest graph
   d. order.append(j*)
   e. remaining.remove(j*)
3. Return order
```

## Distance Modes

### 1. Noisy Mode
**Distance**: `min_P d_H(U_i, P(U_j))` where P is a random permutation

**Process**:
1. Sample permutation P^ε (identity w.p. 1-ε, random w.p. ε)
2. Apply permutation: U_j' = P^ε(U_j)
3. Compute Hamming distance: d_H(U_i, U_j')

**Parameters**:
- `ε`: Probability of random vs identity permutation

### 2. Hamming Mode
**Distance**: `d_H(U_i, U_j)` (direct Hamming distance)

**Process**:
1. Compute Hamming distance directly between adjacency matrices
2. No permutations applied

**Use Case**: Baseline comparison for noisy mode

### 3. Edge Count Mode
**Distance**: `|E_i - E_j|` (absolute edge count difference)

**Process**:
1. Count edges in each graph: E_i = |edges(G_i)|
2. Compute absolute difference: |E_i - E_j|

**Use Case**: Test if edge density alone is sufficient

## Performance Optimizations

### Bit-Packing
- Packs boolean adjacency matrices into uint8
- Reduces memory usage by ~8x
- Uses population count lookup tables for Hamming distance

### GPU Acceleration
- Automatic device detection (CUDA > MPS > CPU)
- Significant speedup for large graphs
- Memory-efficient tensor operations

### Compilation
- Optional `torch.compile` for repeated computations
- Provides additional speedup for inner loops

## Complexity Analysis

### Time Complexity
- **Per trial**: O(T² × n²) for distance computations
- **With bit-packing**: O(T² × n²/8) for Hamming distance
- **Total**: O(trials × T² × n²) for full simulation

### Space Complexity
- **Base**: O(n² × T) for adjacency matrices
- **Peak**: O(n² × T²) during reconstruction
- **With bit-packing**: O(n² × T/8) for packed matrices

### Memory Considerations
- Large graphs (n > 5000) may require CPU or reduced T
- Peak memory occurs during greedy reconstruction
- Bit-packing essential for large-scale experiments

## Evaluation Metrics

### 1. Full Order Success
Binary indicator: `order == [0, 1, 2, ..., T-1]`

### 2. Stepwise Accuracy
Fraction of correct individual steps: `Σᵢ I(πᵢ = i) / T`

### 3. Kendall's Tau
Rank correlation coefficient between true and reconstructed orderings

### 4. Candidate Evaluations
Number of distance computations performed during reconstruction

## Theoretical Properties

### Identifiability
- **Perfect case** (ε = 0): Always identifiable if graphs are distinct
- **Noisy case** (ε > 0): Identifiability depends on noise level and graph similarity
- **Edge count only**: Not generally identifiable (multiple orderings possible)

### Convergence
- Greedy algorithm is not guaranteed to find optimal solution
- Performance degrades with increasing noise (ε)
- Edge count mode most robust to noise

### Robustness
- Noisy mode: Robust to small perturbations, sensitive to large ε
- Hamming mode: Sensitive to any structural changes
- Edge count mode: Most robust but least informative
