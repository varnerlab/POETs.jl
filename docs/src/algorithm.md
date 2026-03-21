# Algorithm

## Overview

ParetoEnsembles combines **simulated annealing** with **Pareto ranking** to explore the solution space of multiobjective optimization problems. The algorithm maintains an archive of solutions and iteratively refines it by generating, evaluating, and filtering candidate solutions.

## How It Works

### 1. Initialization

The algorithm starts with a single solution (`initial_state`). It evaluates the objective function to seed the error cache and parameter cache.

### 2. Inner Loop (Solution Generation)

At each temperature level, the algorithm runs `maximum_number_of_iterations` steps:

1. **Generate** a candidate solution using `neighbor_function(current_best)`
2. **Evaluate** the candidate with `objective_function(candidate)`
3. **Append** the candidate to the archive temporarily
4. **Rank** the new candidate incrementally against existing solutions (O(n·m) instead of full O(n²·m) re-rank)
5. **Accept/reject**: compute `acceptance_probability_function(ranks, temperature)` and compare to a uniform random draw
6. If accepted, **prune** the archive to retain only solutions with rank < `rank_cutoff`, then perform a full re-rank on the smaller pruned archive
7. If rejected, **remove** the candidate from the archive and restore previous ranks

### 3. Outer Loop (Cooling)

After exhausting iterations at a given temperature, `cooling_function(temperature)` reduces the temperature. The algorithm terminates when temperature drops below `temperature_min`.

### 4. Output

The final archive contains the retained solutions: their objective values, parameter vectors, and Pareto ranks.

## Pareto Ranking

A solution **strictly dominates** another if it is at least as good in every objective and strictly better in at least one. The **Pareto rank** of a solution is the number of other solutions in the archive that strictly dominate it.

- **Rank 0**: the solution is on the Pareto front (not dominated by any other solution)
- **Rank k > 0**: k solutions in the archive dominate this one

Identical solutions (equal in all objectives) do **not** dominate each other — they all receive the same rank.

By filtering with `rank_cutoff`, the archive is kept focused on near-optimal solutions. A hard `maximum_archive_size` cap provides an additional safety net against unbounded archive growth.

## Performance Optimizations

### Incremental Ranking

When a single candidate is added, only O(n·m) pairwise comparisons are needed (the new solution vs. each existing one), rather than recomputing all O(n²·m) pairs from scratch. Full re-ranking is only performed after pruning, when the archive is small.

### Pop-on-Reject

Rejected candidates are immediately removed from the archive. This prevents archive bloat between accepted moves, keeping n small and all subsequent ranking operations fast.

### Archive Size Cap

The `maximum_archive_size` keyword argument (default 1000) enforces a hard upper bound. If the archive exceeds this size after pruning, only the best-ranked solutions are retained.

## Parallel Execution

### Threaded Ranking

Set `parallel_evaluation=true` in `estimate_ensemble` to use `Threads.@threads` for Pareto ranking computations. This parallelizes the per-solution dominance checks across available threads. Requires Julia started with multiple threads (`julia -t N`).

### Multi-Chain Parallelism

[`estimate_ensemble_parallel`](@ref) runs multiple independent simulated annealing chains in parallel — one per starting point — then merges their archives and re-ranks. This is embarrassingly parallel and provides:

- Near-linear speedup with thread count
- Better Pareto front coverage from diverse starting points
- Deterministic per-chain RNG for reproducibility

## User-Defined Functions

| Function | Signature | Purpose |
|----------|-----------|---------|
| `objective_function` | `(params) -> Matrix{Float64}` | Returns `n_objectives × 1` array of objective values |
| `neighbor_function` | `(params) -> Vector{Float64}` | Perturbs the current best to generate a candidate |
| `acceptance_probability_function` | `(rank_array, temperature) -> Float64` | Returns probability of accepting (typically `exp(-rank/T)`) |
| `cooling_function` | `(temperature) -> Float64` | Returns reduced temperature (typically `α * T` with `0 < α < 1`) |

### Tips for Callback Design

- **Objective function**: Use penalty terms for constraint violations (see the Binh-Korn test for an example).
- **Neighbor function**: Include bound enforcement to keep parameters feasible. Multiplicative perturbation (`params .* (1 .+ σ * randn(n))`) works well for positive parameters.
- **Acceptance probability**: `exp(-rank[end] / temperature)` is a common choice — it accepts low-rank (good) solutions readily and becomes more selective as temperature decreases.
- **Cooling function**: Geometric cooling `α * temperature` with `α ∈ [0.8, 0.95]` provides a good balance between exploration and convergence.
