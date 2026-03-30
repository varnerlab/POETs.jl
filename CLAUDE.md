# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ParetoEnsembles.jl is a Julia package implementing Pareto Optimal Ensemble Techniques (POETs) for multiobjective parameter estimation. It generates Pareto-optimal ensembles of parameter sets using simulated annealing with strict Pareto dominance ranking. The package has minimal dependencies (only Julia stdlib: LinearAlgebra, Random) and requires Julia 1.10+.

## Common Commands

### Testing
```bash
# Run full test suite
julia --project=. -e 'using Pkg; Pkg.test()'

# Run a single test file
julia --project=. test/runtests.jl

# Run with threads (needed for parallel tests)
julia --threads=auto --project=. -e 'using Pkg; Pkg.test()'
```

### Paper
```bash
cd paper && make all    # Build paper/main.pdf (pdflatex + bibtex)
cd paper && make clean  # Remove build artifacts
```

### Documentation
Documentation uses Documenter.jl. Source is in `docs/src/`.

## Architecture

The entire algorithm lives in two source files:

- **`src/ParetoEnsembles.jl`** — Module definition, exports 5 public functions
- **`src/Main.jl`** — All implementation (~525 lines)

### Public API (5 functions)

| Function | Purpose |
|----------|---------|
| `estimate_ensemble()` | Single-chain Pareto simulated annealing |
| `estimate_ensemble_parallel()` | Multi-chain parallel execution |
| `rank_function()` | Compute Pareto rank for all solutions |
| `hypervolume()` | 2D hypervolume indicator |
| `pareto_front()` | Extract rank-0 (non-dominated) solutions |

### Callback-Based Design

Users provide 4 callback functions to `estimate_ensemble`:
1. **objective** — evaluate parameter vector, return objective values
2. **neighbor** — generate candidate from current solution
3. **acceptance** — probability of accepting worse solution (temperature-dependent)
4. **cooling** — temperature schedule

### Key Algorithmic Details

- **Incremental ranking**: O(n·m) per candidate insertion instead of O(n²·m) full recompute
- **Strict Pareto dominance** (not weak dominance)
- **Pop-on-reject**: rejected candidates immediately removed from archive
- **Full re-rank after pruning**: when filtering by rank_cutoff, ranks recomputed on smaller archive
- **Hard archive cap**: `maximum_archive_size` prevents unbounded growth
- **Threading**: `_rank_columns_threaded()` and `_rank_insert_threaded!()` use `Threads.@threads` with atomic operations

## Testing

Tests use two standard multiobjective benchmarks:
- **Binh-Korn** (`test/binh_korn_function.jl`) — 2 params, 2 objectives, 2 constraints
- **Fonseca-Fleming** (`test/fonseca_fleming_function.jl`) — n params, 2 objectives, unconstrained

The test suite covers: algorithm correctness, edge cases, reproducibility (seeded RNG), incremental vs. full ranking consistency, threaded vs. serial matching, parallel multi-chain execution, hypervolume, and convergence tracing.

## Paper

The research paper is in `paper/` using Elsevier's `elsarticle` class. Reproducible figure scripts are in `paper/code/`, and generated figures go in `paper/figures/` as PDFs. Key examples include synthetic benchmarks, cell-free gene expression, and blood coagulation parameter estimation (with real data from Butenas et al. 1999).
