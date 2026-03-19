## POETs.jl - Pareto Optimal Ensemble Techniques

[![CI](https://github.com/varnerlab/POETs.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/varnerlab/POETs.jl/actions/workflows/CI.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

POETs.jl is a [Julia](https://julialang.org) package that implements the Pareto Optimal Ensemble Techniques (POETs) method for multiobjective optimization using simulated annealing with Pareto ranking.

### References

- [Song S, Chakrabarti A, and J. Varner (2010) Identifying ensembles of signal transduction models using Pareto Optimal Ensemble Techniques (POETs). Biotechnology Journal DOI: 10.1002/biot.201000059](http://www.ncbi.nlm.nih.gov/pmc/articles/PMC3021968/)
- [Bassen D, Vilkhovoy M, Minot M, Butcher J and J. Varner (2016) JuPOETs: A Constrained Multiobjective Optimization Approach to Estimate Biochemical Model Ensembles in the Julia Programming Language. bioArXiv doi: http://dx.doi.org/10.1101/056044](http://biorxiv.org/content/early/2016/05/30/056044)

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/varnerlab/POETs.jl")
```

## Usage

```julia
using POETs
```

POETs requires four user-defined callback functions:

| Function | Signature | Purpose |
|----------|-----------|---------|
| `objective_function` | `(parameter_array) -> error_matrix` | Evaluates fitness (returns `n_objectives × 1` array) |
| `neighbor_function` | `(parameter_array) -> new_parameter_array` | Generates a candidate solution from the current one |
| `acceptance_probability_function` | `(rank_array, temperature) -> Float64` | Computes probability of accepting a new solution |
| `cooling_function` | `(temperature) -> new_temperature` | Decreases temperature for the annealing schedule |

### Running the optimizer

```julia
(error_cache, parameter_cache, rank_array) = estimate_ensemble(
    objective_function,
    neighbor_function,
    acceptance_probability_function,
    cooling_function,
    initial_parameter_array;
    rank_cutoff=4.0,
    maximum_number_of_iterations=20,
    temperature_min=0.0001,
    show_trace=true
)
```

**Returns:**
- `error_cache` - objective values for retained solutions (`n_objectives × n_solutions`)
- `parameter_cache` - parameter vectors for retained solutions (`n_params × n_solutions`)
- `rank_array` - Pareto rank of each retained solution (0 = Pareto optimal)

### Pareto ranking

```julia
ranks = rank_function(error_cache)
```

Computes the Pareto rank for each column in the error cache. A rank of 0 means the solution is on the Pareto front (not dominated by any other solution).

## Examples

See the `sample/biochemical` directory for a complete example of biochemical model parameter estimation from four conflicting training data sets. The driver is `run_biochemical_test.jl`, with callback functions in `hcmem_lib.jl`.

Test problems (Binh-Korn and Fonseca-Fleming benchmarks) are in the `test/` directory.

## Testing

```julia
using Pkg
Pkg.test("POETs")
```
