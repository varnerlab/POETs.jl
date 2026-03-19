# POETs.jl

**Pareto Optimal Ensemble Techniques for multiobjective optimization in Julia.**

POETs.jl implements a simulated annealing algorithm combined with Pareto ranking to generate ensembles of near-optimal solutions for multiobjective optimization problems. Rather than returning a single "best" solution, POETs produces an archive of solutions that approximate the Pareto front — the set of solutions where no objective can be improved without worsening another.

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/varnerlab/POETs.jl")
```

## Quick Start

POETs requires four user-defined callback functions:

```julia
using POETs

# Define your problem-specific functions:
# objective_function(params)    -> n_objectives × 1 error array
# neighbor_function(params)     -> perturbed parameter vector
# acceptance_probability_function(ranks, temp) -> Float64
# cooling_function(temp)        -> new_temperature

initial_params = rand(10)  # starting point

(error_cache, parameter_cache, rank_array) = estimate_ensemble(
    objective_function,
    neighbor_function,
    acceptance_probability_function,
    cooling_function,
    initial_params;
    rank_cutoff = 4.0,
    maximum_number_of_iterations = 40,
    show_trace = false
)

# Find Pareto-optimal solutions
optimal_idx = findall(rank_array .== 0)
```

## References

- Song S, Chakrabarti A, and J. Varner (2010). Identifying ensembles of signal transduction models using Pareto Optimal Ensemble Techniques (POETs). *Biotechnology Journal*. DOI: [10.1002/biot.201000059](http://www.ncbi.nlm.nih.gov/pmc/articles/PMC3021968/)
- Bassen D, Vilkhovoy M, Minot M, Butcher J and J. Varner (2016). JuPOETs: A Constrained Multiobjective Optimization Approach to Estimate Biochemical Model Ensembles in the Julia Programming Language. *bioRxiv*. DOI: [10.1101/056044](http://biorxiv.org/content/early/2016/05/30/056044)
