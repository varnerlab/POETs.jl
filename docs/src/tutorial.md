# Tutorial: Binh-Korn Benchmark

This tutorial walks through solving the Binh-Korn constrained multiobjective optimization problem using ParetoEnsembles.jl.

## Problem Definition

The Binh-Korn problem has two objectives and two constraints:

**Objectives** (minimize):
- ``f_1(x, y) = 4x^2 + 4y^2``
- ``f_2(x, y) = (x - 5)^2 + (y - 5)^2``

**Constraints**:
- ``(x - 5)^2 + y^2 \leq 25``
- ``(x - 8)^2 + (y - 3)^2 \geq 7.7``

**Bounds**: ``0 \leq x \leq 5``, ``0 \leq y \leq 3``

## Step 1: Define the Objective Function

The objective function returns an `n_objectives × 1` matrix. Constraint violations are handled via penalty terms:

```julia
BIG = 1e10

function objective_function(parameter_array)
    x = parameter_array[1]
    y = parameter_array[2]

    obj_array = BIG * ones(2, 1)
    obj_array[1] = 4.0 * (x^2) + 4.0 * (y^2)
    obj_array[2] = (x - 5)^2 + (y - 5)^2

    # Constraint violations as penalties
    lambda_value = 100.0
    v1 = 25 - (x - 5.0)^2 - y^2
    v2 = (x - 8.0)^2 + (y - 3.0)^2 - 7.7
    penalty = zeros(2)
    penalty[1] = lambda_value * (min(0, v1))^2
    penalty[2] = lambda_value * (min(0, v2))^2

    return obj_array + penalty
end
```

## Step 2: Define the Neighbor Function

The neighbor function perturbs the current solution and enforces bounds:

```julia
function neighbor_function(parameter_array)
    SIGMA = 0.05
    n = length(parameter_array)
    new_params = parameter_array .* (fill(1, n) + SIGMA * randn(n))

    # Enforce bounds
    lower = [0.0, 0.0]
    upper = [5.0, 3.0]
    return clamp.(new_params, lower, upper)
end
```

## Step 3: Define Acceptance and Cooling Functions

```julia
function acceptance_probability_function(rank_array, temperature)
    return exp(-rank_array[end] / temperature)
end

function cooling_function(temperature)
    return 0.9 * temperature
end
```

## Step 4: Run the Optimization

```julia
using ParetoEnsembles

initial_state = [2.5, 1.5]

(EC, PC, RA) = estimate_ensemble(
    objective_function,
    neighbor_function,
    acceptance_probability_function,
    cooling_function,
    initial_state;
    rank_cutoff = 4.0,
    maximum_number_of_iterations = 40,
    show_trace = false
)
```

## Step 5: Analyze Results

```julia
# How many solutions in the archive?
println("Solutions retained: ", size(EC, 2))

# Find Pareto-optimal solutions (rank 0)
pareto_idx = findall(RA .== 0)
println("Pareto-optimal solutions: ", length(pareto_idx))

# Extract the Pareto front
pareto_f1 = EC[1, pareto_idx]
pareto_f2 = EC[2, pareto_idx]

# Extract corresponding parameters
pareto_params = PC[:, pareto_idx]
```

## Biochemical Example

For a more realistic application, see the `sample/biochemical/` directory in the repository.
It demonstrates parameter estimation for a metabolic network model with four conflicting
experimental data sets, including local refinement steps and ensemble visualization.
