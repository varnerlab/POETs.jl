#=
 * Copyright (c) 2016. Varnerlab,
 * School of Chemical and Biomolecular Engineering,
 * Cornell University, Ithaca NY 14853
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * Created by jeffreyvarner on 5/4/2016
 =#

# -- PUBLIC FUNCTIONS BELOW HERE ------------------------------------------------------------------------------------------------------------ #

"""
    estimate_ensemble(objective_function, neighbor_function, acceptance_probability_function, cooling_function, initial_state; kwargs...)

Estimate a Pareto optimal ensemble of solutions using simulated annealing with Pareto ranking.

The algorithm iteratively generates candidate solutions via `neighbor_function`, evaluates them with
`objective_function`, and accepts or rejects them based on Pareto rank and a temperature-dependent
acceptance probability. Solutions with rank below `rank_cutoff` are retained in the archive.

# Arguments
- `objective_function::Function`: evaluates a parameter vector and returns an `n_objectives × 1` error array.
- `neighbor_function::Function`: generates a new candidate parameter vector from the current best.
- `acceptance_probability_function::Function`: `(rank_array, temperature) -> Float64` probability of accepting.
- `cooling_function::Function`: `(temperature) -> new_temperature` for the annealing schedule.
- `initial_state::AbstractVector{<:Real}`: starting parameter vector.

# Keyword Arguments
- `maximum_number_of_iterations::Integer = 20`: iterations per temperature level.
- `rank_cutoff::Real = 5.0`: solutions with Pareto rank ≥ `rank_cutoff` are pruned from the archive.
- `temperature_min::Real = 0.0001`: annealing stops when temperature falls below this value.
- `show_trace::Bool = true`: print iteration and temperature at each accepted step.
- `rng::AbstractRNG = Random.default_rng()`: random number generator for reproducibility.

# Returns
A tuple `(error_cache, parameter_cache, pareto_rank_array)` where:
- `error_cache::Matrix{Float64}`: objective values for retained solutions (`n_objectives × n_solutions`).
- `parameter_cache::Matrix{Float64}`: parameter vectors for retained solutions (`n_params × n_solutions`).
- `pareto_rank_array::Vector{Float64}`: Pareto rank of each retained solution (0 = Pareto optimal).

# Example
```julia
using Random
rng = MersenneTwister(42)  # for reproducibility
(EC, PC, RA) = estimate_ensemble(
    obj_fn, neighbor_fn, accept_fn, cool_fn, x0;
    rank_cutoff=4.0, maximum_number_of_iterations=40, rng=rng
)
```

See also [`rank_function`](@ref).
"""
function estimate_ensemble(objective_function::Function, neighbor_function::Function,
  acceptance_probability_function::Function, cooling_function::Function,
  initial_state::AbstractVector{<:Real}; maximum_number_of_iterations::Integer = 20,
  rank_cutoff::Real = 5.0, temperature_min::Real = 0.0001, show_trace::Bool = true,
  rng::AbstractRNG = Random.default_rng())

  # initialize — use column vectors to avoid hcat copies in the inner loop
  temperature = 1.0
  parameter_array_best = collect(Float64, initial_state)
  first_error = vec(objective_function(parameter_array_best))

  error_cols = Vector{Float64}[first_error]
  param_cols = Vector{Float64}[copy(parameter_array_best)]

  # main loop -
  while temperature > temperature_min

    for iteration_index in 1:(maximum_number_of_iterations + 1)

      # generate and evaluate a new solution -
      test_parameter_array = neighbor_function(parameter_array_best)
      test_error = vec(objective_function(test_parameter_array))

      # append to archive -
      push!(error_cols, test_error)
      push!(param_cols, test_parameter_array)

      # compute the Pareto rank -
      pareto_rank_array = _rank_columns(error_cols)

      # do we accept the new solution?
      acceptance_probability = acceptance_probability_function(pareto_rank_array, temperature)
      if acceptance_probability > rand(rng)

        # prune archive to solutions below rank cutoff -
        keep = findall(pareto_rank_array .< rank_cutoff)
        error_cols = error_cols[keep]
        param_cols = param_cols[keep]

        # update the parameters -
        parameter_array_best = test_parameter_array
        if show_trace
          @show iteration_index, temperature
        end
      end
    end

    # update the temperature -
    temperature = cooling_function(temperature)

  end

  # build output matrices -
  error_cache = reduce(hcat, error_cols)
  parameter_cache = reduce(hcat, param_cols)
  pareto_rank_array = rank_function(error_cache)

  return (error_cache, parameter_cache, pareto_rank_array)
end

# Internal: compute Pareto rank directly on column vectors (allocation-free inner loop)
function _rank_columns(cols::Vector{Vector{Float64}})
  n_trials = length(cols)
  n_obj = length(cols[1])
  rank_array = zeros(n_trials)

  @inbounds for i in 1:n_trials
    count = 0
    for j in 1:n_trials
      # check if solution j weakly dominates solution i (j <= i in all objectives)
      dominates = true
      for k in 1:n_obj
        if cols[j][k] > cols[i][k]
          dominates = false
          break
        end
      end
      if dominates
        count += 1
      end
    end
    rank_array[i] = count - 1  # subtract self
  end

  return rank_array
end

"""
    rank_function(error_cache) -> Vector{Float64}

Compute the Pareto rank for each solution (column) in `error_cache`.

The Pareto rank of a solution is the number of other solutions in the archive that dominate it
(i.e., are better or equal in all objectives and strictly better in at least one). A rank of 0
means the solution lies on the Pareto front.

# Arguments
- `error_cache::AbstractMatrix`: an `n_objectives × n_solutions` matrix of objective values (lower is better).

# Returns
- `rank_array::Vector{Float64}`: Pareto rank for each solution.

# Example
```julia
errors = [1.0 2.0 1.5; 2.0 1.0 1.5]
ranks = rank_function(errors)  # [0.0, 0.0, 1.0]
```

See also [`estimate_ensemble`](@ref).
"""
function rank_function(error_cache)

  (n_obj, n_trials) = size(error_cache)
  rank_array = zeros(n_trials)

  @inbounds for i in 1:n_trials
    count = 0
    for j in 1:n_trials
      # check if solution j weakly dominates solution i
      dominates = true
      for k in 1:n_obj
        if error_cache[k, j] > error_cache[k, i]
          dominates = false
          break
        end
      end
      if dominates
        count += 1
      end
    end
    rank_array[i] = count - 1  # subtract self
  end

  return rank_array
end
# -- PUBLIC FUNCTIONS ABOVE HERE ------------------------------------------------------------------------------------------------------------ #
