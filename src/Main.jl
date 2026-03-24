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
- `maximum_archive_size::Integer = 1000`: hard cap on archive size; if exceeded after pruning, only the best-ranked solutions are kept.
- `parallel_evaluation::Bool = false`: use threaded Pareto ranking. Requires Julia started with multiple threads (`julia -t N`).
- `trace::Bool = false`: when `true`, collect convergence diagnostics at each temperature step.
- `trace_reference_point::Union{Nothing,AbstractVector} = nothing`: reference point for hypervolume computation in the trace. Required for 2-objective problems when `trace=true`.

# Returns
When `trace=false` (default), a tuple `(error_cache, parameter_cache, pareto_rank_array)`.
When `trace=true`, a 4-tuple `(error_cache, parameter_cache, pareto_rank_array, trace_log)` where
`trace_log` is a vector of named tuples with fields `temperature`, `archive_size`, and `hypervolume`.

The first three elements are:
- `error_cache::Matrix{Float64}`: objective values for retained solutions (`n_objectives × n_solutions`).
- `parameter_cache::Matrix{Float64}`: parameter vectors for retained solutions (`n_params × n_solutions`).
- `pareto_rank_array::Vector{Float64}`: Pareto rank of each retained solution (0 = Pareto optimal).

# Example
```julia
using Random
rng = MersenneTwister(42)  # for reproducibility
(EC, PC, RA) = estimate_ensemble(
    obj_fn, neighbor_fn, accept_fn, cool_fn, x0;
    rank_cutoff=4.0, maximum_number_of_iterations=40, rng=rng,
    maximum_archive_size=500
)
```

See also [`rank_function`](@ref).
"""
function estimate_ensemble(objective_function::Function, neighbor_function::Function,
  acceptance_probability_function::Function, cooling_function::Function,
  initial_state::AbstractVector{<:Real}; maximum_number_of_iterations::Integer = 20,
  rank_cutoff::Real = 5.0, temperature_min::Real = 0.0001, show_trace::Bool = true,
  rng::AbstractRNG = Random.default_rng(), maximum_archive_size::Integer = 1000,
  parallel_evaluation::Bool = false, trace::Bool = false,
  trace_reference_point::Union{Nothing,AbstractVector} = nothing)

  # select serial or threaded ranking functions
  _rank_fn = parallel_evaluation ? _rank_columns_threaded : _rank_columns
  _insert_fn! = parallel_evaluation ? _rank_insert_threaded! : _rank_insert!

  # initialize — use column vectors to avoid hcat copies in the inner loop
  temperature = 1.0
  parameter_array_best = collect(Float64, initial_state)
  first_error = vec(objective_function(parameter_array_best))

  error_cols = Vector{Float64}[first_error]
  param_cols = Vector{Float64}[copy(parameter_array_best)]
  rank_array = zeros(1)  # initial solution has rank 0

  # convergence trace storage
  trace_log = NamedTuple{(:temperature,:archive_size,:hypervolume),Tuple{Float64,Int,Float64}}[]

  # main loop -
  while temperature > temperature_min

    for iteration_index in 1:(maximum_number_of_iterations + 1)

      # generate and evaluate a new solution -
      test_parameter_array = neighbor_function(parameter_array_best)
      test_error = vec(objective_function(test_parameter_array))

      # save ranks before insertion (for cheap restore on reject)
      old_ranks = copy(rank_array)

      # temporarily append candidate to archive -
      push!(error_cols, test_error)
      push!(param_cols, test_parameter_array)

      # incremental rank update — O(n·m) instead of O(n²·m)
      _insert_fn!(error_cols, rank_array)

      # do we accept the new solution?
      acceptance_probability = acceptance_probability_function(rank_array, temperature)
      if acceptance_probability > rand(rng)

        # prune archive to solutions below rank cutoff -
        keep = findall(rank_array .< rank_cutoff)
        error_cols = error_cols[keep]
        param_cols = param_cols[keep]

        # full re-rank after prune (archive is small post-prune)
        rank_array = _rank_fn(error_cols)

        # enforce hard archive size cap -
        if length(error_cols) > maximum_archive_size
          perm = sortperm(rank_array)
          keep_cap = perm[1:maximum_archive_size]
          error_cols = error_cols[keep_cap]
          param_cols = param_cols[keep_cap]
          rank_array = _rank_fn(error_cols)
        end

        # update the parameters -
        parameter_array_best = test_parameter_array
        if show_trace
          @show iteration_index, temperature
        end
      else
        # rejected — remove candidate and restore ranks
        pop!(error_cols)
        pop!(param_cols)
        rank_array = old_ranks
      end
    end

    # update the temperature -
    temperature = cooling_function(temperature)

    # record trace if requested
    if trace
      n_arch = length(error_cols)
      hv = if !isnothing(trace_reference_point) && length(error_cols[1]) == 2
        ec_tmp = reduce(hcat, error_cols)
        hypervolume(ec_tmp, trace_reference_point)
      else
        NaN
      end
      push!(trace_log, (temperature=temperature, archive_size=n_arch, hypervolume=hv))
    end

  end

  # build output matrices -
  error_cache = reduce(hcat, error_cols)
  parameter_cache = reduce(hcat, param_cols)
  pareto_rank_array = rank_function(error_cache)

  if trace
    return (error_cache, parameter_cache, pareto_rank_array, trace_log)
  else
    return (error_cache, parameter_cache, pareto_rank_array)
  end
end

# Internal: compute Pareto rank directly on column vectors (allocation-free inner loop)
function _rank_columns(cols::Vector{Vector{Float64}})
  n_trials = length(cols)
  n_obj = length(cols[1])
  rank_array = zeros(n_trials)

  @inbounds for i in 1:n_trials
    count = 0
    for j in 1:n_trials
      # check if solution j strictly dominates solution i
      # (j <= i in all objectives AND j < i in at least one)
      dominates = true
      strictly_better = false
      for k in 1:n_obj
        if cols[j][k] > cols[i][k]
          dominates = false
          break
        elseif cols[j][k] < cols[i][k]
          strictly_better = true
        end
      end
      if dominates && strictly_better
        count += 1
      end
    end
    rank_array[i] = count
  end

  return rank_array
end

# Internal: incremental rank update when one new solution is appended — O(n·m)
function _rank_insert!(cols::Vector{Vector{Float64}}, rank_array::Vector{Float64})
  n = length(cols)
  n_obj = length(cols[1])
  new_rank = 0.0

  @inbounds for i in 1:(n - 1)
    dom_new_over_i = true
    strict_new = false
    dom_i_over_new = true
    strict_i = false

    for k in 1:n_obj
      val_new = cols[n][k]
      val_i = cols[i][k]
      if val_new > val_i
        dom_new_over_i = false
      elseif val_new < val_i
        strict_new = true
      end
      if val_i > val_new
        dom_i_over_new = false
      elseif val_i < val_new
        strict_i = true
      end
    end

    if dom_new_over_i && strict_new
      rank_array[i] += 1
    end
    if dom_i_over_new && strict_i
      new_rank += 1
    end
  end

  push!(rank_array, new_rank)
  return rank_array
end

# Internal: threaded full Pareto rank — O(n²·m / nthreads)
function _rank_columns_threaded(cols::Vector{Vector{Float64}})
  n_trials = length(cols)
  n_obj = length(cols[1])
  rank_array = zeros(n_trials)

  Threads.@threads for i in 1:n_trials
    count = 0
    @inbounds for j in 1:n_trials
      dominates = true
      strictly_better = false
      for k in 1:n_obj
        if cols[j][k] > cols[i][k]
          dominates = false
          break
        elseif cols[j][k] < cols[i][k]
          strictly_better = true
        end
      end
      if dominates && strictly_better
        count += 1
      end
    end
    rank_array[i] = count
  end

  return rank_array
end

# Internal: threaded incremental rank update — O(n·m / nthreads)
function _rank_insert_threaded!(cols::Vector{Vector{Float64}}, rank_array::Vector{Float64})
  n = length(cols)
  n_obj = length(cols[1])
  new_rank = Threads.Atomic{Int}(0)

  Threads.@threads for i in 1:(n - 1)
    dom_new_over_i = true
    strict_new = false
    dom_i_over_new = true
    strict_i = false

    @inbounds for k in 1:n_obj
      val_new = cols[n][k]
      val_i = cols[i][k]
      if val_new > val_i
        dom_new_over_i = false
      elseif val_new < val_i
        strict_new = true
      end
      if val_i > val_new
        dom_i_over_new = false
      elseif val_i < val_new
        strict_i = true
      end
    end

    if dom_new_over_i && strict_new
      rank_array[i] += 1
    end
    if dom_i_over_new && strict_i
      Threads.atomic_add!(new_rank, 1)
    end
  end

  push!(rank_array, Float64(new_rank[]))
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
      # check if solution j strictly dominates solution i
      # (j <= i in all objectives AND j < i in at least one)
      dominates = true
      strictly_better = false
      for k in 1:n_obj
        if error_cache[k, j] > error_cache[k, i]
          dominates = false
          break
        elseif error_cache[k, j] < error_cache[k, i]
          strictly_better = true
        end
      end
      if dominates && strictly_better
        count += 1
      end
    end
    rank_array[i] = count
  end

  return rank_array
end

"""
    hypervolume(error_cache::AbstractMatrix, reference_point::AbstractVector) -> Float64

Compute the hypervolume indicator for a 2D Pareto front.

The hypervolume is the area of objective space that is dominated by the Pareto front and bounded
above by the `reference_point`. Larger values indicate better front quality.

# Arguments
- `error_cache::AbstractMatrix`: `2 × n_solutions` matrix of objective values (lower is better).
- `reference_point::AbstractVector`: `[ref1, ref2]` upper bound in each objective.

# Returns
- `Float64`: hypervolume indicator value.

# Example
```julia
errors = [1.0 3.0; 3.0 1.0]
hv = hypervolume(errors, [5.0, 5.0])
```

See also [`rank_function`](@ref), [`pareto_front`](@ref).
"""
function hypervolume(error_cache::AbstractMatrix, reference_point::AbstractVector)
  @assert size(error_cache, 1) == 2 "hypervolume currently supports 2 objectives only"
  @assert length(reference_point) == 2

  n = size(error_cache, 2)
  valid = Tuple{Float64,Float64}[]
  for i in 1:n
    if error_cache[1,i] < reference_point[1] && error_cache[2,i] < reference_point[2]
      push!(valid, (error_cache[1,i], error_cache[2,i]))
    end
  end
  isempty(valid) && return 0.0

  sort!(valid, by=first)

  # extract non-dominated subset (sweep from left, track min f2)
  front = Tuple{Float64,Float64}[]
  min_f2 = Inf
  for pt in valid
    if pt[2] < min_f2
      push!(front, pt)
      min_f2 = pt[2]
    end
  end

  # compute area via sweep-line
  hv = 0.0
  for i in eachindex(front)
    f1_right = (i < length(front)) ? front[i+1][1] : reference_point[1]
    hv += (f1_right - front[i][1]) * (reference_point[2] - front[i][2])
  end
  return hv
end

"""
    pareto_front(error_cache, parameter_cache, rank_array) -> (front_errors, front_params)

Extract Pareto-optimal (rank = 0) solutions from the archive.

# Arguments
- `error_cache::AbstractMatrix`: `n_objectives × n_solutions` objective values.
- `parameter_cache::AbstractMatrix`: `n_params × n_solutions` parameter vectors.
- `rank_array::AbstractVector`: Pareto rank for each solution.

# Returns
A tuple `(front_errors, front_params)` containing only the rank-0 solutions.

# Example
```julia
(EC, PC, RA) = estimate_ensemble(...)
front_E, front_P = pareto_front(EC, PC, RA)
```

See also [`rank_function`](@ref), [`hypervolume`](@ref).
"""
function pareto_front(error_cache::AbstractMatrix, parameter_cache::AbstractMatrix,
                      rank_array::AbstractVector)
  idx = findall(rank_array .== 0)
  return error_cache[:, idx], parameter_cache[:, idx]
end

"""
    estimate_ensemble_parallel(objective_function, neighbor_function, acceptance_probability_function, cooling_function, initial_states; kwargs...)

Run multiple independent Pareto simulated annealing chains in parallel, then merge their archives.

Each element of `initial_states` seeds one chain that runs `estimate_ensemble` on its own thread.
Results are merged into a single archive and re-ranked. This gives near-linear speedup with
thread count and better Pareto front coverage from diverse starting points.

Requires Julia started with multiple threads (`julia -t N`).

# Arguments
- `objective_function`, `neighbor_function`, `acceptance_probability_function`, `cooling_function`: same as [`estimate_ensemble`](@ref).
- `initial_states::AbstractVector{<:AbstractVector{<:Real}}`: one starting parameter vector per chain.

# Keyword Arguments
All keyword arguments from [`estimate_ensemble`](@ref) are supported and forwarded to each chain.
The `rng` kwarg is ignored; each chain gets its own deterministic RNG seeded from `rng_seed` and the chain index.
- `rng_seed::Integer = 42`: base seed for per-chain RNG generation.

# Returns
Same as [`estimate_ensemble`](@ref): `(error_cache, parameter_cache, pareto_rank_array)` for the merged archive.

# Example
```julia
starts = [[2.5, 1.5], [1.0, 2.0], [3.0, 1.0], [4.0, 0.5]]
(EC, PC, RA) = estimate_ensemble_parallel(
    obj_fn, neighbor_fn, accept_fn, cool_fn, starts;
    rank_cutoff=4.0, maximum_number_of_iterations=40
)
```

See also [`estimate_ensemble`](@ref).
"""
function estimate_ensemble_parallel(objective_function::Function,
  neighbor_function::Function,
  acceptance_probability_function::Function,
  cooling_function::Function,
  initial_states::AbstractVector{<:AbstractVector{<:Real}};
  rng_seed::Integer = 42, kwargs...)

  n_chains = length(initial_states)
  results = Vector{Any}(undef, n_chains)

  # filter out rng from kwargs — each chain gets its own
  fwd_kwargs = Dict{Symbol,Any}(k => v for (k, v) in kwargs if k !== :rng)

  Threads.@threads for c in 1:n_chains
    chain_rng = MersenneTwister(hash((rng_seed, c)))
    results[c] = estimate_ensemble(
      objective_function, neighbor_function,
      acceptance_probability_function, cooling_function,
      initial_states[c]; rng=chain_rng, fwd_kwargs...
    )
  end

  # merge all archives
  all_errors = reduce(hcat, [r[1] for r in results])
  all_params = reduce(hcat, [r[2] for r in results])

  # re-rank the merged archive
  merged_ranks = rank_function(all_errors)

  return (all_errors, all_params, merged_ranks)
end
# -- PUBLIC FUNCTIONS ABOVE HERE ------------------------------------------------------------------------------------------------------------ #
