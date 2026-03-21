# Example 2: Multi-chain parallel execution on the Binh-Korn benchmark
# Reproduces Listing 2 from the paper.
# Run with: julia -t4 --project example_parallel.jl

using ParetoEnsembles
using Random

# --- Same callbacks as Example 1 ---
function objective_function(x)
    f = zeros(2, 1)
    f[1] = 4.0 * x[1]^2 + 4.0 * x[2]^2
    f[2] = (x[1] - 5)^2 + (x[2] - 5)^2
    lambda = 100.0
    v1 = 25 - (x[1] - 5)^2 - x[2]^2
    v2 = (x[1] - 8)^2 + (x[2] - 3)^2 - 7.7
    f[1] += lambda * (min(0, v1))^2
    f[2] += lambda * (min(0, v2))^2
    return f
end

neighbor_function(x) = clamp.(x .* (1 .+ 0.05 * randn(length(x))), [0.0, 0.0], [5.0, 3.0])
acceptance_probability_function(R, T) = exp(-R[end] / T)
cooling_function(T) = 0.9 * T

# --- Run multi-chain parallel optimization ---
initial_states = [
    [2.5, 1.5],
    [0.5, 2.5],
    [4.0, 0.5],
    [1.0, 1.0],
]

println("Running $(length(initial_states)) chains on $(Threads.nthreads()) threads...")
(EC, PC, RA) = estimate_ensemble_parallel(
    objective_function, neighbor_function,
    acceptance_probability_function, cooling_function,
    initial_states;
    rank_cutoff=4.0,
    maximum_number_of_iterations=40,
    show_trace=false,
    rng_seed=42
)

println("Solutions from $(length(initial_states)) chains: ", size(EC, 2))
println("Pareto-optimal: ", count(RA .== 0))
println("f1 range: ", extrema(EC[1, RA .== 0]))
println("f2 range: ", extrema(EC[2, RA .== 0]))
