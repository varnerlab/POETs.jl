# Example 1: Binh-Korn constrained multiobjective benchmark
# Reproduces Listing 1 from the paper.

using ParetoEnsembles
using Random

# --- Objective function ---
function objective_function(x)
    f = zeros(2, 1)
    f[1] = 4.0 * x[1]^2 + 4.0 * x[2]^2
    f[2] = (x[1] - 5)^2 + (x[2] - 5)^2

    # penalty for constraint violations
    lambda = 100.0
    v1 = 25 - (x[1] - 5)^2 - x[2]^2
    v2 = (x[1] - 8)^2 + (x[2] - 3)^2 - 7.7
    f[1] += lambda * (min(0, v1))^2
    f[2] += lambda * (min(0, v2))^2
    return f
end

# --- Neighbor function (multiplicative perturbation with bounds) ---
function neighbor_function(x)
    return clamp.(x .* (1 .+ 0.05 * randn(length(x))), [0.0, 0.0], [5.0, 3.0])
end

# --- Acceptance probability (Boltzmann-like on rank) ---
acceptance_probability_function(R, T) = exp(-R[end] / T)

# --- Geometric cooling ---
cooling_function(T) = 0.9 * T

# --- Run single-chain optimization ---
println("Running single-chain Binh-Korn optimization...")
(EC, PC, RA) = estimate_ensemble(
    objective_function, neighbor_function,
    acceptance_probability_function, cooling_function,
    [2.5, 1.5];
    rank_cutoff=4.0,
    maximum_number_of_iterations=40,
    show_trace=false
)

println("Solutions retained: ", size(EC, 2))
pareto_idx = findall(RA .== 0)
println("Pareto-optimal solutions: ", length(pareto_idx))
println("f1 range: ", extrema(EC[1, pareto_idx]))
println("f2 range: ", extrema(EC[2, pareto_idx]))
