# Example 3: Fonseca-Fleming unconstrained benchmark (d=3)
# Reproduces Example 3 from the paper.

using ParetoEnsembles
using Random

const D = 3  # number of decision variables

# --- Objective function ---
function objective_function(x)
    f = zeros(2, 1)
    sum1 = sum((x[i] - 1 / sqrt(D))^2 for i in 1:D)
    sum2 = sum((x[i] + 1 / sqrt(D))^2 for i in 1:D)
    f[1] = 1 - exp(-sum1)
    f[2] = 1 - exp(-sum2)
    return f
end

# --- Neighbor function with box constraints ---
function neighbor_function(x)
    new_x = x .* (1 .+ 0.05 * randn(length(x)))
    return clamp.(new_x, -4.0, 4.0)
end

# --- Acceptance and cooling ---
acceptance_probability_function(R, T) = exp(-R[end] / T)
cooling_function(T) = 0.9 * T

# --- Run optimization ---
println("Running Fonseca-Fleming (d=$D) optimization...")
(EC, PC, RA) = estimate_ensemble(
    objective_function, neighbor_function,
    acceptance_probability_function, cooling_function,
    randn(D);
    rank_cutoff=4.0,
    maximum_number_of_iterations=40,
    show_trace=false
)

println("Solutions retained: ", size(EC, 2))
pareto_idx = findall(RA .== 0)
println("Pareto-optimal solutions: ", length(pareto_idx))
println("f1 range: ", extrema(EC[1, pareto_idx]))
println("f2 range: ", extrema(EC[2, pareto_idx]))

# Check theoretical property: f1 + f2 should approach 1 - exp(-d) on the front
theoretical_sum = 1 - exp(-D)
actual_sums = EC[1, pareto_idx] .+ EC[2, pareto_idx]
println("Theoretical f1+f2 on front: ", round(theoretical_sum, digits=4))
println("Mean f1+f2 on Pareto front: ", round(sum(actual_sums) / length(actual_sums), digits=4))
