# Rank cutoff sensitivity analysis for the coagulation ensemble
#
# Tests whether the choice of rank ≤ 1 for ensemble membership affects
# downstream predictions. Sweeps rank cutoff from 0 to 5 and reports
# held-out prediction accuracy and coverage.
#
# This script re-runs the synthetic coagulation example with different
# rank cutoffs for ensemble selection.
#
# Run from the paper/code directory:
#   julia --project -t4 rank_cutoff_sensitivity.jl

using ParetoEnsembles
using HockinMannModel
using Random
using Statistics

# ──────────────────────────────────────────────────────────────
# Model setup (identical to example_coagulation.jl)
# ──────────────────────────────────────────────────────────────
const P_TRUE = default_rate_constants(HockinMann2002)
const ESTIMATE_INDICES = [10, 15, 16, 17, 22, 26, 31, 32, 38, 41]
const TRUE_LOG = log10.(P_TRUE[ESTIMATE_INDICES])
const N_EST = length(ESTIMATE_INDICES)
const LOG_LOWER = TRUE_LOG .- 1.5
const LOG_UPPER = TRUE_LOG .+ 1.5

const NOISE_CV = 0.15
const TRAIN_TF = [5.0, 15.0, 25.0]
const VALID_TF = [10.0, 20.0, 30.0]

function simulate_thrombin(p_full; TF_pM, saveat=10.0)
    sol = simulate(HockinMann2002;
        TF_concentration = TF_pM * 1e-12,
        tspan = (0.0, 1200.0), saveat = saveat, p = p_full)
    return sol.t, total_thrombin(HockinMann2002, sol)
end

# Generate training data (same seed as main study)
Random.seed!(2024)
train_times = Vector{Vector{Float64}}(undef, 3)
train_data  = Vector{Vector{Float64}}(undef, 3)
for (i, tf) in enumerate(TRAIN_TF)
    t, thr = simulate_thrombin(P_TRUE; TF_pM=tf)
    train_times[i] = t
    train_data[i]  = max.(thr .* (1.0 .+ NOISE_CV .* randn(length(thr))), 0.0)
end

# Objective function
function objective_function(x_log)
    f = zeros(3, 1)
    x = 10.0 .^ x_log
    p_full = copy(P_TRUE)
    for (i, idx) in enumerate(ESTIMATE_INDICES)
        p_full[idx] = x[i]
    end
    try
        for (obj_i, tf) in enumerate(TRAIN_TF)
            _, thrombin = simulate_thrombin(p_full; TF_pM=tf)
            norm = sum(train_data[obj_i] .^ 2) + 1e-30
            f[obj_i] = sum((thrombin .- train_data[obj_i]) .^ 2) / norm
        end
    catch
        f .= 1e6
    end
    return f
end

neighbor_function(x_log) = clamp.(x_log .+ 0.05 .* randn(N_EST), LOG_LOWER, LOG_UPPER)
acceptance_probability_function(R, T) = exp(-R[end] / T)
cooling_function(T) = 0.92 * T

# Run ensemble estimation
println("Running ensemble estimation (8 chains)...")
initial_states = [LOG_LOWER .+ rand(MersenneTwister(s), N_EST) .* (LOG_UPPER .- LOG_LOWER) for s in 1:8]

(EC, PC, RA) = estimate_ensemble_parallel(
    objective_function, neighbor_function,
    acceptance_probability_function, cooling_function,
    initial_states;
    rank_cutoff = 10, maximum_number_of_iterations = 40,
    maximum_archive_size = 2000, show_trace = false, rng_seed = 2024)

println("  Total solutions: $(size(EC, 2)), max rank: $(maximum(RA))")

# ──────────────────────────────────────────────────────────────
# Sweep rank cutoffs and compute validation metrics
# ──────────────────────────────────────────────────────────────
function build_params(x_log)
    p_full = copy(P_TRUE)
    for (i, pidx) in enumerate(ESTIMATE_INDICES)
        p_full[pidx] = 10.0^x_log[i]
    end
    return p_full
end

# True trajectories at validation conditions
valid_true = Vector{Vector{Float64}}(undef, 3)
for (i, tf) in enumerate(VALID_TF)
    _, thr = simulate_thrombin(P_TRUE; TF_pM=tf)
    valid_true[i] = thr
end

rank_cutoffs = [0, 1, 2, 3, 5]

println("\n=== Rank cutoff sensitivity ===")
println("Cutoff | Ensemble size | Peak err 10pM | Peak err 20pM | Peak err 30pM | Mean peak err")
println("-------|---------------|---------------|---------------|---------------|-------------")

for rc in rank_cutoffs
    ens_idx = findall(RA .<= rc)
    n_ens = length(ens_idx)

    if n_ens < 5
        println("  rank ≤ $rc: only $n_ens members, skipping")
        continue
    end

    # Simulate at validation conditions
    n_t = length(valid_true[1])
    Thr_valid = [zeros(n_ens, n_t) for _ in 1:3]

    for (k, idx) in enumerate(ens_idx)
        p_full = build_params(PC[:, idx])
        for (i, tf) in enumerate(VALID_TF)
            try
                _, thr = simulate_thrombin(p_full; TF_pM=tf)
                Thr_valid[i][k, :] = thr
            catch
                Thr_valid[i][k, :] .= NaN
            end
        end
    end

    # Filter valid
    valid_mask = ones(Bool, n_ens)
    for i in 1:3
        valid_mask .&= .!any(isnan.(Thr_valid[i]), dims=2)[:]
    end
    n_valid = sum(valid_mask)

    peak_errs = Float64[]
    for (i, tf) in enumerate(VALID_TF)
        M = Thr_valid[i][valid_mask, :]
        μ = vec(mean(M, dims=1)) .* 1e9
        true_peak = maximum(valid_true[i]) * 1e9
        ens_peak = maximum(μ)
        err_pct = abs(ens_peak - true_peak) / true_peak * 100
        push!(peak_errs, err_pct)
    end

    mean_err = mean(peak_errs)
    println("  ≤ $rc   |     $n_valid        | $(round(peak_errs[1], digits=1))%          | $(round(peak_errs[2], digits=1))%          | $(round(peak_errs[3], digits=1))%          | $(round(mean_err, digits=1))%")
end

println("\nDone!")
