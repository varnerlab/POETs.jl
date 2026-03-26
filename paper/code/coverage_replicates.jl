# Multi-replicate coverage analysis for the coagulation ensemble
#
# Runs the synthetic coagulation study 5 times with different noise seeds
# to assess whether coverage failures are systematic or noise-dependent.
#
# Run from the paper/code directory:
#   julia --project -t4 coverage_replicates.jl

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

function build_params(x_log)
    p_full = copy(P_TRUE)
    for (i, pidx) in enumerate(ESTIMATE_INDICES)
        p_full[pidx] = 10.0^x_log[i]
    end
    return p_full
end

# True trajectories at validation conditions
valid_true = Vector{Vector{Float64}}(undef, 3)
valid_times = Vector{Vector{Float64}}(undef, 3)
for (i, tf) in enumerate(VALID_TF)
    t, thr = simulate_thrombin(P_TRUE; TF_pM=tf)
    valid_times[i] = t
    valid_true[i] = thr
end

# TGA features
feature_names = [:lagtime, :peak, :tpeak, :max_rate, :etp]
feature_labels = ["Lag time", "Peak", "Time-to-peak", "Max rate", "ETP"]
feature_scale = [1.0, 1e9, 1.0, 1e9, 1e9]

# ──────────────────────────────────────────────────────────────
# Run 5 replicates with different noise seeds
# ──────────────────────────────────────────────────────────────
const N_REPS = 5
const NOISE_SEEDS = [2024, 2025, 2026, 2027, 2028]

# Collect results across replicates
all_peak_errs = Dict(tf => Float64[] for tf in VALID_TF)
all_traj_coverage = Dict(tf => Float64[] for tf in VALID_TF)
all_feature_coverage = Dict((tf, fn) => Bool[] for tf in VALID_TF, fn in feature_names)

for (rep, noise_seed) in enumerate(NOISE_SEEDS)
    println("\n=== Replicate $rep (noise seed = $noise_seed) ===")

    # Generate noisy training data
    Random.seed!(noise_seed)
    train_data = Vector{Vector{Float64}}(undef, 3)
    for (i, tf) in enumerate(TRAIN_TF)
        _, thr = simulate_thrombin(P_TRUE; TF_pM=tf)
        train_data[i] = max.(thr .* (1.0 .+ NOISE_CV .* randn(length(thr))), 0.0)
    end

    # Objective function (closure over this replicate's data)
    function obj_func(x_log)
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

    neighbor_func(x_log) = clamp.(x_log .+ 0.05 .* randn(N_EST), LOG_LOWER, LOG_UPPER)
    accept_func(R, T) = exp(-R[end] / T)
    cool_func(T) = 0.92 * T

    # Run ensemble
    initial_states = [LOG_LOWER .+ rand(MersenneTwister(s + 100*rep), N_EST) .* (LOG_UPPER .- LOG_LOWER) for s in 1:8]

    (EC, PC, RA) = estimate_ensemble_parallel(
        obj_func, neighbor_func, accept_func, cool_func,
        initial_states;
        rank_cutoff = 10, maximum_number_of_iterations = 40,
        maximum_archive_size = 2000, show_trace = false,
        rng_seed = noise_seed)

    ens_idx = findall(RA .<= 1)
    n_ens = length(ens_idx)
    println("  Ensemble size: $n_ens")

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
    for i in 1:3
        Thr_valid[i] = Thr_valid[i][valid_mask, :]
    end
    ens_valid = ens_idx[valid_mask]
    n_valid = sum(valid_mask)

    # Compute metrics
    for (i, tf) in enumerate(VALID_TF)
        μ = vec(mean(Thr_valid[i], dims=1)) .* 1e9
        lo = vec(mapslices(x -> quantile(x, 0.025), Thr_valid[i], dims=1)) .* 1e9
        hi = vec(mapslices(x -> quantile(x, 0.975), Thr_valid[i], dims=1)) .* 1e9
        true_nM = valid_true[i] .* 1e9

        # Peak error
        true_peak = maximum(true_nM)
        ens_peak = maximum(μ)
        err_pct = abs(ens_peak - true_peak) / true_peak * 100
        push!(all_peak_errs[tf], err_pct)

        # Trajectory coverage (fraction of time points where true falls in 95% CI)
        covered_frac = mean(lo .<= true_nM .<= hi)
        push!(all_traj_coverage[tf], covered_frac)

        # TGA feature coverage
        for fn in feature_names
            fs = feature_scale[findfirst(==(fn), feature_names)]
            sol_true = simulate(HockinMann2002;
                TF_concentration = tf * 1e-12,
                tspan = (0.0, 1200.0), saveat = 1.0, p = P_TRUE)
            feat_true = extract_tga_features(HockinMann2002, sol_true)
            true_val = getfield(feat_true, fn) * fs

            feat_vals = Float64[]
            for idx in ens_valid
                p_full = build_params(PC[:, idx])
                try
                    sol = simulate(HockinMann2002;
                        TF_concentration = tf * 1e-12,
                        tspan = (0.0, 1200.0), saveat = 1.0, p = p_full)
                    feat = extract_tga_features(HockinMann2002, sol)
                    push!(feat_vals, getfield(feat, fn) * fs)
                catch; end
            end

            if length(feat_vals) > 10
                lo_f = quantile(feat_vals, 0.025)
                hi_f = quantile(feat_vals, 0.975)
                push!(all_feature_coverage[(tf, fn)], lo_f <= true_val <= hi_f)
            end
        end

        println("  $(tf) pM: peak err=$(round(err_pct, digits=1))%, traj coverage=$(round(covered_frac*100, digits=1))%")
    end
end

# ──────────────────────────────────────────────────────────────
# Summary across replicates
# ──────────────────────────────────────────────────────────────
println("\n" * "="^70)
println("SUMMARY ACROSS $N_REPS REPLICATES")
println("="^70)

println("\nPeak thrombin error (%) at held-out conditions:")
for tf in VALID_TF
    errs = all_peak_errs[tf]
    println("  $(tf) pM: $(round(mean(errs), digits=1)) ± $(round(std(errs), digits=1))% " *
            "(range: $(round(minimum(errs), digits=1))--$(round(maximum(errs), digits=1))%)")
end

println("\nTrajectory-level coverage (fraction of time points in 95% CI):")
for tf in VALID_TF
    covs = all_traj_coverage[tf]
    println("  $(tf) pM: $(round(mean(covs)*100, digits=1)) ± $(round(std(covs)*100, digits=1))%")
end

println("\nTGA feature coverage (fraction of replicates where true value in 95% CI):")
for fn in feature_names
    fl = feature_labels[findfirst(==(fn), feature_names)]
    for tf in VALID_TF
        covs = all_feature_coverage[(tf, fn)]
        rate = mean(covs) * 100
        println("  $fl at $(tf) pM: $(round(rate, digits=0))% ($(sum(covs))/$N_REPS)")
    end
end

println("\nDone!")
