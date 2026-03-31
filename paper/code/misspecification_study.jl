# Model misspecification study for the coagulation ensemble
#
# Data are generated from a PERTURBED Hockin-Mann model (6 fixed rate constants
# shifted by ±30%), then fitted with the NOMINAL model. This breaks the inverse
# crime: the fitting model cannot perfectly reproduce the data.
#
# Compares: correct-model ensemble vs misspecified-model ensemble
# on held-out prediction accuracy and coverage.
#
# Run from the paper/code directory:
#   julia --project -t4 misspecification_study.jl

using ParetoEnsembles
using HockinMannModel
using CairoMakie
using Random
using Statistics
using JLD2

const FIGDIR = joinpath(@__DIR__, "..", "figures")
const CACHEDIR = joinpath(@__DIR__, "data")
const CACHE_FILE = joinpath(CACHEDIR, "misspecification_results.jld2")

# ──────────────────────────────────────────────────────────────
# Model setup (same estimated parameters as main study)
# ──────────────────────────────────────────────────────────────
const P_NOMINAL = default_rate_constants(HockinMann2002)
const ESTIMATE_INDICES = [10, 15, 16, 17, 22, 26, 31, 32, 38, 41]
const ESTIMATE_NAMES = [
    "extrinsic Xase kcat",      # p[10]
    "TF=VIIa→IX kcat",          # p[15]
    "Xa→IIa k",                 # p[16]
    "IIa→VIIIa k",              # p[17]
    "intrinsic Xase kcat",      # p[22]
    "IIa→Va k",                 # p[26]
    "prothrombinase kcat",      # p[31]
    "mIIa→IIa k",               # p[32]
    "Xa+ATIII k",                # p[38]
    "IIa+ATIII k",               # p[41]
]
const TRUE_LOG = log10.(P_NOMINAL[ESTIMATE_INDICES])
const N_EST = length(ESTIMATE_INDICES)
const LOG_LOWER = TRUE_LOG .- 1.5
const LOG_UPPER = TRUE_LOG .+ 1.5

# ──────────────────────────────────────────────────────────────
# Perturbed model: shift 6 FIXED rate constants by ±30%
# These span initiation, amplification, and inhibition pathways
# but are NOT in the estimated set.
# ──────────────────────────────────────────────────────────────
const PERTURB_INDICES = [5, 9, 14, 28, 34, 39]
const PERTURB_NAMES = [
    "TF=VIIa+VII→VIIa (initiation)",     # p[5]  = 4.4e5
    "extrinsic Xase Km (binding)",         # p[9]  = 2.5e7
    "IX activation Km (binding)",          # p[14] = 1.0e7
    "prothrombinase assembly on-rate",     # p[28] = 4.0e8
    "Xa+TFPI on-rate (inhibition)",        # p[34] = 9.0e5
    "mIIa+ATIII (inhibition)",             # p[39] = 7.1e3
]
# Alternate signs: +30%, -30%, +30%, -30%, +30%, -30%
const PERTURB_FACTORS = [1.3, 0.7, 1.3, 0.7, 1.3, 0.7]

P_PERTURBED = copy(P_NOMINAL)
for (idx, fac) in zip(PERTURB_INDICES, PERTURB_FACTORS)
    P_PERTURBED[idx] *= fac
end

println("Model misspecification study")
println("Perturbed parameters (NOT estimated):")
for (i, (idx, name, fac)) in enumerate(zip(PERTURB_INDICES, PERTURB_NAMES, PERTURB_FACTORS))
    println("  p[$idx] $name: $(P_NOMINAL[idx]) → $(P_PERTURBED[idx]) (×$fac)")
end

# ──────────────────────────────────────────────────────────────
# Generate data from PERTURBED model
# ──────────────────────────────────────────────────────────────
function simulate_thrombin(p_full; TF_pM, saveat=10.0)
    sol = simulate(HockinMann2002;
        TF_concentration = TF_pM * 1e-12,
        tspan = (0.0, 1200.0),
        saveat = saveat,
        p = p_full)
    return sol.t, total_thrombin(HockinMann2002, sol)
end

const NOISE_CV = 0.15
const TRAIN_TF = [5.0, 15.0, 25.0]
const VALID_TF = [10.0, 20.0, 30.0]

Random.seed!(2024)

# Training data from perturbed model
train_times = Vector{Vector{Float64}}(undef, 3)
train_true_perturbed = Vector{Vector{Float64}}(undef, 3)
train_data = Vector{Vector{Float64}}(undef, 3)

for (i, tf) in enumerate(TRAIN_TF)
    t, thr = simulate_thrombin(P_PERTURBED; TF_pM=tf)
    train_times[i] = t
    train_true_perturbed[i] = thr
    noisy = thr .* (1.0 .+ NOISE_CV .* randn(length(thr)))
    train_data[i] = max.(noisy, 0.0)
    println("Training data ($(tf) pM TF): peak = $(round(maximum(thr)*1e9, digits=1)) nM (from perturbed model)")
end

# True trajectories at validation conditions (from perturbed model — this is ground truth)
valid_times = Vector{Vector{Float64}}(undef, 3)
valid_true = Vector{Vector{Float64}}(undef, 3)
for (i, tf) in enumerate(VALID_TF)
    t, thr = simulate_thrombin(P_PERTURBED; TF_pM=tf)
    valid_times[i] = t
    valid_true[i] = thr
end

# ──────────────────────────────────────────────────────────────
# Fit with NOMINAL model (misspecified)
# ──────────────────────────────────────────────────────────────
function objective_function(x_log)
    f = zeros(3, 1)
    x = 10.0 .^ x_log
    p_full = copy(P_NOMINAL)  # <-- nominal, not perturbed
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

function neighbor_function(x_log)
    new_x = x_log .+ 0.05 .* randn(N_EST)
    return clamp.(new_x, LOG_LOWER, LOG_UPPER)
end

acceptance_probability_function(R, T) = exp(-R[end] / T)
cooling_function(T) = 0.92 * T

println("\nRunning ensemble estimation with misspecified model (8 chains)...")
initial_states = [LOG_LOWER .+ rand(MersenneTwister(s), N_EST) .* (LOG_UPPER .- LOG_LOWER) for s in 1:8]

(EC, PC, RA) = estimate_ensemble_parallel(
    objective_function, neighbor_function,
    acceptance_probability_function, cooling_function,
    initial_states;
    rank_cutoff = 10,
    maximum_number_of_iterations = 40,
    maximum_archive_size = 2000,
    show_trace = false,
    rng_seed = 2024
)

n_total = size(EC, 2)
n_pareto = count(RA .== 0)
println("  Total solutions: $n_total, Pareto-optimal: $n_pareto")

# ──────────────────────────────────────────────────────────────
# Select ensemble and compute predictions
# ──────────────────────────────────────────────────────────────
ensemble_idx = findall(RA .<= 1)
n_ens = length(ensemble_idx)
println("  Ensemble size (rank ≤ 1): $n_ens")

function build_params_nominal(x_log)
    p_full = copy(P_NOMINAL)
    x = 10.0 .^ x_log
    for (i, pidx) in enumerate(ESTIMATE_INDICES)
        p_full[pidx] = x[i]
    end
    return p_full
end

function ensemble_stats_nM(M)
    μ  = vec(mean(M, dims=1)) .* 1e9
    lo = vec(mapslices(x -> quantile(x, 0.025), M, dims=1)) .* 1e9
    hi = vec(mapslices(x -> quantile(x, 0.975), M, dims=1)) .* 1e9
    return μ, lo, hi
end

# Simulate ensemble at training conditions
n_t = length(train_times[1])
Thr_train = [zeros(n_ens, n_t) for _ in 1:3]
for (k, idx) in enumerate(ensemble_idx)
    p_full = build_params_nominal(PC[:, idx])
    for (i, tf) in enumerate(TRAIN_TF)
        try
            _, thr = simulate_thrombin(p_full; TF_pM=tf)
            Thr_train[i][k, :] = thr
        catch
            Thr_train[i][k, :] .= NaN
        end
    end
end

# Simulate ensemble at validation conditions
Thr_valid = [zeros(n_ens, n_t) for _ in 1:3]
for (k, idx) in enumerate(ensemble_idx)
    p_full = build_params_nominal(PC[:, idx])
    for (i, tf) in enumerate(VALID_TF)
        try
            _, thr = simulate_thrombin(p_full; TF_pM=tf)
            Thr_valid[i][k, :] = thr
        catch
            Thr_valid[i][k, :] .= NaN
        end
    end
end

# Filter valid members
valid_train = .!any(hcat([any(isnan.(M), dims=2)[:] for M in Thr_train]...), dims=2)[:]
valid_pred  = .!any(hcat([any(isnan.(M), dims=2)[:] for M in Thr_valid]...), dims=2)[:]
valid_mask = valid_train .& valid_pred

for i in 1:3
    Thr_train[i] = Thr_train[i][valid_mask, :]
    Thr_valid[i] = Thr_valid[i][valid_mask, :]
end
ens_idx_valid = ensemble_idx[valid_mask]
n_valid = sum(valid_mask)
println("  Valid ensemble members: $n_valid / $n_ens")

# ──────────────────────────────────────────────────────────────
# Compute validation metrics
# ──────────────────────────────────────────────────────────────
println("\n=== Validation: Held-out predictions (misspecified model) ===")
for (i, tf) in enumerate(VALID_TF)
    μ, lo, hi = ensemble_stats_nM(Thr_valid[i])
    true_nM = valid_true[i] .* 1e9
    true_peak = maximum(true_nM)
    ens_peak = maximum(μ)
    err_pct = abs(ens_peak - true_peak) / true_peak * 100
    # Coverage: fraction of time points where true falls within 95% CI
    covered_frac = mean(lo .<= true_nM .<= hi)
    println("  $(tf) pM TF: true peak=$(round(true_peak, digits=1)) nM, " *
            "pred peak=$(round(ens_peak, digits=1)) nM, err=$(round(err_pct, digits=1))%, " *
            "coverage=$(round(covered_frac*100, digits=1))%")
end

# TGA features at held-out conditions
println("\nTGA feature predictions (misspecified model):")
feature_names = [:lagtime, :peak, :tpeak, :max_rate, :etp]
feature_labels = ["Lag time (s)", "Peak (nM)", "Time to peak (s)", "Max rate (nM/s)", "ETP (nM·s)"]
feature_scale = [1.0, 1e9, 1.0, 1e9, 1e9]

for tf in VALID_TF
    # True features from perturbed model
    sol_true = simulate(HockinMann2002;
        TF_concentration = tf * 1e-12,
        tspan = (0.0, 1200.0), saveat = 1.0, p = P_PERTURBED)
    feat_true = extract_tga_features(HockinMann2002, sol_true)

    # Ensemble features (fitted with nominal model)
    feat_vals = Dict(fn => Float64[] for fn in feature_names)
    for idx in ens_idx_valid
        p_full = build_params_nominal(PC[:, idx])
        try
            sol = simulate(HockinMann2002;
                TF_concentration = tf * 1e-12,
                tspan = (0.0, 1200.0), saveat = 1.0, p = p_full)
            feat = extract_tga_features(HockinMann2002, sol)
            for fn in feature_names
                push!(feat_vals[fn], getfield(feat, fn))
            end
        catch; end
    end

    println("  $(tf) pM TF:")
    for (fn, fl, fs) in zip(feature_names, feature_labels, feature_scale)
        vals = feat_vals[fn] .* fs
        true_val = getfield(feat_true, fn) * fs
        μ = mean(vals)
        lo = quantile(vals, 0.025)
        hi = quantile(vals, 0.975)
        covered = lo <= true_val <= hi
        err_pct = abs(μ - true_val) / (abs(true_val) + 1e-30) * 100
        println("    $fl: true=$(round(true_val, sigdigits=3)), " *
                "pred=$(round(μ, sigdigits=3)) [$(round(lo, sigdigits=3)), $(round(hi, sigdigits=3))], " *
                "err=$(round(err_pct, digits=1))%, covered=$covered")
    end
end

# ──────────────────────────────────────────────────────────────
# Cache simulation results
# ──────────────────────────────────────────────────────────────
println("\nSaving simulation results to cache...")
@save CACHE_FILE EC PC RA train_times train_data train_true_perturbed Thr_train Thr_valid valid_times valid_true ens_idx_valid valid_mask
println("  Saved to $CACHE_FILE")

# ──────────────────────────────────────────────────────────────
# Figure: Misspecification comparison (2×2)
#   (a) Training fits (misspecified model)
#   (b) Held-out predictions (misspecified model)
#   (c) TGA feature accuracy comparison
#   (d) Parameter recovery (misspecified model)
# ──────────────────────────────────────────────────────────────
include("paper_theme.jl")
set_paper_theme!()

println("\nGenerating misspecification figure...")

# Colors for TF conditions (same as main study)
const C_TF_TRAIN = [
    (RGBAf(0.90, 0.45, 0.10, 0.25), RGBf(0.85, 0.35, 0.05), RGBf(0.70, 0.25, 0.00)),  # 5 pM
    (RGBAf(0.55, 0.25, 0.70, 0.25), RGBf(0.45, 0.20, 0.60), RGBf(0.35, 0.12, 0.50)),  # 15 pM
    (RGBAf(0.15, 0.65, 0.45, 0.25), RGBf(0.10, 0.50, 0.35), RGBf(0.05, 0.40, 0.25)),  # 25 pM
]
const C_TF_VALID = [
    (RGBAf(0.20, 0.45, 0.78, 0.20), RGBf(0.20, 0.45, 0.78), RGBf(0.10, 0.30, 0.60)),  # 10 pM
    (RGBAf(0.75, 0.40, 0.55, 0.20), RGBf(0.65, 0.30, 0.45), RGBf(0.50, 0.20, 0.35)),  # 20 pM
    (RGBAf(0.50, 0.70, 0.20, 0.20), RGBf(0.40, 0.60, 0.15), RGBf(0.30, 0.45, 0.10)),  # 30 pM
]
const TF_TRAIN_LABELS = ["5 pM TF", "15 pM TF", "25 pM TF"]
const TF_VALID_LABELS = ["10 pM TF", "20 pM TF", "30 pM TF"]

let
    fig = Figure(size = (1100, 820))
    sparse = 1:20:n_t

    # --- (a) Training fits ---
    ax_a = Axis(fig[1, 1],
        xlabel = "Time (s)", ylabel = "Total thrombin (nM)",
        title = "(a)  Training fits (misspecified model)")
    for i in 1:3
        c_fill, c_mean, c_data = C_TF_TRAIN[i]
        μ, lo, hi = ensemble_stats_nM(Thr_train[i])
        band!(ax_a, train_times[i], lo, hi, color = c_fill, label = "95% CI ($(TF_TRAIN_LABELS[i]))")
        lines!(ax_a, train_times[i], μ, color = c_mean, linewidth = 2, linestyle = :dash, label = "Mean ($(TF_TRAIN_LABELS[i]))")
        scatter!(ax_a, train_times[i][sparse], train_data[i][sparse] .* 1e9,
            color = c_data, markersize = 10, label = "Data ($(TF_TRAIN_LABELS[i]))")
    end
    axislegend(ax_a, position = :rt, framevisible = false, labelsize = 15)

    # --- (b) Held-out predictions ---
    ax_b = Axis(fig[1, 2],
        xlabel = "Time (s)", ylabel = "Total thrombin (nM)",
        title = "(b)  Held-out predictions (10, 20, 30 pM TF)")
    for i in 1:3
        c_fill, c_mean, c_true = C_TF_VALID[i]
        μ, lo, hi = ensemble_stats_nM(Thr_valid[i])
        band!(ax_b, valid_times[i], lo, hi, color = c_fill, label = "95% CI ($(TF_VALID_LABELS[i]))")
        lines!(ax_b, valid_times[i], μ, color = c_mean, linewidth = 2, linestyle = :dash, label = "Mean ($(TF_VALID_LABELS[i]))")
        lines!(ax_b, valid_times[i], valid_true[i] .* 1e9,
            color = c_true, linewidth = 2, label = "True ($(TF_VALID_LABELS[i]))")
    end
    axislegend(ax_b, position = :rt, framevisible = false, labelsize = 15)

    # --- (c) TGA feature accuracy ---
    ax_c = Axis(fig[2, 1],
        ylabel = "Predicted / True",
        title = "(c)  TGA feature accuracy (misspecified)")

    display_features = [:lagtime, :peak, :etp]
    display_labels = ["Lag time", "Peak IIa", "ETP"]
    display_scale_c = [1.0, 1e9, 1e9]

    x_pos = 0
    xtick_pos = Float64[]
    xtick_labels_c = String[]

    for (fi, (fn, fl, fs)) in enumerate(zip(display_features, display_labels, display_scale_c))
        for (ti, tf) in enumerate(VALID_TF)
            x_pos += 1
            sol_true = simulate(HockinMann2002;
                TF_concentration = tf * 1e-12,
                tspan = (0.0, 1200.0), saveat = 1.0, p = P_PERTURBED)
            feat_true = extract_tga_features(HockinMann2002, sol_true)
            true_val = getfield(feat_true, fn) * fs

            vals = Float64[]
            for idx in ens_idx_valid
                p_full = build_params_nominal(PC[:, idx])
                try
                    sol = simulate(HockinMann2002;
                        TF_concentration = tf * 1e-12,
                        tspan = (0.0, 1200.0), saveat = 1.0, p = p_full)
                    feat = extract_tga_features(HockinMann2002, sol)
                    push!(vals, getfield(feat, fn) * fs)
                catch; end
            end

            vals_norm = vals ./ true_val
            μ = mean(vals_norm)
            lo_q = quantile(vals_norm, 0.025)
            hi_q = quantile(vals_norm, 0.975)
            covered = lo_q <= 1.0 <= hi_q

            _, c_mean, _ = C_TF_VALID[ti]
            lines!(ax_c, [x_pos, x_pos], [lo_q, hi_q], color = c_mean, linewidth = 3)
            scatter!(ax_c, [x_pos], [μ], color = c_mean, markersize = 13)
            mkr_color = covered ? :black : C_THEORY
            scatter!(ax_c, [x_pos], [1.0], color = mkr_color, markersize = 11, marker = :xcross)

            push!(xtick_pos, x_pos)
            push!(xtick_labels_c, "$(Int(tf))")
        end
        x_pos += 0.5
    end

    hlines!(ax_c, [1.0], color = :gray50, linewidth = 0.8, linestyle = :dash)
    ax_c.xticks = (xtick_pos, xtick_labels_c)
    for sep in [3.75, 7.25]
        vlines!(ax_c, [sep], color = :gray70, linewidth = 0.6, linestyle = :dot)
    end
    for (fi, fl) in enumerate(display_labels)
        mid_x = mean(xtick_pos[(fi-1)*3+1 : fi*3])
        text!(ax_c, mid_x, 1.0; text = fl,
            align = (:center, :bottom), fontsize = 11, offset = (0, 150))
    end
    cur_ylims = ax_c.finallimits[].origin[2], ax_c.finallimits[].origin[2] + ax_c.finallimits[].widths[2]
    ylims!(ax_c, cur_ylims[1], cur_ylims[2] + 0.15 * (cur_ylims[2] - cur_ylims[1]))

    # --- (d) Parameter recovery ---
    ax_d = Axis(fig[2, 2],
        xlabel = "True value (log₁₀)", ylabel = "Estimated value (log₁₀)",
        title = "(d)  Parameter recovery (misspecified)")
    for k in 1:min(n_valid, 200)
        scatter!(ax_d, TRUE_LOG, PC[:, ens_idx_valid[k]],
            color = (C_PE, 0.08), markersize = 5)
    end
    lims = (minimum(LOG_LOWER) - 0.5, maximum(LOG_UPPER) + 0.5)
    lines!(ax_d, [lims[1], lims[2]], [lims[1], lims[2]],
        color = C_THEORY, linewidth = 1.5, linestyle = :dash, label = "Identity line")
    median_est = vec(median(PC[:, ens_idx_valid], dims=2))
    scatter!(ax_d, TRUE_LOG, median_est, color = C_DATA, markersize = 11, marker = :diamond, label = "Median estimate")
    axislegend(ax_d, position = :rb, framevisible = false, labelsize = 15)

    save(joinpath(FIGDIR, "fig_misspecification.pdf"), fig)
    println("  Saved fig_misspecification.pdf")
end

println("\nDone!")
