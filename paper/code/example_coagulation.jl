# Coagulation cascade ensemble estimation using HockinMann2002 model
# 34 species, 42 rate constants (10 estimated), 3 objectives
#
# Training: TGA curves at 5, 15, 25 pM TF (15% multiplicative noise)
# Validation: TGA scalar features at 10, 20, 30 pM TF
#
# Run from the paper/code directory:
#   julia --project -t4 example_coagulation.jl

using ParetoEnsembles
using HockinMannModel
using CairoMakie
using Random
using Statistics
using JLD2

const FIGDIR = joinpath(@__DIR__, "..", "figures")
const CACHEDIR = joinpath(@__DIR__, "data")
const CACHE_FILE = joinpath(CACHEDIR, "coagulation_results.jld2")

# ──────────────────────────────────────────────────────────────
# Model setup: HockinMann2002 (34 species, 42 rate constants)
# We estimate 10 key catalytic / inhibition rate constants.
# ──────────────────────────────────────────────────────────────

const P_TRUE = default_rate_constants(HockinMann2002)

# Indices and names of the 10 parameters we estimate
const ESTIMATE_INDICES = [10, 15, 16, 17, 22, 26, 31, 32, 38, 41]
const ESTIMATE_NAMES = [
    "extrinsic Xase kcat",      # p[10] = 6.0
    "TF=VIIa→IX kcat",          # p[15] = 1.8
    "Xa→IIa k",                 # p[16] = 7.5e3
    "IIa→VIIIa k",              # p[17] = 2.0e7
    "intrinsic Xase kcat",      # p[22] = 8.2
    "IIa→Va k",                 # p[26] = 2.0e7
    "prothrombinase kcat",      # p[31] = 63.5
    "mIIa→IIa k",               # p[32] = 1.5e7
    "Xa+ATIII k",                # p[38] = 1.5e3
    "IIa+ATIII k",               # p[41] = 7.1e3
]
const TRUE_LOG = log10.(P_TRUE[ESTIMATE_INDICES])
const N_EST = length(ESTIMATE_INDICES)

# Bounds: ±1.5 orders of magnitude from true values (in log-space)
const LOG_LOWER = TRUE_LOG .- 1.5
const LOG_UPPER = TRUE_LOG .+ 1.5

println("Estimating $(N_EST) rate constants from HockinMann2002 model")
println("True values (log10): ", round.(TRUE_LOG, digits=2))

# ──────────────────────────────────────────────────────────────
# Generate "experimental" data at three TF concentrations
# Training: 5, 15, 25 pM TF with 15% multiplicative noise
# ──────────────────────────────────────────────────────────────
function simulate_thrombin(p_full; TF_pM, saveat=10.0)
    sol = simulate(HockinMann2002;
        TF_concentration = TF_pM * 1e-12,
        tspan = (0.0, 1200.0),
        saveat = saveat,
        p = p_full)
    return sol.t, total_thrombin(HockinMann2002, sol)
end

const NOISE_CV = 0.15  # 15% multiplicative noise
const TRAIN_TF = [5.0, 15.0, 25.0]

Random.seed!(2024)

# Generate true trajectories and noisy training data
train_times = Vector{Vector{Float64}}(undef, 3)
train_true  = Vector{Vector{Float64}}(undef, 3)
train_data  = Vector{Vector{Float64}}(undef, 3)

for (i, tf) in enumerate(TRAIN_TF)
    t, thr = simulate_thrombin(P_TRUE; TF_pM=tf)
    train_times[i] = t
    train_true[i]  = thr
    noisy = thr .* (1.0 .+ NOISE_CV .* randn(length(thr)))
    train_data[i]  = max.(noisy, 0.0)
    println("Training data ($(tf) pM TF): $(length(t)) points, peak = $(round(maximum(thr)*1e9, digits=1)) nM")
end

# ──────────────────────────────────────────────────────────────
# ParetoEnsembles callbacks — 3 objectives (no regularization)
# ──────────────────────────────────────────────────────────────
function objective_function(x_log)
    f = zeros(3, 1)

    # Map from log-space to actual rate constants
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

function neighbor_function(x_log)
    new_x = x_log .+ 0.05 .* randn(N_EST)
    return clamp.(new_x, LOG_LOWER, LOG_UPPER)
end

acceptance_probability_function(R, T) = exp(-R[end] / T)
cooling_function(T) = 0.92 * T

# ──────────────────────────────────────────────────────────────
# Run ensemble estimation (8 parallel chains)
# ──────────────────────────────────────────────────────────────
println("\nRunning ensemble estimation (8 chains)...")
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
println("  Total solutions: $n_total")
println("  Pareto-optimal: $n_pareto")

# ──────────────────────────────────────────────────────────────
# Select ensemble (rank ≤ 1) and simulate training conditions
# ──────────────────────────────────────────────────────────────
ensemble_idx = findall(RA .<= 1)
n_ens = length(ensemble_idx)
println("  Ensemble size (rank ≤ 1): $n_ens")

# Helper: build full parameter vector from log-space estimates
function build_params(x_log)
    p_full = copy(P_TRUE)
    x = 10.0 .^ x_log
    for (i, pidx) in enumerate(ESTIMATE_INDICES)
        p_full[pidx] = x[i]
    end
    return p_full
end

# Simulate ensemble at training conditions
n_t = length(train_times[1])
Thr_train = [zeros(n_ens, n_t) for _ in 1:3]

for (k, idx) in enumerate(ensemble_idx)
    p_full = build_params(PC[:, idx])
    for (i, tf) in enumerate(TRAIN_TF)
        try
            _, thr = simulate_thrombin(p_full; TF_pM=tf)
            Thr_train[i][k, :] = thr
        catch
            Thr_train[i][k, :] .= NaN
        end
    end
end

# Remove failed simulations (any training condition failed)
valid = .!any(hcat([any(isnan.(M), dims=2)[:] for M in Thr_train]...), dims=2)[:]
for i in 1:3
    Thr_train[i] = Thr_train[i][valid, :]
end
ens_idx_valid = ensemble_idx[valid]
n_valid = sum(valid)
println("  Valid ensemble members: $n_valid / $n_ens")

# Compute statistics (nM) for each training condition
function ensemble_stats_nM(M)
    μ  = vec(mean(M, dims=1)) .* 1e9
    lo = vec(mapslices(x -> quantile(x, 0.025), M, dims=1)) .* 1e9
    hi = vec(mapslices(x -> quantile(x, 0.975), M, dims=1)) .* 1e9
    return μ, lo, hi
end

train_stats = [ensemble_stats_nM(M) for M in Thr_train]

# ──────────────────────────────────────────────────────────────
# Figure: Coagulation training results (3-panel)
#   (a) TGA fits at 5, 15, 25 pM TF
#   (b) Parameter recovery
#   (c) Pareto front
# ──────────────────────────────────────────────────────────────
include("paper_theme.jl")
set_paper_theme!()

println("\nGenerating coagulation figure...")

# Colors for three TF conditions
const C_TF = [
    (RGBAf(0.90, 0.45, 0.10, 0.25), RGBf(0.85, 0.35, 0.05), RGBf(0.70, 0.25, 0.00)),  # 5 pM: amber
    (RGBAf(0.55, 0.25, 0.70, 0.25), RGBf(0.45, 0.20, 0.60), RGBf(0.35, 0.12, 0.50)),  # 15 pM: purple
    (RGBAf(0.15, 0.65, 0.45, 0.25), RGBf(0.10, 0.50, 0.35), RGBf(0.05, 0.40, 0.25)),  # 25 pM: teal
]
const TF_LABELS = ["5 pM TF", "15 pM TF", "25 pM TF"]

let
    fig = Figure(size = (1100, 820))
    sparse = 1:20:n_t

    # --- (a) Thrombin at 5, 15, 25 pM TF ---
    ax_a = Axis(fig[1, 1:2],
        xlabel = "Time (s)", ylabel = "Total thrombin (nM)",
        title = "(a)  Training data: TGA at 5, 15, and 25 pM TF")
    for i in 1:3
        c_fill, c_mean, c_data = C_TF[i]
        μ, lo, hi = train_stats[i]
        band!(ax_a, train_times[i], lo, hi, color = c_fill)
        lines!(ax_a, train_times[i], μ, color = c_mean, linewidth = 2, linestyle = :dash)
        scatter!(ax_a, train_times[i][sparse], train_data[i][sparse] .* 1e9,
            color = c_data, markersize = 7, label = TF_LABELS[i])
    end
    axislegend(ax_a, position = :rt, framevisible = false, labelsize = 11)

    # --- (b) Parameter recovery (log-log) ---
    ax_b = Axis(fig[2, 1],
        xlabel = "True value (log₁₀)", ylabel = "Estimated value (log₁₀)",
        title = "(b)  Parameter recovery")
    for k in 1:min(n_valid, 200)
        scatter!(ax_b, TRUE_LOG, PC[:, ens_idx_valid[k]],
            color = (C_PE, 0.12), markersize = 3)
    end
    lims = (minimum(LOG_LOWER) - 0.5, maximum(LOG_UPPER) + 0.5)
    lines!(ax_b, [lims[1], lims[2]], [lims[1], lims[2]],
        color = C_THEORY, linewidth = 1.5, linestyle = :dash, label = "Identity line")
    median_est = vec(median(PC[:, ens_idx_valid], dims=2))
    scatter!(ax_b, TRUE_LOG, median_est, color = C_DATA, markersize = 8, marker = :diamond, label = "Median estimate")
    axislegend(ax_b, position = :rb, framevisible = false, labelsize = 11)

    # --- (c) Pareto front projection (log scale) ---
    ax_c = Axis(fig[2, 2],
        xlabel = "log₁₀(ε₁, 5 pM TF)",
        ylabel = "log₁₀(ε₂, 15 pM TF)",
        title = "(c)  Pareto front")

    log_e1 = log10.(EC[1, :] .+ 1e-10)
    log_e2 = log10.(EC[2, :] .+ 1e-10)
    # Color by ε₃ (25 pM TF)
    e3_vals = EC[3, :]
    e3_norm = clamp.((e3_vals .- minimum(e3_vals)) ./ (maximum(e3_vals) - minimum(e3_vals) + 1e-10), 0, 1)
    colors = [RGBAf(0.20 + 0.40*t, 0.45 + 0.25*t, 0.78 - 0.25*t, 0.55) for t in e3_norm]

    order = sortperm(RA, rev=true)
    scatter!(ax_c, log_e1[order], log_e2[order], color = colors[order], markersize = 4, label = "Near-optimal")
    p_idx = findall(RA .== 0)
    scatter!(ax_c, log_e1[p_idx], log_e2[p_idx], color = C_FRONT, markersize = 6, label = "Pareto front")
    axislegend(ax_c, position = :rt, framevisible = false, labelsize = 11)

    save(joinpath(FIGDIR, "fig_coagulation.pdf"), fig)
    println("  Saved fig_coagulation.pdf")
end

# Print parameter recovery summary
println("\nParameter recovery summary:")
median_est = vec(median(PC[:, ens_idx_valid], dims=2))
for i in 1:N_EST
    err_pct = abs(10^median_est[i] - P_TRUE[ESTIMATE_INDICES[i]]) / P_TRUE[ESTIMATE_INDICES[i]] * 100
    println("  $(ESTIMATE_NAMES[i]): true=$(round(P_TRUE[ESTIMATE_INDICES[i]], sigdigits=3)), " *
            "est=$(round(10^median_est[i], sigdigits=3)), err=$(round(err_pct, digits=1))%")
end

# ══════════════════════════════════════════════════════════════
# VALIDATION: Thrombin profiles at held-out TF concentrations
# Trained on TGA curves at 5, 15, 25 pM — validate on full
# thrombin time courses at 10, 20, 30 pM TF
# ══════════════════════════════════════════════════════════════
const VALID_TF = [10.0, 20.0, 30.0]

println("\n=== Validation: Thrombin profiles at held-out TF concentrations ===")

# True thrombin trajectories at validation conditions
valid_times = Vector{Vector{Float64}}(undef, 3)
valid_true  = Vector{Vector{Float64}}(undef, 3)
for (i, tf) in enumerate(VALID_TF)
    t, thr = simulate_thrombin(P_TRUE; TF_pM=tf)
    valid_times[i] = t
    valid_true[i]  = thr
    println("  True peak at $(tf) pM TF: $(round(maximum(thr)*1e9, digits=1)) nM")
end

# Ensemble thrombin trajectories at validation conditions
Thr_valid = [zeros(n_valid, length(valid_times[1])) for _ in 1:3]
for (k, idx) in enumerate(ens_idx_valid)
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

# Remove any failed simulations
valid_mask = .!any(hcat([any(isnan.(M), dims=2)[:] for M in Thr_valid]...), dims=2)[:]
for i in 1:3
    Thr_valid[i] = Thr_valid[i][valid_mask, :]
end
n_valid_pred = sum(valid_mask)
println("  Valid ensemble predictions: $n_valid_pred / $n_valid")

valid_stats = [ensemble_stats_nM(M) for M in Thr_valid]

# Print validation summary
println("\nValidation summary (ensemble predictions vs truth):")
for (i, tf) in enumerate(VALID_TF)
    μ, lo, hi = valid_stats[i]
    true_peak = maximum(valid_true[i]) * 1e9
    ens_peak = maximum(μ)
    err_pct = abs(ens_peak - true_peak) / true_peak * 100
    # Check if true trajectory falls within 95% CI at all time points
    true_nM = valid_true[i] .* 1e9
    covered = all(lo .<= true_nM .<= hi)
    println("  $(tf) pM TF: true peak=$(round(true_peak, digits=1)) nM, " *
            "pred peak=$(round(ens_peak, digits=1)) nM, err=$(round(err_pct, digits=1))%, " *
            "trajectory covered=$(covered)")
end

# Also compute TGA feature predictions for quantitative summary
println("\nTGA feature predictions at held-out conditions:")
feature_names = [:lagtime, :peak, :tpeak, :max_rate, :etp]
feature_labels = ["Lag time (s)", "Peak (nM)", "Time to peak (s)", "Max rate (nM/s)", "ETP (nM·s)"]
feature_scale = [1.0, 1e9, 1.0, 1e9, 1e9]

for tf in VALID_TF
    # True features
    sol_true = simulate(HockinMann2002;
        TF_concentration = tf * 1e-12,
        tspan = (0.0, 1200.0), saveat = 1.0, p = P_TRUE)
    feat_true = extract_tga_features(HockinMann2002, sol_true)

    # Ensemble features
    feat_vals = Dict(fn => Float64[] for fn in feature_names)
    for idx in ens_idx_valid[valid_mask]
        p_full = build_params(PC[:, idx])
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
        println("    $fl: true=$(round(true_val, sigdigits=3)), " *
                "pred=$(round(μ, sigdigits=3)) [$(round(lo, sigdigits=3)), $(round(hi, sigdigits=3))], " *
                "covered=$covered")
    end
end

# ══════════════════════════════════════════════════════════════
# ANALYSIS 2: Parameter identifiability (pairwise correlations)
# ══════════════════════════════════════════════════════════════
println("\n=== Analysis: Parameter correlations ===")
ens_params = PC[:, ens_idx_valid]  # N_EST x n_valid
cor_matrix = zeros(N_EST, N_EST)
for i in 1:N_EST, j in 1:N_EST
    cor_matrix[i, j] = cor(ens_params[i, :], ens_params[j, :])
end
println("  Strongest correlations (|r| > 0.3):")
for i in 1:N_EST, j in (i+1):N_EST
    r = cor_matrix[i, j]
    if abs(r) > 0.3
        println("    $(ESTIMATE_NAMES[i]) vs $(ESTIMATE_NAMES[j]): r = $(round(r, digits=3))")
    end
end

# ══════════════════════════════════════════════════════════════
# ANALYSIS 3: Patient-specific — Factor VIII deficiency (hemophilia A)
# ══════════════════════════════════════════════════════════════
println("\n=== Analysis: Factor VIII deficiency (hemophilia A) ===")

function simulate_patient(p_full; TF_pM, VIII_pct, saveat=10.0)
    VIII_M = 7.0e-10 * VIII_pct / 100.0  # nominal = 7e-10 M
    u0 = patient_initial_conditions(HockinMann2002;
        TF=TF_pM*1e-12, VIII=VIII_M)
    sol = simulate(HockinMann2002;
        TF_concentration=TF_pM*1e-12,
        tspan=(0.0, 1200.0), saveat=saveat,
        p=p_full, u0=u0)
    return sol.t, total_thrombin(HockinMann2002, sol)
end

# Simulate all hemophilia conditions
Thr_normal = zeros(n_valid, n_t)
Thr_mild   = zeros(n_valid, n_t)
Thr_severe = zeros(n_valid, n_t)
for (k, idx) in enumerate(ens_idx_valid)
    p_full = build_params(PC[:, idx])
    try
        _, thr_n = simulate_patient(p_full; TF_pM=5.0, VIII_pct=100.0)
        _, thr_m = simulate_patient(p_full; TF_pM=5.0, VIII_pct=30.0)
        _, thr_s = simulate_patient(p_full; TF_pM=5.0, VIII_pct=5.0)
        Thr_normal[k, :] = thr_n
        Thr_mild[k, :]   = thr_m
        Thr_severe[k, :] = thr_s
    catch
        Thr_normal[k, :] .= NaN
        Thr_mild[k, :]   .= NaN
        Thr_severe[k, :] .= NaN
    end
end

# Print hemophilia summary
for (label, viii_pct, M) in [("Normal (100%)", 100.0, Thr_normal),
                              ("Mild (30%)", 30.0, Thr_mild),
                              ("Severe (5%)", 5.0, Thr_severe)]
    v = .!any(isnan.(M), dims=2)[:]
    peaks = maximum(M[v, :], dims=2) .* 1e9
    println("  $label: peak = $(round(mean(peaks), digits=1)) ± $(round(std(peaks), digits=1)) nM (n=$(sum(v)))")
end

# True trajectories for reference
_, true_normal = simulate_patient(P_TRUE; TF_pM=5.0, VIII_pct=100.0)
_, true_mild   = simulate_patient(P_TRUE; TF_pM=5.0, VIII_pct=30.0)
_, true_severe = simulate_patient(P_TRUE; TF_pM=5.0, VIII_pct=5.0)

# ══════════════════════════════════════════════════════════════
# Cache all simulation results so figures can be regenerated
# without re-running expensive simulations
# ══════════════════════════════════════════════════════════════
println("\nSaving simulation results to cache...")
@save CACHE_FILE EC PC RA train_times train_true train_data Thr_train ens_idx_valid valid_mask Thr_valid valid_times valid_true cor_matrix Thr_normal Thr_mild Thr_severe true_normal true_mild true_severe
println("  Saved to $CACHE_FILE")

# ══════════════════════════════════════════════════════════════
# Figure 2: Ensemble insights (4-panel, 2×2)
#   (a) Held-out thrombin profiles at 10, 20, 30 pM TF
#   (b) Parameter correlation heatmap
#   (c) Hemophilia A patient predictions
#   (d) Parameter recovery with uncertainty
# ══════════════════════════════════════════════════════════════
println("\nGenerating ensemble insights figure...")

# Colors for validation TF conditions (distinct from training colors)
const C_VALID_TF = [
    (RGBAf(0.20, 0.45, 0.78, 0.20), RGBf(0.20, 0.45, 0.78), RGBf(0.10, 0.30, 0.60)),  # 10 pM: blue
    (RGBAf(0.75, 0.40, 0.55, 0.20), RGBf(0.65, 0.30, 0.45), RGBf(0.50, 0.20, 0.35)),  # 20 pM: rose
    (RGBAf(0.50, 0.70, 0.20, 0.20), RGBf(0.40, 0.60, 0.15), RGBf(0.30, 0.45, 0.10)),  # 30 pM: olive
]
const VALID_LABELS = ["10 pM TF", "20 pM TF", "30 pM TF"]

let
    fig = Figure(size = (1100, 820))

    # --- (a) Held-out thrombin profiles ---
    ax_a = Axis(fig[1, 1:2],
        xlabel = "Time (s)", ylabel = "Total thrombin (nM)",
        title = "(a)  Held-out predictions (10, 20, 30 pM TF)")

    for i in 1:3
        c_fill, c_mean, c_true = C_VALID_TF[i]
        μ, lo, hi = valid_stats[i]
        # Ensemble CI band and mean
        band!(ax_a, valid_times[i], lo, hi, color = c_fill, label = VALID_LABELS[i])
        lines!(ax_a, valid_times[i], μ, color = c_mean, linewidth = 2, linestyle = :dash)
        # True trajectory
        lines!(ax_a, valid_times[i], valid_true[i] .* 1e9,
            color = c_true, linewidth = 2)
    end
    axislegend(ax_a, position = :rt, framevisible = false, labelsize = 11)

    # --- (b) Parameter correlations ---
    ax_b = Axis(fig[1, 3],
        title = "(b)  Parameter correlations",
        xticks = (1:N_EST, ["p$(ESTIMATE_INDICES[i])" for i in 1:N_EST]),
        yticks = (1:N_EST, ["p$(ESTIMATE_INDICES[i])" for i in 1:N_EST]),
        xticklabelrotation = π/4,
        yreversed = true,
        backgroundcolor = :white)

    hm = heatmap!(ax_b, 1:N_EST, 1:N_EST, cor_matrix,
        colormap = :RdBu, colorrange = (-1, 1))
    Colorbar(fig[1, 4], hm, label = "r", width = 12)

    # --- (c) Patient predictions ---
    ax_c = Axis(fig[2, 1:2],
        xlabel = "Time (s)", ylabel = "Total thrombin (nM)",
        title = "(c)  Factor VIII deficiency (hemophilia A)")

    for (label, M, true_traj, col) in [
            ("100% FVIII", Thr_normal, true_normal, C_NORMAL),
            ("30% FVIII",  Thr_mild,   true_mild,   C_MILD),
            ("5% FVIII",   Thr_severe, true_severe,  C_SEVERE)]
        v = .!any(isnan.(M), dims=2)[:]
        μ, lo, hi = ensemble_stats_nM(M[v, :])
        band!(ax_c, train_times[1], lo, hi, color = (col, 0.15))
        lines!(ax_c, train_times[1], μ, color = col, linewidth = 2, label = label)
        lines!(ax_c, train_times[1], true_traj .* 1e9,
            color = C_TRUE, linewidth = 1, linestyle = :dash)
    end
    # Add true trajectory to legend
    lines!(ax_c, [NaN], [NaN], color = C_TRUE, linewidth = 1, linestyle = :dash, label = "True")
    axislegend(ax_c, position = :rt, framevisible = false, labelsize = 11)

    # --- (d) TGA feature predictions at held-out conditions ---
    # Dot-and-whisker: ensemble mean ± 95% CI normalized to true value
    ax_d = Axis(fig[2, 3],
        ylabel = "Predicted / True",
        title = "(d)  TGA feature accuracy")

    display_features = [:lagtime, :peak, :etp]
    display_labels = ["Lag time", "Peak IIa", "ETP"]
    display_scale = [1.0, 1e9, 1e9]

    x_pos = 0
    xtick_pos = Float64[]
    xtick_labels_d = String[]

    for (fi, (fn, fl, fs)) in enumerate(zip(display_features, display_labels, display_scale))
        for (ti, tf) in enumerate(VALID_TF)
            x_pos += 1

            # Recompute features for this condition
            sol_true = simulate(HockinMann2002;
                TF_concentration = tf * 1e-12,
                tspan = (0.0, 1200.0), saveat = 1.0, p = P_TRUE)
            feat_true = extract_tga_features(HockinMann2002, sol_true)
            true_val = getfield(feat_true, fn) * fs

            vals = Float64[]
            for idx in ens_idx_valid[valid_mask]
                p_full = build_params(PC[:, idx])
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
            lo = quantile(vals_norm, 0.025)
            hi = quantile(vals_norm, 0.975)
            covered = lo <= 1.0 <= hi

            # CI whisker
            c_fill, c_mean, _ = C_VALID_TF[ti]
            lines!(ax_d, [x_pos, x_pos], [lo, hi],
                color = c_mean, linewidth = 3)
            # Mean dot
            scatter!(ax_d, [x_pos], [μ],
                color = c_mean, markersize = 10)
            # True value marker
            mkr_color = covered ? :black : C_THEORY
            mkr_label = covered ? "Truth covered" : "Truth not covered"
            scatter!(ax_d, [x_pos], [1.0],
                color = mkr_color, markersize = 8, marker = :xcross, label = mkr_label)

            push!(xtick_pos, x_pos)
            push!(xtick_labels_d, "$(Int(tf))")
        end
        x_pos += 0.5  # gap between feature groups
    end

    hlines!(ax_d, [1.0], color = :gray50, linewidth = 0.8, linestyle = :dash)
    ax_d.xticks = (xtick_pos, xtick_labels_d)

    # Vertical separators between feature groups
    for sep in [3.75, 7.25]
        vlines!(ax_d, [sep], color = :gray70, linewidth = 0.6, linestyle = :dot)
    end

    # Add top padding so feature group labels don't overlap data
    ylims!(ax_d, nothing, nothing)
    cur_lo, cur_hi = ax_d.finallimits[].origin[2], ax_d.finallimits[].origin[2] + ax_d.finallimits[].widths[2]
    ylims!(ax_d, cur_lo, cur_hi + 0.15 * (cur_hi - cur_lo))

    # Feature group labels at top
    for (fi, fl) in enumerate(display_labels)
        mid_x = mean(xtick_pos[(fi-1)*3+1 : fi*3])
        text!(ax_d, mid_x, 1.0; text = fl,
            align = (:center, :bottom), fontsize = 11,
            offset = (0, 150))
    end

    axislegend(ax_d, position = :rb, framevisible = false, labelsize = 11, unique = true)

    save(joinpath(FIGDIR, "fig_ensemble_insights.pdf"), fig)
    println("  Saved fig_ensemble_insights.pdf")
end

println("\nDone!")
