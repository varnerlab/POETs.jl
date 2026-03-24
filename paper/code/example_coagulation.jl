# Coagulation cascade ensemble estimation using HockinMann2002 model
# 34 species, 42 rate constants (10 estimated), 3 objectives
#
# Demonstrates ParetoEnsembles.jl on a realistic-scale ODE model where
# the true parameters are known, allowing assessment of parameter recovery.
#
# Run from the paper/code directory:
#   julia --project -t4 example_coagulation.jl

using ParetoEnsembles
using HockinMannModel
using CairoMakie
using Random
using Statistics

const FIGDIR = joinpath(@__DIR__, "..", "figures")

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
# Generate "experimental" data at two TF concentrations
# ──────────────────────────────────────────────────────────────
function simulate_thrombin(p_full; TF_pM, saveat=10.0)
    sol = simulate(HockinMann2002;
        TF_concentration = TF_pM * 1e-12,
        tspan = (0.0, 1200.0),
        saveat = saveat,
        p = p_full)
    return sol.t, total_thrombin(HockinMann2002, sol)
end

Random.seed!(2024)
t_5pM, true_5pM = simulate_thrombin(P_TRUE; TF_pM=5.0)
t_25pM, true_25pM = simulate_thrombin(P_TRUE; TF_pM=25.0)

# Add 10% multiplicative noise
const NOISE_CV = 0.10
data_5pM = true_5pM .* (1.0 .+ NOISE_CV .* randn(length(true_5pM)))
data_5pM .= max.(data_5pM, 0.0)
data_25pM = true_25pM .* (1.0 .+ NOISE_CV .* randn(length(true_25pM)))
data_25pM .= max.(data_25pM, 0.0)

println("Data generated: $(length(t_5pM)) points at 5 pM TF, $(length(t_25pM)) points at 25 pM TF")
println("  Peak thrombin (5 pM): $(round(maximum(true_5pM)*1e9, digits=1)) nM")
println("  Peak thrombin (25 pM): $(round(maximum(true_25pM)*1e9, digits=1)) nM")

# ──────────────────────────────────────────────────────────────
# ParetoEnsembles callbacks — 3 objectives
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
        # Objective 1: normalized SSE at 5 pM TF
        _, thrombin1 = simulate_thrombin(p_full; TF_pM=5.0)
        norm1 = sum(data_5pM .^ 2) + 1e-30
        f[1] = sum((thrombin1 .- data_5pM) .^ 2) / norm1

        # Objective 2: normalized SSE at 25 pM TF
        _, thrombin2 = simulate_thrombin(p_full; TF_pM=25.0)
        norm2 = sum(data_25pM .^ 2) + 1e-30
        f[2] = sum((thrombin2 .- data_25pM) .^ 2) / norm2

        # Objective 3: regularization (log-distance from literature)
        f[3] = sum((x_log .- TRUE_LOG) .^ 2) / N_EST
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
# Select ensemble (rank ≤ 1) and simulate
# ──────────────────────────────────────────────────────────────
ensemble_idx = findall(RA .<= 1)
n_ens = length(ensemble_idx)
println("  Ensemble size (rank ≤ 1): $n_ens")

# Simulate ensemble members
Thr5_ens = zeros(n_ens, length(t_5pM))
Thr25_ens = zeros(n_ens, length(t_25pM))

for (k, idx) in enumerate(ensemble_idx)
    x_log = PC[:, idx]
    x = 10.0 .^ x_log
    p_full = copy(P_TRUE)
    for (i, pidx) in enumerate(ESTIMATE_INDICES)
        p_full[pidx] = x[i]
    end
    try
        _, thr5 = simulate_thrombin(p_full; TF_pM=5.0)
        _, thr25 = simulate_thrombin(p_full; TF_pM=25.0)
        Thr5_ens[k, :] = thr5
        Thr25_ens[k, :] = thr25
    catch
        Thr5_ens[k, :] .= NaN
        Thr25_ens[k, :] .= NaN
    end
end

# Remove failed simulations
valid = .!any(isnan.(Thr5_ens), dims=2)[:]
Thr5_ens = Thr5_ens[valid, :]
Thr25_ens = Thr25_ens[valid, :]
ens_idx_valid = ensemble_idx[valid]
n_valid = sum(valid)
println("  Valid ensemble members: $n_valid / $n_ens")

# Statistics (convert to nM for plotting)
thr5_mean = vec(mean(Thr5_ens, dims=1)) .* 1e9
thr5_lo = vec(mapslices(x -> quantile(x, 0.025), Thr5_ens, dims=1)) .* 1e9
thr5_hi = vec(mapslices(x -> quantile(x, 0.975), Thr5_ens, dims=1)) .* 1e9

thr25_mean = vec(mean(Thr25_ens, dims=1)) .* 1e9
thr25_lo = vec(mapslices(x -> quantile(x, 0.025), Thr25_ens, dims=1)) .* 1e9
thr25_hi = vec(mapslices(x -> quantile(x, 0.975), Thr25_ens, dims=1)) .* 1e9

# ──────────────────────────────────────────────────────────────
# Figure: 4-panel coagulation results
# ──────────────────────────────────────────────────────────────
println("\nGenerating coagulation figure...")
let
    fig = Figure(size = (1000, 800), fontsize = 13)

    # --- (a) Thrombin at 5 pM TF ---
    ax_a = Axis(fig[1, 1],
        xlabel = "Time (s)",
        ylabel = "Total thrombin (nM)",
        title = "(a)  5 pM TF")

    band!(ax_a, t_5pM, thr5_lo, thr5_hi, color = (:dodgerblue, 0.2))
    lines!(ax_a, t_5pM, thr5_mean, color = :dodgerblue, linewidth = 2, linestyle = :dash)
    # Plot data as sparse points (every 20th point to avoid clutter)
    sparse = 1:20:length(t_5pM)
    scatter!(ax_a, t_5pM[sparse], data_5pM[sparse] .* 1e9, color = :black, markersize = 4)

    # --- (b) Thrombin at 25 pM TF ---
    ax_b = Axis(fig[1, 2],
        xlabel = "Time (s)",
        ylabel = "Total thrombin (nM)",
        title = "(b)  25 pM TF")

    band!(ax_b, t_25pM, thr25_lo, thr25_hi, color = (:dodgerblue, 0.2))
    lines!(ax_b, t_25pM, thr25_mean, color = :dodgerblue, linewidth = 2, linestyle = :dash)
    scatter!(ax_b, t_25pM[sparse], data_25pM[sparse] .* 1e9, color = :black, markersize = 4)

    # --- (c) Parameter recovery (log-log) ---
    ax_c = Axis(fig[2, 1],
        xlabel = "True value (log10)",
        ylabel = "Estimated value (log10)",
        title = "(c)  Parameter recovery")

    # Plot all ensemble member estimates
    for k in 1:min(n_valid, 200)
        scatter!(ax_c, TRUE_LOG, PC[:, ens_idx_valid[k]],
            color = (:dodgerblue, 0.15), markersize = 3)
    end
    # Identity line
    lims = (minimum(LOG_LOWER) - 0.5, maximum(LOG_UPPER) + 0.5)
    lines!(ax_c, [lims[1], lims[2]], [lims[1], lims[2]],
        color = :red, linewidth = 1.5, linestyle = :dash)
    # Median estimates
    median_est = vec(median(PC[:, ens_idx_valid], dims=2))
    scatter!(ax_c, TRUE_LOG, median_est, color = :black, markersize = 8, marker = :diamond)

    # --- (d) Pareto front projection (ε₁ vs ε₂, colored by ε₃) ---
    ax_d = Axis(fig[2, 2],
        xlabel = "\u03b5 (5 pM TF)",
        ylabel = "\u03b5 (25 pM TF)",
        title = "(d)  Pareto front")

    # Color by regularization objective (ε₃)
    e3_vals = EC[3, :]
    e3_norm = clamp.((e3_vals .- minimum(e3_vals)) ./ (maximum(e3_vals) - minimum(e3_vals) + 1e-10), 0, 1)
    colors = [RGBAf(0.1 + 0.5*t, 0.3 + 0.3*t, 0.85 - 0.3*t, 0.6) for t in e3_norm]

    # Plot near-optimal first, then front on top
    order = sortperm(RA, rev=true)
    scatter!(ax_d, EC[1, order], EC[2, order],
        color = colors[order], markersize = 4)

    # Front points in black
    p_idx = findall(RA .== 0)
    scatter!(ax_d, EC[1, p_idx], EC[2, p_idx],
        color = :black, markersize = 5)

    # Shared legend
    elem_band = PolyElement(color = (:dodgerblue, 0.2))
    elem_mean = LineElement(color = :dodgerblue, linewidth = 2, linestyle = :dash)
    elem_data = MarkerElement(color = :black, marker = :circle, markersize = 4)
    elem_identity = LineElement(color = :red, linewidth = 1.5, linestyle = :dash)
    elem_median = MarkerElement(color = :black, marker = :diamond, markersize = 8)
    Legend(fig[3, 1:2],
        [elem_data, elem_mean, elem_band, elem_identity, elem_median],
        ["Noisy data", "Ensemble mean", "95% CI", "Identity line", "Median estimate"],
        orientation = :horizontal, framevisible = false,
        tellwidth = false, tellheight = true)

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
# ANALYSIS 1: Held-out prediction at 1 pM TF
# (trained on 5 pM and 25 pM only)
# ══════════════════════════════════════════════════════════════
println("\n=== Analysis 1: Held-out prediction at 1 pM TF ===")
t_1pM, true_1pM = simulate_thrombin(P_TRUE; TF_pM=1.0)

# Best single-fit: take the ensemble member with lowest total training error
total_train_err = EC[1, ens_idx_valid] .+ EC[2, ens_idx_valid]
best_idx = ens_idx_valid[argmin(total_train_err)]
best_log = PC[:, best_idx]
best_p = copy(P_TRUE)
for (i, pidx) in enumerate(ESTIMATE_INDICES)
    best_p[pidx] = 10.0^best_log[i]
end
_, best_1pM = simulate_thrombin(best_p; TF_pM=1.0)

# Ensemble prediction
Thr1_ens = zeros(n_valid, length(t_1pM))
for (k, idx) in enumerate(ens_idx_valid)
    x_log = PC[:, idx]
    x = 10.0 .^ x_log
    p_full = copy(P_TRUE)
    for (i, pidx) in enumerate(ESTIMATE_INDICES)
        p_full[pidx] = x[i]
    end
    try
        _, thr1 = simulate_thrombin(p_full; TF_pM=1.0)
        Thr1_ens[k, :] = thr1
    catch
        Thr1_ens[k, :] .= NaN
    end
end

valid1 = .!any(isnan.(Thr1_ens), dims=2)[:]
Thr1_valid = Thr1_ens[valid1, :]
thr1_mean = vec(mean(Thr1_valid, dims=1)) .* 1e9
thr1_lo = vec(mapslices(x -> quantile(x, 0.025), Thr1_valid, dims=1)) .* 1e9
thr1_hi = vec(mapslices(x -> quantile(x, 0.975), Thr1_valid, dims=1)) .* 1e9
println("  Ensemble members with valid 1 pM predictions: $(sum(valid1))")
println("  True peak (1 pM): $(round(maximum(true_1pM)*1e9, digits=1)) nM")
println("  Ensemble mean peak: $(round(maximum(thr1_mean), digits=1)) nM")
println("  Best single-fit peak: $(round(maximum(best_1pM)*1e9, digits=1)) nM")

# ══════════════════════════════════════════════════════════════
# ANALYSIS 2: Parameter identifiability (pairwise correlations)
# ══════════════════════════════════════════════════════════════
println("\n=== Analysis 2: Parameter correlations ===")
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
println("\n=== Analysis 3: Factor VIII deficiency (hemophilia A) ===")

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

# Normal plasma (100% FVIII) and severe hemophilia (5% FVIII)
for (label, viii_pct) in [("Normal (100% FVIII)", 100.0), ("Mild hemophilia (30% FVIII)", 30.0), ("Severe hemophilia (5% FVIII)", 5.0)]
    Thr_patient = zeros(n_valid, length(t_5pM))
    n_ok = 0
    for (k, idx) in enumerate(ens_idx_valid)
        x_log = PC[:, idx]
        x = 10.0 .^ x_log
        p_full = copy(P_TRUE)
        for (i, pidx) in enumerate(ESTIMATE_INDICES)
            p_full[pidx] = x[i]
        end
        try
            _, thr = simulate_patient(p_full; TF_pM=5.0, VIII_pct=viii_pct)
            Thr_patient[k, :] = thr
            n_ok += 1
        catch
            Thr_patient[k, :] .= NaN
        end
    end
    valid_p = .!any(isnan.(Thr_patient), dims=2)[:]
    peaks = maximum(Thr_patient[valid_p, :], dims=2) .* 1e9
    println("  $label: peak = $(round(mean(peaks), digits=1)) ± $(round(std(peaks), digits=1)) nM (n=$(sum(valid_p)))")
end

# Full ensemble simulation for the figure
Thr_normal = zeros(n_valid, length(t_5pM))
Thr_mild = zeros(n_valid, length(t_5pM))
Thr_severe = zeros(n_valid, length(t_5pM))
for (k, idx) in enumerate(ens_idx_valid)
    x_log = PC[:, idx]
    x = 10.0 .^ x_log
    p_full = copy(P_TRUE)
    for (i, pidx) in enumerate(ESTIMATE_INDICES)
        p_full[pidx] = x[i]
    end
    try
        _, thr_n = simulate_patient(p_full; TF_pM=5.0, VIII_pct=100.0)
        _, thr_m = simulate_patient(p_full; TF_pM=5.0, VIII_pct=30.0)
        _, thr_s = simulate_patient(p_full; TF_pM=5.0, VIII_pct=5.0)
        Thr_normal[k, :] = thr_n
        Thr_mild[k, :] = thr_m
        Thr_severe[k, :] = thr_s
    catch
        Thr_normal[k, :] .= NaN
        Thr_mild[k, :] .= NaN
        Thr_severe[k, :] .= NaN
    end
end

# True trajectories for reference
_, true_normal = simulate_patient(P_TRUE; TF_pM=5.0, VIII_pct=100.0)
_, true_mild = simulate_patient(P_TRUE; TF_pM=5.0, VIII_pct=30.0)
_, true_severe = simulate_patient(P_TRUE; TF_pM=5.0, VIII_pct=5.0)

# ══════════════════════════════════════════════════════════════
# Figure 2: Ensemble insights (3-panel)
#   (a) Held-out prediction at 1 pM TF
#   (b) Parameter correlation heatmap
#   (c) Hemophilia A patient predictions
# ══════════════════════════════════════════════════════════════
println("\nGenerating ensemble insights figure...")
let
    fig = Figure(size = (1100, 400), fontsize = 13)

    # --- (a) Held-out prediction ---
    ax_a = Axis(fig[1, 1],
        xlabel = "Time (s)",
        ylabel = "Total thrombin (nM)",
        title = "(a)  Held-out prediction (1 pM TF)")

    band!(ax_a, t_1pM, thr1_lo, thr1_hi, color = (:dodgerblue, 0.2))
    lines!(ax_a, t_1pM, thr1_mean, color = :dodgerblue, linewidth = 2, linestyle = :dash,
        label = "Ensemble mean")
    lines!(ax_a, t_1pM, true_1pM .* 1e9, color = :black, linewidth = 2,
        label = "True trajectory")
    lines!(ax_a, t_1pM, best_1pM .* 1e9, color = :red, linewidth = 1.5, linestyle = :dot,
        label = "Best single fit")
    axislegend(ax_a, position = :lt, framevisible = false, labelsize = 10)

    # --- (b) Correlation heatmap ---
    ax_b = Axis(fig[1, 2],
        title = "(b)  Parameter correlations",
        xticks = (1:N_EST, ["p$(ESTIMATE_INDICES[i])" for i in 1:N_EST]),
        yticks = (1:N_EST, ["p$(ESTIMATE_INDICES[i])" for i in 1:N_EST]),
        xticklabelrotation = π/4,
        yreversed = true)

    hm = heatmap!(ax_b, 1:N_EST, 1:N_EST, cor_matrix,
        colormap = :RdBu, colorrange = (-1, 1))
    Colorbar(fig[1, 2][1, 2], hm, label = "Correlation (r)")

    # --- (c) Patient predictions ---
    ax_c = Axis(fig[1, 3],
        xlabel = "Time (s)",
        ylabel = "Total thrombin (nM)",
        title = "(c)  Factor VIII deficiency")

    # Normal
    vn = .!any(isnan.(Thr_normal), dims=2)[:]
    n_mean = vec(mean(Thr_normal[vn,:], dims=1)) .* 1e9
    n_lo = vec(mapslices(x -> quantile(x, 0.025), Thr_normal[vn,:], dims=1)) .* 1e9
    n_hi = vec(mapslices(x -> quantile(x, 0.975), Thr_normal[vn,:], dims=1)) .* 1e9
    band!(ax_c, t_5pM, n_lo, n_hi, color = (:dodgerblue, 0.15))
    lines!(ax_c, t_5pM, n_mean, color = :dodgerblue, linewidth = 2, label = "100% FVIII")

    # Mild hemophilia
    vm = .!any(isnan.(Thr_mild), dims=2)[:]
    m_mean = vec(mean(Thr_mild[vm,:], dims=1)) .* 1e9
    m_lo = vec(mapslices(x -> quantile(x, 0.025), Thr_mild[vm,:], dims=1)) .* 1e9
    m_hi = vec(mapslices(x -> quantile(x, 0.975), Thr_mild[vm,:], dims=1)) .* 1e9
    band!(ax_c, t_5pM, m_lo, m_hi, color = (:orange, 0.15))
    lines!(ax_c, t_5pM, m_mean, color = :orange, linewidth = 2, label = "30% FVIII")

    # Severe hemophilia
    vs = .!any(isnan.(Thr_severe), dims=2)[:]
    s_mean = vec(mean(Thr_severe[vs,:], dims=1)) .* 1e9
    s_lo = vec(mapslices(x -> quantile(x, 0.025), Thr_severe[vs,:], dims=1)) .* 1e9
    s_hi = vec(mapslices(x -> quantile(x, 0.975), Thr_severe[vs,:], dims=1)) .* 1e9
    band!(ax_c, t_5pM, s_lo, s_hi, color = (:red, 0.15))
    lines!(ax_c, t_5pM, s_mean, color = :red, linewidth = 2, label = "5% FVIII")

    # True trajectories as thin dashed lines
    lines!(ax_c, t_5pM, true_normal .* 1e9, color = :black, linewidth = 1, linestyle = :dash)
    lines!(ax_c, t_5pM, true_mild .* 1e9, color = :black, linewidth = 1, linestyle = :dash)
    lines!(ax_c, t_5pM, true_severe .* 1e9, color = :black, linewidth = 1, linestyle = :dash)

    axislegend(ax_c, position = :rt, framevisible = false, labelsize = 10)

    save(joinpath(FIGDIR, "fig_ensemble_insights.pdf"), fig)
    println("  Saved fig_ensemble_insights.pdf")
end

println("\nDone!")
