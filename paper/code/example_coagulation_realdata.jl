# Real-data coagulation ensemble estimation
# Fits the Hockin-Mann model to experimental prothrombin titration data
# from Butenas et al. 1999, Blood 94(7):2169-2178, Figure 3
#
# Training:   FII at 50%, 100%, 150% of normal (3 objectives)
# Validation: FII at 75%, 125% of normal (held out)
# System:     Synthetic plasma, 5 pM TF, no protein C pathway
#
# Run from the paper/code directory:
#   julia --project -t4 example_coagulation_realdata.jl

using ParetoEnsembles
using HockinMannModel
using CairoMakie
using Random
using Statistics
using DelimitedFiles

const FIGDIR = joinpath(@__DIR__, "..", "figures")

# ──────────────────────────────────────────────────────────────
# Load experimental data (Butenas et al. 1999, Figure 3)
# ──────────────────────────────────────────────────────────────
println("Loading experimental data from Butenas et al. 1999...")

# Parse CSV manually (skip comment lines starting with #)
datafile = joinpath(@__DIR__, "data", "butenas1999_prothrombin.csv")
lines = filter(l -> !startswith(l, "#") && !isempty(strip(l)), readlines(datafile))
header = split(lines[1], ",")
data_rows = [split(l, ",") for l in lines[2:end]]

# Organize by condition
conditions = ["FII_50pct", "FII_75pct", "FII_100pct", "FII_125pct", "FII_150pct"]
condition_labels = ["50% FII", "75% FII", "100% FII", "125% FII", "150% FII"]
prothrombin_fractions = [0.50, 0.75, 1.00, 1.25, 1.50]

exp_data = Dict{String, NamedTuple{(:time_min, :thrombin_nM), Tuple{Vector{Float64}, Vector{Float64}}}}()
for cond in conditions
    rows = filter(r -> strip(r[3]) == cond, data_rows)
    t = [parse(Float64, r[1]) for r in rows]
    thr = [parse(Float64, r[2]) for r in rows]
    exp_data[cond] = (time_min=t, thrombin_nM=thr)
end

for (cond, label) in zip(conditions, condition_labels)
    d = exp_data[cond]
    println("  $label: $(length(d.time_min)) points, peak = $(maximum(d.thrombin_nM)) nM")
end

# Training conditions: 50%, 100%, 150%
# Validation conditions: 75%, 125%
const TRAIN_CONDITIONS = ["FII_50pct", "FII_100pct", "FII_150pct"]
const VALID_CONDITIONS = ["FII_75pct", "FII_125pct"]
const TRAIN_FRACTIONS = [0.50, 1.00, 1.50]
const VALID_FRACTIONS = [0.75, 1.25]
const TRAIN_LABELS = ["50% FII", "100% FII", "150% FII"]
const VALID_LABELS = ["75% FII", "125% FII"]

# ──────────────────────────────────────────────────────────────
# Model setup
# ──────────────────────────────────────────────────────────────
const P_NOMINAL = default_rate_constants(HockinMann2002)

# Same 10 estimated parameters as the synthetic study
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
const NOMINAL_LOG = log10.(P_NOMINAL[ESTIMATE_INDICES])
const N_EST = length(ESTIMATE_INDICES)

# Bounds: ±1.5 orders of magnitude from nominal literature values
const LOG_LOWER = NOMINAL_LOG .- 1.5
const LOG_UPPER = NOMINAL_LOG .+ 1.5

# ──────────────────────────────────────────────────────────────
# Simulation helper: vary prothrombin level
# ──────────────────────────────────────────────────────────────
function simulate_thrombin_FII(p_full; FII_fraction, saveat_min=1.0)
    # Prothrombin nominal = 1.4 µM = 1.4e-6 M (species index 14)
    u0 = default_initial_conditions(HockinMann2002; TF_concentration=5e-12)
    u0[14] = 1.4e-6 * FII_fraction  # scale prothrombin

    sol = simulate(HockinMann2002;
        TF_concentration = 5e-12,
        tspan = (0.0, 14.0 * 60.0),  # 14 minutes in seconds
        saveat = saveat_min * 60.0,    # convert to seconds
        p = p_full,
        u0 = u0)

    t_min = sol.t ./ 60.0  # convert back to minutes
    thr_nM = total_thrombin(HockinMann2002, sol) .* 1e9  # convert M to nM
    return t_min, thr_nM
end

# ──────────────────────────────────────────────────────────────
# ParetoEnsembles callbacks — 3 objectives (training conditions)
# ──────────────────────────────────────────────────────────────

# Precompute interpolated experimental data at simulation time points
const SIM_TIMES_MIN = collect(0.0:1.0:14.0)  # match experimental sampling

# Simple linear interpolation for experimental data
function interp_exp(cond)
    d = exp_data[cond]
    # Interpolate to SIM_TIMES_MIN
    result = zeros(length(SIM_TIMES_MIN))
    for (i, t) in enumerate(SIM_TIMES_MIN)
        # Find bracketing points
        idx = searchsortedlast(d.time_min, t)
        if idx == 0
            result[i] = d.thrombin_nM[1]
        elseif idx >= length(d.time_min)
            result[i] = d.thrombin_nM[end]
        elseif d.time_min[idx] == t
            result[i] = d.thrombin_nM[idx]
        else
            # Linear interpolation
            t1, t2 = d.time_min[idx], d.time_min[idx+1]
            y1, y2 = d.thrombin_nM[idx], d.thrombin_nM[idx+1]
            result[i] = y1 + (y2 - y1) * (t - t1) / (t2 - t1)
        end
    end
    return result
end

const TRAIN_DATA_INTERP = [interp_exp(cond) for cond in TRAIN_CONDITIONS]

function objective_function(x_log)
    f = zeros(3, 1)

    x = 10.0 .^ x_log
    p_full = copy(P_NOMINAL)
    for (i, idx) in enumerate(ESTIMATE_INDICES)
        p_full[idx] = x[i]
    end

    try
        for (obj_i, frac) in enumerate(TRAIN_FRACTIONS)
            t_sim, thr_sim = simulate_thrombin_FII(p_full; FII_fraction=frac)
            # SSE normalized by data magnitude
            norm = sum(TRAIN_DATA_INTERP[obj_i] .^ 2) + 1e-6
            f[obj_i] = sum((thr_sim .- TRAIN_DATA_INTERP[obj_i]) .^ 2) / norm
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
# Run ensemble estimation
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
println("  Total solutions: $n_total, Pareto-optimal: $n_pareto")

# ──────────────────────────────────────────────────────────────
# Select ensemble and simulate all conditions
# ──────────────────────────────────────────────────────────────
ensemble_idx = findall(RA .<= 1)
n_ens = length(ensemble_idx)
println("  Ensemble size (rank ≤ 1): $n_ens")

function build_params(x_log)
    p_full = copy(P_NOMINAL)
    x = 10.0 .^ x_log
    for (i, pidx) in enumerate(ESTIMATE_INDICES)
        p_full[pidx] = x[i]
    end
    return p_full
end

function ensemble_stats(M)
    μ  = vec(mean(M, dims=1))
    lo = vec(mapslices(x -> quantile(x, 0.025), M, dims=1))
    hi = vec(mapslices(x -> quantile(x, 0.975), M, dims=1))
    return μ, lo, hi
end

# Simulate at all 5 conditions
all_conditions = conditions
all_fractions = prothrombin_fractions
n_t = length(SIM_TIMES_MIN)

Thr_all = Dict{String, Matrix{Float64}}()
for cond in all_conditions
    Thr_all[cond] = zeros(n_ens, n_t)
end

for (k, idx) in enumerate(ensemble_idx)
    p_full = build_params(PC[:, idx])
    for (cond, frac) in zip(all_conditions, all_fractions)
        try
            _, thr = simulate_thrombin_FII(p_full; FII_fraction=frac)
            Thr_all[cond][k, :] = thr
        catch
            Thr_all[cond][k, :] .= NaN
        end
    end
end

# Filter valid members (all conditions must succeed)
valid_mask = ones(Bool, n_ens)
for cond in all_conditions
    valid_mask .&= .!any(isnan.(Thr_all[cond]), dims=2)[:]
end
for cond in all_conditions
    Thr_all[cond] = Thr_all[cond][valid_mask, :]
end
ens_idx_valid = ensemble_idx[valid_mask]
n_valid = sum(valid_mask)
println("  Valid ensemble members: $n_valid / $n_ens")

# ──────────────────────────────────────────────────────────────
# Print results
# ──────────────────────────────────────────────────────────────
println("\nTraining fit quality:")
for (cond, label) in zip(TRAIN_CONDITIONS, TRAIN_LABELS)
    μ, _, _ = ensemble_stats(Thr_all[cond])
    d = exp_data[cond]
    d_interp = interp_exp(cond)
    peak_data = maximum(d.thrombin_nM)
    peak_ens = maximum(μ)
    err = abs(peak_ens - peak_data) / (peak_data + 1e-6) * 100
    println("  $label: data peak=$(round(peak_data, digits=1)) nM, " *
            "ensemble peak=$(round(peak_ens, digits=1)) nM, err=$(round(err, digits=1))%")
end

println("\nValidation predictions:")
for (cond, label) in zip(VALID_CONDITIONS, VALID_LABELS)
    μ, lo, hi = ensemble_stats(Thr_all[cond])
    d = exp_data[cond]
    d_interp = interp_exp(cond)
    peak_data = maximum(d.thrombin_nM)
    peak_ens = maximum(μ)
    err = abs(peak_ens - peak_data) / (peak_data + 1e-6) * 100
    # Check coverage at experimental time points
    covered = all(i -> lo[i] <= d_interp[i] <= hi[i] || d_interp[i] < 1.0,
                  1:length(SIM_TIMES_MIN))
    println("  $label: data peak=$(round(peak_data, digits=1)) nM, " *
            "ensemble peak=$(round(peak_ens, digits=1)) nM, err=$(round(err, digits=1))%, " *
            "covered=$covered")
end

# ──────────────────────────────────────────────────────────────
# Figure: Real-data coagulation results (2×2)
#   (a) Training fits (50%, 100%, 150% FII)
#   (b) Held-out predictions (75%, 125% FII)
#   (c) Pareto front
#   (d) Parameter estimates vs nominal
# ──────────────────────────────────────────────────────────────
include("paper_theme.jl")
set_paper_theme!()

println("\nGenerating real-data coagulation figure...")

# Colors: warm-to-cool gradient across FII levels
const C_FII = [
    (RGBAf(0.80, 0.15, 0.15, 0.20), RGBf(0.80, 0.15, 0.15), RGBf(0.60, 0.10, 0.10)),  # 50% - red
    (RGBAf(0.90, 0.55, 0.10, 0.20), RGBf(0.85, 0.45, 0.05), RGBf(0.70, 0.35, 0.00)),  # 75% - amber
    (RGBAf(0.30, 0.30, 0.30, 0.15), RGBf(0.25, 0.25, 0.25), RGBf(0.15, 0.15, 0.15)),  # 100% - dark gray
    (RGBAf(0.15, 0.65, 0.45, 0.20), RGBf(0.10, 0.55, 0.35), RGBf(0.05, 0.40, 0.25)),  # 125% - teal
    (RGBAf(0.20, 0.45, 0.78, 0.20), RGBf(0.20, 0.45, 0.78), RGBf(0.10, 0.30, 0.60)),  # 150% - blue
]
# Map conditions to color indices
const COND_CIDX = Dict("FII_50pct"=>1, "FII_75pct"=>2, "FII_100pct"=>3, "FII_125pct"=>4, "FII_150pct"=>5)

let
    fig = Figure(size = (1100, 820))

    # --- (a) Training fits ---
    ax_a = Axis(fig[1, 1],
        xlabel = "Time (min)", ylabel = "Thrombin (nmol/L)",
        title = "(a)  Training: 50%, 100%, 150% prothrombin")

    for (cond, label) in zip(TRAIN_CONDITIONS, TRAIN_LABELS)
        ci = COND_CIDX[cond]
        c_fill, c_mean, c_data = C_FII[ci]
        μ, lo, hi = ensemble_stats(Thr_all[cond])
        d = exp_data[cond]
        band!(ax_a, SIM_TIMES_MIN, lo, hi, color = c_fill)
        lines!(ax_a, SIM_TIMES_MIN, μ, color = c_mean, linewidth = 2, linestyle = :dash)
        scatter!(ax_a, d.time_min, d.thrombin_nM, color = c_data, markersize = 8)
    end

    # --- (b) Held-out predictions ---
    ax_b = Axis(fig[1, 2],
        xlabel = "Time (min)", ylabel = "Thrombin (nmol/L)",
        title = "(b)  Validation: 75%, 125% prothrombin")

    for (cond, label) in zip(VALID_CONDITIONS, VALID_LABELS)
        ci = COND_CIDX[cond]
        c_fill, c_mean, c_data = C_FII[ci]
        μ, lo, hi = ensemble_stats(Thr_all[cond])
        d = exp_data[cond]
        band!(ax_b, SIM_TIMES_MIN, lo, hi, color = c_fill)
        lines!(ax_b, SIM_TIMES_MIN, μ, color = c_mean, linewidth = 2, linestyle = :dash)
        scatter!(ax_b, d.time_min, d.thrombin_nM, color = c_data, markersize = 8)
    end

    # --- (c) Pareto front ---
    ax_c = Axis(fig[2, 1],
        xlabel = "log₁₀(ε₁, 50% FII)",
        ylabel = "log₁₀(ε₂, 100% FII)",
        title = "(c)  Pareto front")

    log_e1 = log10.(EC[1, :] .+ 1e-10)
    log_e2 = log10.(EC[2, :] .+ 1e-10)
    e3_vals = EC[3, :]
    e3_norm = clamp.((e3_vals .- minimum(e3_vals)) ./ (maximum(e3_vals) - minimum(e3_vals) + 1e-10), 0, 1)
    colors = [RGBAf(0.20 + 0.40*t, 0.45 + 0.25*t, 0.78 - 0.25*t, 0.55) for t in e3_norm]

    order = sortperm(RA, rev=true)
    scatter!(ax_c, log_e1[order], log_e2[order], color = colors[order], markersize = 4)
    p_idx = findall(RA .== 0)
    scatter!(ax_c, log_e1[p_idx], log_e2[p_idx], color = C_FRONT, markersize = 6)

    # --- (d) Parameter estimates vs nominal ---
    ax_d = Axis(fig[2, 2],
        xlabel = "Nominal value (log₁₀)", ylabel = "Estimated value (log₁₀)",
        title = "(d)  Parameter estimates")
    for k in 1:min(n_valid, 200)
        scatter!(ax_d, NOMINAL_LOG, PC[:, ens_idx_valid[k]],
            color = (C_PE, 0.08), markersize = 3)
    end
    lims = (minimum(LOG_LOWER) - 0.5, maximum(LOG_UPPER) + 0.5)
    lines!(ax_d, [lims[1], lims[2]], [lims[1], lims[2]],
        color = C_THEORY, linewidth = 1.5, linestyle = :dash)
    median_est = vec(median(PC[:, ens_idx_valid], dims=2))
    scatter!(ax_d, NOMINAL_LOG, median_est, color = C_DATA, markersize = 8, marker = :diamond)

    # Compact legend
    legend_elems = [
        MarkerElement(color = :gray40, marker = :circle, markersize = 8),
        LineElement(color = :gray40, linewidth = 2, linestyle = :dash),
        PolyElement(color = RGBAf(0.5, 0.5, 0.5, 0.20)),
        LineElement(color = C_THEORY, linewidth = 1.5, linestyle = :dash),
        MarkerElement(color = C_DATA, marker = :diamond, markersize = 8),
        MarkerElement(color = C_FRONT, marker = :circle, markersize = 6),
        # FII level colors
        PolyElement(color = C_FII[1][1], strokecolor = C_FII[1][2], strokewidth = 1),
        PolyElement(color = C_FII[2][1], strokecolor = C_FII[2][2], strokewidth = 1),
        PolyElement(color = C_FII[3][1], strokecolor = C_FII[3][2], strokewidth = 1),
        PolyElement(color = C_FII[4][1], strokecolor = C_FII[4][2], strokewidth = 1),
        PolyElement(color = C_FII[5][1], strokecolor = C_FII[5][2], strokewidth = 1),
    ]
    legend_labels = [
        "Exp. data", "Ensemble mean", "95% CI",
        "Identity / Nominal", "Median estimate", "Pareto front",
        "50% FII", "75% FII", "100% FII", "125% FII", "150% FII",
    ]
    Legend(fig[3, 1:2], legend_elems, legend_labels,
        orientation = :horizontal, tellwidth = false, tellheight = true, nbanks = 1)

    save(joinpath(FIGDIR, "fig_coagulation_realdata.pdf"), fig)
    println("  Saved fig_coagulation_realdata.pdf")
end

println("\nDone!")
