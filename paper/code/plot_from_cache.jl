# Plot ALL figures from cached JLD2 results — NO simulations.
# Run:  julia --project=. plot_from_cache.jl
#
# This script loads pre-computed results and generates publication figures.
# To regenerate simulation data, run the individual *_study.jl scripts.

using JLD2, CairoMakie, Statistics

const FIGDIR = joinpath(@__DIR__, "..", "figures")
const ARXIVDIR = joinpath(@__DIR__, "..", "arxiv", "figures")
const CACHEDIR = joinpath(@__DIR__, "data")
mkpath(FIGDIR)

include("paper_theme.jl")
set_paper_theme!()

function safe_save(path, fig)
    for attempt in 1:3
        try
            save(path, fig)
            println("  Saved $(basename(path))")
            return
        catch e
            @warn "Attempt $attempt failed: $e"
            sleep(1)
        end
    end
    @error "FAILED to save $(basename(path))"
end

function copy_to_arxiv(name)
    src = joinpath(FIGDIR, name)
    dst = joinpath(ARXIVDIR, name)
    if isfile(src) && isdir(dirname(dst))
        cp(src, dst, force=true)
        println("  Copied to arxiv/figures/")
    end
end

# Colors from paper_theme.jl are already loaded

function ensemble_stats_nM(M)
    μ  = vec(mean(M, dims=1)) .* 1e9
    lo = vec(mapslices(x -> quantile(x, 0.025), M, dims=1)) .* 1e9
    hi = vec(mapslices(x -> quantile(x, 0.975), M, dims=1)) .* 1e9
    return μ, lo, hi
end

# ══════════════════════════════════════════════════════════════
# FIG 3: Ensemble insights (4-panel)
# ══════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════
# FIG: Cell-free gene expression (3-panel)
# ══════════════════════════════════════════════════════════════
println("\n=== Cell-free ===")
@load joinpath(CACHEDIR, "cellfree_results.jld2") EC PC RA t_fine m_mean m_lo m_hi p_mean p_lo p_hi T_OBS m_data m_err p_data_uM p_err_uM

# Rename to avoid collision with coagulation EC/PC/RA loaded later
EC_cf, PC_cf, RA_cf = EC, PC, RA

let
    fig = Figure(size = (1100, 400))

    # (a) mRNA — solid = ensemble mean, scatter = experimental data
    ax_a = Axis(fig[1, 1], xlabel = "Time (h)", ylabel = "mRNA deGFP (nM)",
        title = "(a)  mRNA")
    band!(ax_a, t_fine, m_lo, m_hi, color = RGBAf(0.20, 0.45, 0.78, 0.15))
    lines!(ax_a, t_fine, m_mean, color = C_MEAN, linewidth = 2, label = "Ensemble mean")
    errorbars!(ax_a, T_OBS, m_data, m_err, color = C_DATA, whiskerwidth = 5)
    scatter!(ax_a, T_OBS, m_data, color = C_DATA, markersize = 10, label = "Exp. data")
    axislegend(ax_a, position = :rb, framevisible = false, labelsize = 15)

    # (b) Protein — solid = ensemble mean, scatter = experimental data
    ax_b = Axis(fig[1, 2], xlabel = "Time (h)", ylabel = "Protein deGFP (μM)",
        title = "(b)  Protein")
    band!(ax_b, t_fine, p_lo ./ 1000, p_hi ./ 1000, color = RGBAf(0.20, 0.45, 0.78, 0.15))
    lines!(ax_b, t_fine, p_mean ./ 1000, color = C_MEAN, linewidth = 2, label = "Ensemble mean")
    errorbars!(ax_b, T_OBS, p_data_uM, p_err_uM, color = C_DATA, whiskerwidth = 5)
    scatter!(ax_b, T_OBS, p_data_uM, color = C_DATA, markersize = 10, label = "Exp. data")
    axislegend(ax_b, position = :rb, framevisible = false, labelsize = 15)

    # (c) Pareto front
    ax_c = Axis(fig[1, 3], xlabel = "log₁₀(ε mRNA)", ylabel = "log₁₀(ε protein)",
        title = "(c)  Pareto front")
    log_e1 = log10.(EC_cf[1, :] .+ 1e-10)
    log_e2 = log10.(EC_cf[2, :] .+ 1e-10)
    n_idx = RA_cf .> 0
    p_idx = RA_cf .== 0
    scatter!(ax_c, log_e1[n_idx], log_e2[n_idx], color = (C_PE, 0.3), markersize = 7, label = "Near-optimal")
    scatter!(ax_c, log_e1[p_idx], log_e2[p_idx], color = C_FRONT, markersize = 9, label = "Pareto front")
    axislegend(ax_c, position = :rt, framevisible = false, labelsize = 15)

    safe_save(joinpath(FIGDIR, "fig_cellfree.pdf"), fig)
    copy_to_arxiv("fig_cellfree.pdf")
end

# ══════════════════════════════════════════════════════════════
# FIG: Ensemble insights (4-panel)
# ══════════════════════════════════════════════════════════════
println("\n=== Ensemble insights ===")
@load joinpath(CACHEDIR, "coagulation_results.jld2") EC PC RA train_times train_true train_data ens_idx_valid valid_mask cor_matrix
@load joinpath(CACHEDIR, "coagulation_fine.jld2") train_times_fine train_true_fine Thr_train_fine valid_times_fine valid_true_fine Thr_valid_fine Thr_normal_fine Thr_mild_fine Thr_severe_fine true_normal_fine true_mild_fine true_severe_fine
@load joinpath(CACHEDIR, "coagulation_tga_features.jld2") tga_data

N_EST = size(PC, 1)
ESTIMATE_INDICES = [10, 15, 16, 17, 22, 26, 31, 32, 38, 41]
VALID_TF = [10.0, 20.0, 30.0]
VALID_LABELS = ["10 pM TF", "20 pM TF", "30 pM TF"]
n_valid = length(ens_idx_valid)

C_VALID_TF = [
    (RGBAf(0.85, 0.35, 0.10, 0.20), RGBf(0.75, 0.30, 0.05), RGBf(0.60, 0.20, 0.00)),
    (RGBAf(0.55, 0.25, 0.70, 0.20), RGBf(0.45, 0.18, 0.55), RGBf(0.35, 0.10, 0.45)),
    (RGBAf(0.15, 0.65, 0.45, 0.20), RGBf(0.10, 0.50, 0.35), RGBf(0.05, 0.40, 0.25)),
]

# C_NORMAL, C_MILD, C_SEVERE from paper_theme.jl

valid_stats = [ensemble_stats_nM(M) for M in Thr_valid_fine]

let
    fig = Figure(size = (1200, 820))

    # (a) Held-out predictions
    ax_a = Axis(fig[1, 1:2], xlabel = "Time (s)", ylabel = "Total thrombin (nM)",
        title = "(a)  Held-out predictions (10, 20, 30 pM TF)")
    for i in 1:3
        c_fill, c_mean, c_true = C_VALID_TF[i]
        μ, lo, hi = valid_stats[i]
        band!(ax_a, valid_times_fine[i], lo, hi, color = c_fill)
        lines!(ax_a, valid_times_fine[i], μ, color = c_mean, linewidth = 2, label = VALID_LABELS[i])
        lines!(ax_a, valid_times_fine[i], valid_true_fine[i] .* 1e9,
            color = c_true, linewidth = 1, linestyle = :dash)
    end
    lines!(ax_a, [NaN], [NaN], color = C_TRUE, linewidth = 1, linestyle = :dash, label = "True")
    axislegend(ax_a, position = :rt, framevisible = false, labelsize = 15)

    # (b) Correlation heatmap
    ax_b = Axis(fig[1, 3], title = "(b)  Parameter correlations",
        xticks = (1:N_EST, ["p$(ESTIMATE_INDICES[i])" for i in 1:N_EST]),
        yticks = (1:N_EST, ["p$(ESTIMATE_INDICES[i])" for i in 1:N_EST]),
        xticklabelrotation = π/4, yreversed = true, backgroundcolor = :white)
    hm = heatmap!(ax_b, 1:N_EST, 1:N_EST, cor_matrix, colormap = :RdBu, colorrange = (-1, 1))
    Colorbar(fig[1, 4], hm, label = "r", width = 12)

    # (c) Hemophilia predictions
    ax_c = Axis(fig[2, 1:2], xlabel = "Time (s)", ylabel = "Total thrombin (nM)",
        title = "(c)  Factor VIII deficiency (hemophilia A)")
    for (label, M, true_traj, col) in [
            ("100% FVIII", Thr_normal_fine, true_normal_fine, C_NORMAL),
            ("30% FVIII",  Thr_mild_fine,   true_mild_fine,   C_MILD),
            ("5% FVIII",   Thr_severe_fine, true_severe_fine,  C_SEVERE)]
        v = .!any(isnan.(M), dims=2)[:]
        μ, lo, hi = ensemble_stats_nM(M[v, :])
        band!(ax_c, train_times_fine[1], lo, hi, color = (col, 0.15))
        lines!(ax_c, train_times_fine[1], μ, color = col, linewidth = 2, label = label)
        lines!(ax_c, train_times_fine[1], true_traj .* 1e9, color = C_TRUE, linewidth = 1, linestyle = :dash)
    end
    lines!(ax_c, [NaN], [NaN], color = C_TRUE, linewidth = 1, linestyle = :dash, label = "True")
    axislegend(ax_c, position = :rt, framevisible = false, labelsize = 15)

    # (d) TGA feature accuracy — from cached tga_data
    ax_d = Axis(fig[2, 3], ylabel = "Predicted / True", title = "(d)  TGA feature accuracy")
    display_features = [:lagtime, :peak, :etp]
    display_labels = ["Lag time", "Peak IIa", "ETP"]

    x_pos = 0
    xtick_pos = Float64[]
    xtick_labels_d = String[]
    for (fi, fn) in enumerate(display_features)
        for (ti, tf) in enumerate(VALID_TF)
            x_pos += 1
            key = "tf$(Int(tf))_$(fn)"
            d = tga_data[key]
            c_fill, c_mean, _ = C_VALID_TF[ti]
            lines!(ax_d, [x_pos, x_pos], [d.lo, d.hi], color = c_mean, linewidth = 3)
            scatter!(ax_d, [x_pos], [d.mu], color = c_mean, markersize = 13)
            mkr_color = d.covered ? :black : C_THEORY
            mkr_label = d.covered ? "Truth covered" : "Truth not covered"
            scatter!(ax_d, [x_pos], [1.0], color = mkr_color, markersize = 11, marker = :xcross, label = mkr_label)
            push!(xtick_pos, x_pos)
            push!(xtick_labels_d, "$(Int(tf))")
        end
        x_pos += 0.5
    end
    hlines!(ax_d, [1.0], color = :gray50, linewidth = 0.8, linestyle = :dash)
    ax_d.xticks = (xtick_pos, xtick_labels_d)
    for sep in [3.75, 7.25]
        vlines!(ax_d, [sep], color = :gray70, linewidth = 0.6, linestyle = :dot)
    end

    # Feature labels — place using data coordinates above the max CI
    all_hi = [tga_data["tf$(Int(tf))_$(fn)"].hi for fn in display_features for tf in VALID_TF]
    y_label = maximum(all_hi) + 0.03
    for (fi, fl) in enumerate(display_labels)
        mid_x = mean(xtick_pos[(fi-1)*3+1 : fi*3])
        text!(ax_d, mid_x, y_label; text = fl, align = (:center, :bottom), fontsize = 17, font = :bold)
    end
    ylims!(ax_d, nothing, y_label + 0.05)

    axislegend(ax_d, position = :rb, framevisible = false, labelsize = 15, unique = true)

    safe_save(joinpath(FIGDIR, "fig_ensemble_insights.pdf"), fig)
    copy_to_arxiv("fig_ensemble_insights.pdf")
end

# ══════════════════════════════════════════════════════════════
# FIG: Misspecification (4-panel)
# ══════════════════════════════════════════════════════════════
println("\n=== Misspecification ===")
@load joinpath(CACHEDIR, "misspecification_results.jld2") EC PC RA train_times train_data train_true_perturbed Thr_train Thr_valid valid_times valid_true ens_idx_valid valid_mask
@load joinpath(CACHEDIR, "misspecification_tga_features.jld2") misspec_tga_data NOMINAL_LOG TRUE_LOG

n_valid = sum(valid_mask)

C_TF_TRAIN = [
    (RGBAf(0.90, 0.45, 0.10, 0.25), RGBf(0.85, 0.35, 0.05), RGBf(0.70, 0.25, 0.00)),
    (RGBAf(0.55, 0.25, 0.70, 0.25), RGBf(0.45, 0.20, 0.60), RGBf(0.35, 0.12, 0.50)),
    (RGBAf(0.15, 0.65, 0.45, 0.25), RGBf(0.10, 0.50, 0.35), RGBf(0.05, 0.40, 0.25)),
]
C_TF_VALID = [
    (RGBAf(0.85, 0.35, 0.10, 0.20), RGBf(0.75, 0.30, 0.05), RGBf(0.60, 0.20, 0.00)),
    (RGBAf(0.55, 0.25, 0.70, 0.20), RGBf(0.45, 0.18, 0.55), RGBf(0.35, 0.10, 0.45)),
    (RGBAf(0.15, 0.65, 0.45, 0.20), RGBf(0.10, 0.50, 0.35), RGBf(0.05, 0.40, 0.25)),
]
TF_TRAIN_LABELS = ["5 pM TF", "15 pM TF", "25 pM TF"]
TF_VALID_LABELS = ["10 pM TF", "20 pM TF", "30 pM TF"]
TRAIN_TF = [5.0, 15.0, 25.0]
VALID_TF = [10.0, 20.0, 30.0]
LOG_LOWER = TRUE_LOG .- 1.5
LOG_UPPER = TRUE_LOG .+ 1.5

let
    fig = Figure(size = (1100, 820))
    sparse = 1:3:121

    # (a) Training fits — solid = ensemble mean, scatter = noisy data
    ax_a = Axis(fig[1, 1], xlabel = "Time (s)", ylabel = "Total thrombin (nM)",
        title = "(a)  Training fits (misspecified model)")
    for i in 1:3
        c_fill, c_mean, c_data = C_TF_TRAIN[i]
        μ, lo, hi = ensemble_stats_nM(Thr_train[i])
        band!(ax_a, train_times[i], lo, hi, color = c_fill)
        lines!(ax_a, train_times[i], μ, color = c_mean, linewidth = 2, label = TF_TRAIN_LABELS[i])
        scatter!(ax_a, train_times[i][sparse], train_data[i][sparse] .* 1e9,
            color = c_data, markersize = 8)
    end
    scatter!(ax_a, [NaN], [NaN], color = C_DATA, markersize = 8, label = "Synthetic training data")
    axislegend(ax_a, position = :rt, framevisible = false, labelsize = 15)

    # (b) Held-out predictions — solid = ensemble mean, dashed = true
    ax_b = Axis(fig[1, 2], xlabel = "Time (s)", ylabel = "Total thrombin (nM)",
        title = "(b)  Held-out predictions (10, 20, 30 pM TF)")
    for i in 1:3
        c_fill, c_mean, c_true = C_TF_VALID[i]
        μ, lo, hi = ensemble_stats_nM(Thr_valid[i])
        band!(ax_b, valid_times[i], lo, hi, color = c_fill)
        lines!(ax_b, valid_times[i], μ, color = c_mean, linewidth = 2, label = TF_VALID_LABELS[i])
        lines!(ax_b, valid_times[i], valid_true[i] .* 1e9,
            color = c_true, linewidth = 1, linestyle = :dash)
    end
    lines!(ax_b, [NaN], [NaN], color = C_TRUE, linewidth = 1, linestyle = :dash, label = "Perturbed model")
    axislegend(ax_b, position = :rt, framevisible = false, labelsize = 15)

    # (c) TGA feature accuracy — from cached data
    ax_c = Axis(fig[2, 1], ylabel = "Predicted / True",
        title = "(c)  TGA feature accuracy (misspecified)")
    display_features = [:lagtime, :peak, :etp]
    display_labels = ["Lag time", "Peak IIa", "ETP"]

    x_pos = 0
    xtick_pos = Float64[]
    xtick_labels_c = String[]
    for (fi, fn) in enumerate(display_features)
        for (ti, tf) in enumerate(VALID_TF)
            x_pos += 1
            key = "tf$(Int(tf))_$(fn)"
            d = misspec_tga_data[key]
            c_fill, c_mean, _ = C_TF_VALID[ti]
            lines!(ax_c, [x_pos, x_pos], [d.lo, d.hi], color = c_mean, linewidth = 3)
            scatter!(ax_c, [x_pos], [d.mu], color = c_mean, markersize = 13)
            mkr_color = d.covered ? :black : C_THEORY
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
    # Feature labels above data
    all_hi = [misspec_tga_data["tf$(Int(tf))_$(fn)"].hi for fn in display_features for tf in VALID_TF]
    y_label = maximum(all_hi) + 0.03
    for (fi, fl) in enumerate(display_labels)
        mid_x = mean(xtick_pos[(fi-1)*3+1 : fi*3])
        text!(ax_c, mid_x, y_label; text = fl, align = (:center, :bottom), fontsize = 17, font = :bold)
    end
    ylims!(ax_c, nothing, y_label + 0.05)

    # (d) Parameter recovery
    ax_d = Axis(fig[2, 2], xlabel = "True value (log₁₀)", ylabel = "Estimated value (log₁₀)",
        title = "(d)  Parameter recovery (misspecified)")
    lims = (minimum(LOG_LOWER) - 0.5, maximum(LOG_UPPER) + 0.5)
    lines!(ax_d, [lims[1], lims[2]], [lims[1], lims[2]],
        color = C_THEORY, linewidth = 1.5, linestyle = :dash, label = "Identity")
    for k in 1:min(n_valid, 200)
        scatter!(ax_d, TRUE_LOG, PC[:, ens_idx_valid[k]],
            color = (C_PE, 0.08), markersize = 8)
    end
    scatter!(ax_d, [NaN], [NaN], color = (C_PE, 0.4), markersize = 9, label = "Ensemble members")
    median_est = vec(median(PC[:, ens_idx_valid], dims=2))
    scatter!(ax_d, TRUE_LOG, median_est, color = C_DATA, markersize = 13, marker = :diamond, label = "Median")
    axislegend(ax_d, position = :rb, framevisible = false, labelsize = 15)

    safe_save(joinpath(FIGDIR, "fig_misspecification.pdf"), fig)
    copy_to_arxiv("fig_misspecification.pdf")
end

# ══════════════════════════════════════════════════════════════
# FIG: Coagulation training (3-panel, 2-row)
# ══════════════════════════════════════════════════════════════
println("\n=== Coagulation training ===")
# Re-use coagulation data already loaded above

ESTIMATE_INDICES_coag = [10, 15, 16, 17, 22, 26, 31, 32, 38, 41]
TRUE_LOG_coag = log10.([6.0, 1.8, 7.5e3, 2.0e7, 8.2, 2.0e7, 63.5, 1.5e7, 1.5e3, 7.1e3])
LOG_LOWER_coag = TRUE_LOG_coag .- 1.5
LOG_UPPER_coag = TRUE_LOG_coag .+ 1.5

C_TF_coag = [
    (RGBAf(0.90, 0.45, 0.10, 0.25), RGBf(0.85, 0.35, 0.05), RGBf(0.70, 0.25, 0.00)),  # 5 pM amber
    (RGBAf(0.55, 0.25, 0.70, 0.25), RGBf(0.45, 0.20, 0.60), RGBf(0.35, 0.12, 0.50)),  # 15 pM purple
    (RGBAf(0.15, 0.65, 0.45, 0.25), RGBf(0.10, 0.50, 0.35), RGBf(0.05, 0.40, 0.25)),  # 25 pM teal
]
TF_LABELS_coag = ["5 pM TF", "15 pM TF", "25 pM TF"]

train_stats_coag = [ensemble_stats_nM(M) for M in Thr_train_fine]

let
    fig = Figure(size = (1100, 820))
    sparse = 1:3:121

    # (a) Training fits — solid = ensemble mean, scatter = synthetic data
    ax_a = Axis(fig[1, 1:2], xlabel = "Time (s)", ylabel = "Total thrombin (nM)",
        title = "(a)  Training data: TGA at 5, 15, and 25 pM TF")
    for i in 1:3
        c_fill, c_mean, c_data = C_TF_coag[i]
        μ, lo, hi = train_stats_coag[i]
        band!(ax_a, train_times_fine[i], lo, hi, color = c_fill)
        lines!(ax_a, train_times_fine[i], μ, color = c_mean, linewidth = 2, label = TF_LABELS_coag[i])
        scatter!(ax_a, train_times[i][sparse], train_data[i][sparse] .* 1e9,
            color = c_data, markersize = 8)
    end
    scatter!(ax_a, [NaN], [NaN], color = C_DATA, markersize = 8, label = "Synthetic training data")
    axislegend(ax_a, position = :rt, framevisible = false, labelsize = 15)

    # (b) Parameter recovery — identity + ensemble cloud + median
    ax_b = Axis(fig[2, 1], xlabel = "True value (log₁₀)", ylabel = "Estimated value (log₁₀)",
        title = "(b)  Parameter recovery")
    lims = (minimum(LOG_LOWER_coag) - 0.5, maximum(LOG_UPPER_coag) + 0.5)
    lines!(ax_b, [lims[1], lims[2]], [lims[1], lims[2]],
        color = C_THEORY, linewidth = 1.5, linestyle = :dash, label = "Identity")
    for k in 1:min(length(ens_idx_valid), 200)
        scatter!(ax_b, TRUE_LOG_coag, PC[:, ens_idx_valid[k]],
            color = (C_PE, 0.12), markersize = 8)
    end
    scatter!(ax_b, [NaN], [NaN], color = (C_PE, 0.4), markersize = 9, label = "Ensemble members")
    median_est_coag = vec(median(PC[:, ens_idx_valid], dims=2))
    scatter!(ax_b, TRUE_LOG_coag, median_est_coag, color = C_DATA, markersize = 13, marker = :diamond, label = "Median")
    axislegend(ax_b, position = :rb, framevisible = false, labelsize = 15)

    # (c) Pareto front projection — rank-based coloring
    ax_c = Axis(fig[2, 2], xlabel = "log₁₀(ε₁, 5 pM TF)", ylabel = "log₁₀(ε₂, 15 pM TF)",
        title = "(c)  Pareto front")
    log_e1 = log10.(EC[1, :] .+ 1e-10)
    log_e2 = log10.(EC[2, :] .+ 1e-10)
    near_idx = findall(RA .> 0)
    p_idx = findall(RA .== 0)
    scatter!(ax_c, log_e1[near_idx], log_e2[near_idx],
        color = (C_PE, 0.3), markersize = 7, label = "Near-optimal")
    scatter!(ax_c, log_e1[p_idx], log_e2[p_idx],
        color = C_FRONT, markersize = 9, label = "Pareto front")
    axislegend(ax_c, position = :rt, framevisible = false, labelsize = 15)

    safe_save(joinpath(FIGDIR, "fig_coagulation.pdf"), fig)
    copy_to_arxiv("fig_coagulation.pdf")
end

# ══════════════════════════════════════════════════════════════
# FIG: Real-data coagulation (4-panel)
# ══════════════════════════════════════════════════════════════
println("\n=== Real-data coagulation ===")
@load joinpath(CACHEDIR, "realdata_results.jld2") EC PC RA Thr_all exp_data PLOT_TIMES NOMINAL_LOG LOG_LOWER LOG_UPPER ESTIMATE_NAMES TRAIN_CONDITIONS VALID_CONDITIONS TRAIN_LABELS VALID_LABELS ensemble_idx ens_idx_valid n_valid

function ensemble_stats_rd(M)
    μ  = vec(mean(M, dims=1))
    lo = vec(mapslices(x -> quantile(x, 0.025), M, dims=1))
    hi = vec(mapslices(x -> quantile(x, 0.975), M, dims=1))
    return μ, lo, hi
end

C_FII = [
    (RGBAf(0.80, 0.15, 0.15, 0.20), RGBf(0.80, 0.15, 0.15), RGBf(0.60, 0.10, 0.10)),  # 50% red
    (RGBAf(0.90, 0.55, 0.10, 0.20), RGBf(0.85, 0.45, 0.05), RGBf(0.70, 0.35, 0.00)),  # 75% amber
    (RGBAf(0.30, 0.30, 0.30, 0.15), RGBf(0.25, 0.25, 0.25), RGBf(0.15, 0.15, 0.15)),  # 100% gray
    (RGBAf(0.15, 0.65, 0.45, 0.20), RGBf(0.10, 0.55, 0.35), RGBf(0.05, 0.40, 0.25)),  # 125% teal
    (RGBAf(0.20, 0.45, 0.78, 0.20), RGBf(0.20, 0.45, 0.78), RGBf(0.10, 0.30, 0.60)),  # 150% blue
]
COND_CIDX = Dict("FII_50pct"=>1, "FII_75pct"=>2, "FII_100pct"=>3, "FII_125pct"=>4, "FII_150pct"=>5)

let
    fig = Figure(size = (1100, 820))

    # (a) Training fits — solid = ensemble mean, scatter = experimental data
    ax_a = Axis(fig[1, 1], xlabel = "Time (min)", ylabel = "Thrombin (nmol/L)",
        title = "(a)  Training: 50%, 100%, 150% prothrombin")
    for (cond, label) in zip(TRAIN_CONDITIONS, TRAIN_LABELS)
        ci = COND_CIDX[cond]
        c_fill, c_mean, c_data = C_FII[ci]
        μ, lo, hi = ensemble_stats_rd(Thr_all[cond])
        d = exp_data[cond]
        band!(ax_a, PLOT_TIMES, lo, hi, color = c_fill)
        lines!(ax_a, PLOT_TIMES, μ, color = c_mean, linewidth = 2, label = label)
        scatter!(ax_a, d.time_min, d.thrombin_nM, color = c_data, markersize = 11)
    end
    scatter!(ax_a, [NaN], [NaN], color = C_DATA, markersize = 11, label = "Exp. data")
    axislegend(ax_a, position = :rt, framevisible = false, labelsize = 15)

    # (b) Held-out predictions — solid = ensemble mean, scatter = experimental data
    ax_b = Axis(fig[1, 2], xlabel = "Time (min)", ylabel = "Thrombin (nmol/L)",
        title = "(b)  Validation: 75%, 125% prothrombin")
    for (cond, label) in zip(VALID_CONDITIONS, VALID_LABELS)
        ci = COND_CIDX[cond]
        c_fill, c_mean, c_data = C_FII[ci]
        μ, lo, hi = ensemble_stats_rd(Thr_all[cond])
        d = exp_data[cond]
        band!(ax_b, PLOT_TIMES, lo, hi, color = c_fill)
        lines!(ax_b, PLOT_TIMES, μ, color = c_mean, linewidth = 2, label = label)
        scatter!(ax_b, d.time_min, d.thrombin_nM, color = c_data, markersize = 11)
    end
    scatter!(ax_b, [NaN], [NaN], color = C_DATA, markersize = 11, label = "Exp. data")
    axislegend(ax_b, position = :rt, framevisible = false, labelsize = 15)

    # (c) Pareto front
    ax_c = Axis(fig[2, 1], xlabel = "log₁₀(ε₁, 50% FII)", ylabel = "log₁₀(ε₂, 100% FII)",
        title = "(c)  Pareto front")
    log_e1 = log10.(EC[1, :] .+ 1e-10)
    log_e2 = log10.(EC[2, :] .+ 1e-10)
    e3_vals = EC[3, :]
    e3_norm = clamp.((e3_vals .- minimum(e3_vals)) ./ (maximum(e3_vals) - minimum(e3_vals) + 1e-10), 0, 1)
    colors = [RGBAf(0.20 + 0.40*t, 0.45 + 0.25*t, 0.78 - 0.25*t, 0.55) for t in e3_norm]
    order = sortperm(RA, rev=true)
    scatter!(ax_c, log_e1[order], log_e2[order], color = colors[order], markersize = 7, label = "Near-optimal")
    p_idx = findall(RA .== 0)
    scatter!(ax_c, log_e1[p_idx], log_e2[p_idx], color = C_FRONT, markersize = 9, label = "Pareto front")
    axislegend(ax_c, position = :rb, framevisible = false, labelsize = 15)

    # (d) Parameter estimates vs nominal — bigger markers
    ax_d = Axis(fig[2, 2], xlabel = "Nominal value (log₁₀)", ylabel = "Estimated value (log₁₀)",
        title = "(d)  Parameter estimates")
    lims = (minimum(LOG_LOWER) - 0.5, maximum(LOG_UPPER) + 0.5)
    lines!(ax_d, [lims[1], lims[2]], [lims[1], lims[2]],
        color = C_THEORY, linewidth = 1.5, linestyle = :dash, label = "Identity")
    for k in 1:min(n_valid, 200)
        scatter!(ax_d, NOMINAL_LOG, PC[:, ens_idx_valid[k]],
            color = (C_PE, 0.08), markersize = 9)
    end
    scatter!(ax_d, [NaN], [NaN], color = (C_PE, 0.4), markersize = 9, label = "Ensemble members")
    median_est = vec(median(PC[:, ens_idx_valid], dims=2))
    scatter!(ax_d, NOMINAL_LOG, median_est, color = C_DATA, markersize = 13, marker = :diamond, label = "Median")
    axislegend(ax_d, position = :rb, framevisible = false, labelsize = 15)

    safe_save(joinpath(FIGDIR, "fig_coagulation_realdata.pdf"), fig)
    copy_to_arxiv("fig_coagulation_realdata.pdf")
end

println("\n=== Done ===")
