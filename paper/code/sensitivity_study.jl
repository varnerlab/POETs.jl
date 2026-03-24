# Hyperparameter sensitivity analysis on the Binh-Korn benchmark
# Sweeps rank_cutoff, cooling_rate, and N_iter, reports hypervolume.
#
# Run from the paper/code directory:
#   julia --project -t4 sensitivity_study.jl

using ParetoEnsembles
using CairoMakie
using Random
using Statistics

const FIGDIR = joinpath(@__DIR__, "..", "figures")

# ──────────────────────────────────────────────────────────────
# Binh-Korn problem (same as generate_figures.jl)
# ──────────────────────────────────────────────────────────────
function bk_objective(x)
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

bk_neighbor(x) = clamp.(x .* (1 .+ 0.15 * randn(length(x))), [0.0, 0.0], [5.0, 3.0])
accept_fn(R, T) = exp(-R[end] / T)

const HV_REF = [200.0, 50.0]
const N_REPS = 5
const N_CHAINS = 5  # fewer chains for speed in sweeps

# Fixed starting points
const STARTS = [[x1, x2] for (x1, x2) in zip(
    range(0.1, 4.5, length=N_CHAINS),
    range(0.1, 2.8, length=N_CHAINS))]

# Default values
const DEFAULT_RC = 8
const DEFAULT_ALPHA = 0.90
const DEFAULT_NITER = 40

function run_config(; rank_cutoff, alpha, n_iter, rep)
    cool_fn(T) = alpha * T
    t = @elapsed (EC, PC, RA) = estimate_ensemble_parallel(
        bk_objective, bk_neighbor, accept_fn, cool_fn, STARTS;
        rank_cutoff=rank_cutoff, maximum_number_of_iterations=n_iter,
        show_trace=false, rng_seed=rep)
    front_E, _ = pareto_front(EC, PC, RA)
    hv = hypervolume(front_E, HV_REF)
    return (hv=hv, n_front=size(front_E, 2), n_total=size(EC, 2), time=t)
end

# ──────────────────────────────────────────────────────────────
# Sweep 1: rank_cutoff
# ──────────────────────────────────────────────────────────────
rank_cutoffs = [2, 4, 8, 12, 20]
println("Sweep 1: rank_cutoff ∈ $rank_cutoffs")

rc_hvs = Dict{Int, Vector{Float64}}()
for rc in rank_cutoffs
    hvs = Float64[]
    for rep in 1:N_REPS
        r = run_config(rank_cutoff=rc, alpha=DEFAULT_ALPHA, n_iter=DEFAULT_NITER, rep=rep)
        push!(hvs, r.hv)
    end
    rc_hvs[rc] = hvs
    println("  rc=$rc: HV = $(round(median(hvs), digits=1)) [$(round(quantile(hvs, 0.25), digits=1)) – $(round(quantile(hvs, 0.75), digits=1))]")
end

# ──────────────────────────────────────────────────────────────
# Sweep 2: cooling rate
# ──────────────────────────────────────────────────────────────
cooling_rates = [0.80, 0.85, 0.90, 0.95]
println("\nSweep 2: cooling_rate ∈ $cooling_rates")

alpha_hvs = Dict{Float64, Vector{Float64}}()
for alpha in cooling_rates
    hvs = Float64[]
    for rep in 1:N_REPS
        r = run_config(rank_cutoff=DEFAULT_RC, alpha=alpha, n_iter=DEFAULT_NITER, rep=rep)
        push!(hvs, r.hv)
    end
    alpha_hvs[alpha] = hvs
    println("  α=$alpha: HV = $(round(median(hvs), digits=1)) [$(round(quantile(hvs, 0.25), digits=1)) – $(round(quantile(hvs, 0.75), digits=1))]")
end

# ──────────────────────────────────────────────────────────────
# Sweep 3: N_iter
# ──────────────────────────────────────────────────────────────
n_iters = [10, 20, 40, 60, 100]
println("\nSweep 3: N_iter ∈ $n_iters")

niter_hvs = Dict{Int, Vector{Float64}}()
for ni in n_iters
    hvs = Float64[]
    for rep in 1:N_REPS
        r = run_config(rank_cutoff=DEFAULT_RC, alpha=DEFAULT_ALPHA, n_iter=ni, rep=rep)
        push!(hvs, r.hv)
    end
    niter_hvs[ni] = hvs
    println("  N_iter=$ni: HV = $(round(median(hvs), digits=1)) [$(round(quantile(hvs, 0.25), digits=1)) – $(round(quantile(hvs, 0.75), digits=1))]")
end

# ──────────────────────────────────────────────────────────────
# Figure: 3-panel sensitivity analysis (with swarm/strip + box)
# ──────────────────────────────────────────────────────────────
include("paper_theme.jl")
set_paper_theme!()

const C_SWEEP = RGBf(0.20, 0.45, 0.78)

println("\nGenerating sensitivity figure...")
let
    fig = Figure(size = (1100, 380))

    # --- (a) rank_cutoff ---
    ax_a = Axis(fig[1, 1],
        xlabel = "Rank cutoff (R\u209c\u1d64\u209c\u2092ff)",
        ylabel = "Hypervolume",
        title = "(a)  Rank cutoff",
        xticks = (1:length(rank_cutoffs), string.(rank_cutoffs)))

    for (i, rc) in enumerate(rank_cutoffs)
        hvs = rc_hvs[rc]
        # Individual points with jitter
        jitter = 0.15 .* (rand(length(hvs)) .- 0.5)
        scatter!(ax_a, fill(i, length(hvs)) .+ jitter, hvs,
            color = (C_SWEEP, 0.5), markersize = 6)
        # Median bar
        med = median(hvs)
        lines!(ax_a, [i - 0.25, i + 0.25], [med, med],
            color = C_SWEEP, linewidth = 3)
    end

    # --- (b) cooling rate ---
    ax_b = Axis(fig[1, 2],
        xlabel = "Cooling rate (\u03b1)",
        ylabel = "Hypervolume",
        title = "(b)  Cooling rate",
        xticks = (1:length(cooling_rates), string.(cooling_rates)))

    for (i, alpha) in enumerate(cooling_rates)
        hvs = alpha_hvs[alpha]
        jitter = 0.15 .* (rand(length(hvs)) .- 0.5)
        scatter!(ax_b, fill(i, length(hvs)) .+ jitter, hvs,
            color = (C_SWEEP, 0.5), markersize = 6)
        med = median(hvs)
        lines!(ax_b, [i - 0.25, i + 0.25], [med, med],
            color = C_SWEEP, linewidth = 3)
    end

    # --- (c) N_iter ---
    ax_c = Axis(fig[1, 3],
        xlabel = "Iterations per temperature (N\u1d62\u209c\u2091\u1d63)",
        ylabel = "Hypervolume",
        title = "(c)  Iterations",
        xticks = (1:length(n_iters), string.(n_iters)))

    for (i, ni) in enumerate(n_iters)
        hvs = niter_hvs[ni]
        jitter = 0.15 .* (rand(length(hvs)) .- 0.5)
        scatter!(ax_c, fill(i, length(hvs)) .+ jitter, hvs,
            color = (C_SWEEP, 0.5), markersize = 6)
        med = median(hvs)
        lines!(ax_c, [i - 0.25, i + 0.25], [med, med],
            color = C_SWEEP, linewidth = 3)
    end

    save(joinpath(FIGDIR, "fig_sensitivity.pdf"), fig)
    println("  Saved fig_sensitivity.pdf")
end

println("\nDone!")
