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
# Figure: 3-panel sensitivity analysis
# ──────────────────────────────────────────────────────────────
println("\nGenerating sensitivity figure...")
let
    fig = Figure(size = (1100, 350), fontsize = 13)

    # --- (a) rank_cutoff ---
    ax_a = Axis(fig[1, 1],
        xlabel = "Rank cutoff (R_cutoff)",
        ylabel = "Hypervolume",
        title = "(a)  Rank cutoff",
        xticks = rank_cutoffs)

    medians_rc = [median(rc_hvs[rc]) for rc in rank_cutoffs]
    lo_rc = [quantile(rc_hvs[rc], 0.25) for rc in rank_cutoffs]
    hi_rc = [quantile(rc_hvs[rc], 0.75) for rc in rank_cutoffs]
    band!(ax_a, rank_cutoffs, lo_rc, hi_rc, color = (:dodgerblue, 0.2))
    lines!(ax_a, rank_cutoffs, medians_rc, color = :dodgerblue, linewidth = 2)
    scatter!(ax_a, rank_cutoffs, medians_rc, color = :dodgerblue, markersize = 8)

    # --- (b) cooling rate ---
    ax_b = Axis(fig[1, 2],
        xlabel = "Cooling rate (α)",
        ylabel = "Hypervolume",
        title = "(b)  Cooling rate",
        xticks = cooling_rates)

    medians_alpha = [median(alpha_hvs[a]) for a in cooling_rates]
    lo_alpha = [quantile(alpha_hvs[a], 0.25) for a in cooling_rates]
    hi_alpha = [quantile(alpha_hvs[a], 0.75) for a in cooling_rates]
    band!(ax_b, cooling_rates, lo_alpha, hi_alpha, color = (:dodgerblue, 0.2))
    lines!(ax_b, cooling_rates, medians_alpha, color = :dodgerblue, linewidth = 2)
    scatter!(ax_b, cooling_rates, medians_alpha, color = :dodgerblue, markersize = 8)

    # --- (c) N_iter ---
    ax_c = Axis(fig[1, 3],
        xlabel = "Iterations per temperature (N_iter)",
        ylabel = "Hypervolume",
        title = "(c)  Iterations",
        xticks = n_iters)

    medians_ni = [median(niter_hvs[ni]) for ni in n_iters]
    lo_ni = [quantile(niter_hvs[ni], 0.25) for ni in n_iters]
    hi_ni = [quantile(niter_hvs[ni], 0.75) for ni in n_iters]
    band!(ax_c, n_iters, lo_ni, hi_ni, color = (:dodgerblue, 0.2))
    lines!(ax_c, n_iters, medians_ni, color = :dodgerblue, linewidth = 2)
    scatter!(ax_c, n_iters, medians_ni, color = :dodgerblue, markersize = 8)

    save(joinpath(FIGDIR, "fig_sensitivity.pdf"), fig)
    println("  Saved fig_sensitivity.pdf")
end

println("\nDone!")
