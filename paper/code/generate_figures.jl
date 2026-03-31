# Generate all figures for the ParetoEnsembles.jl paper.
# Run from the paper/code directory:
#   julia --project -t4 generate_figures.jl
#
# Produces PDF figures in ../figures/

using ParetoEnsembles
using CairoMakie
using Random
using JLD2

const FIGDIR = joinpath(@__DIR__, "..", "figures")
const CACHEDIR = joinpath(@__DIR__, "data")
const CACHE_FILE = joinpath(CACHEDIR, "benchmarks_results.jld2")

# ──────────────────────────────────────────────────────────────
# Shared styling
# ──────────────────────────────────────────────────────────────
const MARKER_SIZE_PARETO = 4
const MARKER_SIZE_NEAR   = 6

# Color by rank: near-optimal are prominent colored points,
# Pareto front is a subtle dark line through the cloud.
function rank_colors(RA)
    max_rank = maximum(RA)
    colors = Vector{Any}(undef, length(RA))
    for i in eachindex(RA)
        if RA[i] == 0
            # front: small dark points (plotted on top as thin line)
            colors[i] = RGBAf(0.1, 0.1, 0.1, 0.8)
        else
            # near-optimal: rank 1 → saturated blue, high rank → light steel blue
            t = clamp((RA[i] - 1) / max(max_rank - 1, 1), 0, 1)
            colors[i] = RGBAf(0.15 + 0.45*t, 0.35 + 0.35*t, 0.85 - 0.15*t, 0.75 - 0.25*t)
        end
    end
    return colors
end

# ──────────────────────────────────────────────────────────────
# Binh–Korn problem definition
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
cool_fn(T) = 0.95 * T

# ──────────────────────────────────────────────────────────────
# Fonseca–Fleming problem definition (d=3)
# ──────────────────────────────────────────────────────────────
const FF_D = 3

function ff_objective(x)
    f = zeros(2, 1)
    sum1 = sum((x[i] - 1 / sqrt(FF_D))^2 for i in 1:FF_D)
    sum2 = sum((x[i] + 1 / sqrt(FF_D))^2 for i in 1:FF_D)
    f[1] = 1 - exp(-sum1)
    f[2] = 1 - exp(-sum2)
    return f
end

function ff_neighbor(x)
    new_x = x .+ 0.15 * randn(length(x))
    return clamp.(new_x, -4.0, 4.0)
end

# ──────────────────────────────────────────────────────────────
# Run optimizations
# Use rank_cutoff=8 and 10 chains to retain a visible cloud
# of near-optimal solutions alongside the Pareto front.
# ──────────────────────────────────────────────────────────────
const RANK_CUTOFF = 12
const N_ITER      = 60

println("Running Binh–Korn (10-chain parallel, rank_cutoff=$RANK_CUTOFF)...")
bk_starts = [[x1, x2] for (x1, x2) in zip(
    range(0.1, 4.5, length=10),
    range(0.1, 2.8, length=10)
)]
(EC_bk, PC_bk, RA_bk) = estimate_ensemble_parallel(
    bk_objective, bk_neighbor, accept_fn, cool_fn,
    bk_starts;
    rank_cutoff = RANK_CUTOFF,
    maximum_number_of_iterations = N_ITER,
    show_trace = false,
    rng_seed = 42
)
n_pareto_bk = count(RA_bk .== 0)
println("  Retained: $(size(EC_bk, 2)),  Pareto: $n_pareto_bk,  Near-optimal: $(size(EC_bk,2) - n_pareto_bk)")

println("Running Fonseca–Fleming (10-chain parallel, rank_cutoff=$RANK_CUTOFF)...")
ff_starts = [randn(MersenneTwister(s), FF_D) for s in 1:10]
(EC_ff, PC_ff, RA_ff) = estimate_ensemble_parallel(
    ff_objective, ff_neighbor, accept_fn, cool_fn,
    ff_starts;
    rank_cutoff = RANK_CUTOFF,
    maximum_number_of_iterations = N_ITER,
    show_trace = false,
    rng_seed = 99
)
n_pareto_ff = count(RA_ff .== 0)
println("  Retained: $(size(EC_ff, 2)),  Pareto: $n_pareto_ff,  Near-optimal: $(size(EC_ff,2) - n_pareto_ff)")

# Also run a single chain for the comparison figure
println("Running Binh–Korn (single chain, rank_cutoff=$RANK_CUTOFF)...")
Random.seed!(42)
(EC_bk1, PC_bk1, RA_bk1) = estimate_ensemble(
    bk_objective, bk_neighbor, accept_fn, cool_fn,
    [2.5, 1.5];
    rank_cutoff = RANK_CUTOFF,
    maximum_number_of_iterations = N_ITER,
    show_trace = false
)
println("  Retained: $(size(EC_bk1, 2)),  Pareto: $(count(RA_bk1 .== 0))")


# ──────────────────────────────────────────────────────────────
# Save results to cache
# ──────────────────────────────────────────────────────────────
mkpath(CACHEDIR)
@save CACHE_FILE EC_bk PC_bk RA_bk EC_ff PC_ff RA_ff EC_bk1 PC_bk1 RA_bk1
println("Saved benchmark results to $CACHE_FILE")

# ──────────────────────────────────────────────────────────────
# Figure 1: Combined 2×2 panel
#   (a) BK objective space   (b) FF objective space
#   (c) BK parameter space   (d) FF parameter space (x1 vs x2)
# ──────────────────────────────────────────────────────────────
include("paper_theme.jl")
set_paper_theme!()

println("Generating Figure 1 (combined 2×2 panel)...")
let
    fig = Figure(size = (900, 800))

    # Compute rank-based colors for each dataset
    colors_bk = rank_colors(RA_bk)
    colors_ff = rank_colors(RA_ff)

    # Sort by rank descending so front (rank=0) is plotted on top
    order_bk = sortperm(RA_bk)
    order_ff = sortperm(RA_ff)

    # --- (a) Binh–Korn objective space ---
    ax_a = Axis(fig[1, 1],
        xlabel = L"f_1", ylabel = L"f_2",
        title = "(a)  Binh–Korn (objective space)")
    scatter!(ax_a, EC_bk[1, order_bk], EC_bk[2, order_bk],
        color = colors_bk[order_bk],
        markersize = [RA_bk[i] == 0 ? MARKER_SIZE_PARETO : MARKER_SIZE_NEAR for i in order_bk])

    # --- (b) Fonseca–Fleming objective space ---
    ax_b = Axis(fig[1, 2],
        xlabel = L"f_1", ylabel = L"f_2",
        title = "(b)  Fonseca–Fleming (objective space)")

    scatter!(ax_b, EC_ff[1, order_ff], EC_ff[2, order_ff],
        color = colors_ff[order_ff],
        markersize = [RA_ff[i] == 0 ? MARKER_SIZE_PARETO : MARKER_SIZE_NEAR for i in order_ff])

    # theoretical front: t ∈ [-1, 1] (plotted last so it's visible on top)
    t_vals = range(-1, 1, length = 300)
    f1_front = 1 .- exp.(-(t_vals .- 1).^2)
    f2_front = 1 .- exp.(-(t_vals .+ 1).^2)
    lines!(ax_b, f1_front, f2_front,
        color = :red, linewidth = 1.5, linestyle = :dash)

    # --- (c) Binh–Korn parameter space ---
    ax_c = Axis(fig[2, 1],
        xlabel = L"x_1", ylabel = L"x_2",
        title = "(c)  Binh–Korn (parameter space)")
    scatter!(ax_c, PC_bk[1, order_bk], PC_bk[2, order_bk],
        color = colors_bk[order_bk],
        markersize = [RA_bk[i] == 0 ? MARKER_SIZE_PARETO : MARKER_SIZE_NEAR for i in order_bk])

    # --- (d) Fonseca–Fleming parameter space (x1 vs x2 projection) ---
    ax_d = Axis(fig[2, 2],
        xlabel = L"x_1", ylabel = L"x_2",
        title = "(d)  Fonseca–Fleming (parameter space)")
    scatter!(ax_d, PC_ff[1, order_ff], PC_ff[2, order_ff],
        color = colors_ff[order_ff],
        markersize = [RA_ff[i] == 0 ? MARKER_SIZE_PARETO : MARKER_SIZE_NEAR for i in order_ff])

    # shared legend at bottom
    elem_pareto = MarkerElement(color = RGBAf(0.1, 0.1, 0.1, 0.8), marker = :circle,
        markersize = MARKER_SIZE_PARETO)
    elem_near_lo = MarkerElement(color = RGBAf(0.15, 0.35, 0.85, 0.75), marker = :circle,
        markersize = MARKER_SIZE_NEAR)
    elem_near_hi = MarkerElement(color = RGBAf(0.6, 0.7, 0.7, 0.5), marker = :circle,
        markersize = MARKER_SIZE_NEAR)
    elem_theo = LineElement(color = :red, linewidth = 1.5, linestyle = :dash)
    Legend(fig[3, 1:2],
        [elem_pareto, elem_near_lo, elem_near_hi, elem_theo],
        ["Rank = 0 (Pareto front)", "Low rank (near front)", "High rank", "Theoretical front"],
        orientation = :horizontal, framevisible = false,
        tellwidth = false, tellheight = true)

    save(joinpath(FIGDIR, "fig_benchmarks.pdf"), fig)
    println("  Saved fig_benchmarks.pdf")
end

# ──────────────────────────────────────────────────────────────
# Figure 2: Single-chain vs multi-chain comparison (Binh–Korn)
# ──────────────────────────────────────────────────────────────
println("Generating Figure 2 (single vs multi-chain)...")
let
    fig = Figure(size = (900, 400))

    colors_bk1 = rank_colors(RA_bk1)
    order_bk1 = sortperm(RA_bk1)
    colors_bk = rank_colors(RA_bk)
    order_bk = sortperm(RA_bk)

    # (a) single chain
    ax1 = Axis(fig[1, 1],
        xlabel = L"f_1", ylabel = L"f_2",
        title = "(a)  Single chain")
    scatter!(ax1, EC_bk1[1, order_bk1], EC_bk1[2, order_bk1],
        color = colors_bk1[order_bk1],
        markersize = [RA_bk1[i] == 0 ? MARKER_SIZE_PARETO : MARKER_SIZE_NEAR for i in order_bk1])

    # (b) 10-chain parallel
    ax2 = Axis(fig[1, 2],
        xlabel = L"f_1", ylabel = L"f_2",
        title = "(b)  Ten chains (parallel)")
    scatter!(ax2, EC_bk[1, order_bk], EC_bk[2, order_bk],
        color = colors_bk[order_bk],
        markersize = [RA_bk[i] == 0 ? MARKER_SIZE_PARETO : MARKER_SIZE_NEAR for i in order_bk])

    linkaxes!(ax1, ax2)

    # legend
    elem_pareto = MarkerElement(color = RGBAf(0.1, 0.1, 0.1, 0.8), marker = :circle,
        markersize = MARKER_SIZE_PARETO)
    elem_near_lo = MarkerElement(color = RGBAf(0.15, 0.35, 0.85, 0.75), marker = :circle,
        markersize = MARKER_SIZE_NEAR)
    elem_near_hi = MarkerElement(color = RGBAf(0.6, 0.7, 0.7, 0.5), marker = :circle,
        markersize = MARKER_SIZE_NEAR)
    Legend(fig[1, 3],
        [elem_pareto, elem_near_lo, elem_near_hi],
        ["Rank = 0", "Low rank", "High rank"],
        framevisible = false)

    save(joinpath(FIGDIR, "fig_parallel.pdf"), fig)
    println("  Saved fig_parallel.pdf")
end

println("\nAll figures generated in $(FIGDIR)")
