# Head-to-head comparison: ParetoEnsembles.jl vs NSGA-II (Metaheuristics.jl)
# on Binh-Korn and Fonseca-Fleming benchmarks
#
# Computes hypervolume and IGD at matched function evaluation budgets.
#
# Run from the paper/code directory:
#   julia --project -t4 comparison_study.jl

using ParetoEnsembles
import ParetoEnsembles: pareto_front as pe_pareto_front, hypervolume as pe_hypervolume
using Metaheuristics
import Metaheuristics: pareto_front as nsga_pareto_front
using CairoMakie
using JLD2
using Random
using Statistics

const FIGDIR = joinpath(@__DIR__, "..", "figures")
const CACHEDIR = joinpath(@__DIR__, "data")
const CACHE_FILE = joinpath(CACHEDIR, "comparison_results.jld2")
const N_REPS = 5

# ──────────────────────────────────────────────────────────────
# Utility: Inverted Generational Distance
# ──────────────────────────────────────────────────────────────
function igd(approx::AbstractMatrix, reference::AbstractMatrix)
    # approx: 2 x n_approx, reference: 2 x n_ref
    total = 0.0
    for j in 1:size(reference, 2)
        min_dist = Inf
        for i in 1:size(approx, 2)
            d = sqrt((approx[1,i] - reference[1,j])^2 + (approx[2,i] - reference[2,j])^2)
            min_dist = min(min_dist, d)
        end
        total += min_dist
    end
    return total / size(reference, 2)
end

# ──────────────────────────────────────────────────────────────
# Binh-Korn problem definitions
# ──────────────────────────────────────────────────────────────
function bk_objective_pe(x)
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

# NSGA-II formulation
function bk_objective_nsga(x)
    f1 = 4.0 * x[1]^2 + 4.0 * x[2]^2
    f2 = (x[1] - 5)^2 + (x[2] - 5)^2
    g1 = (x[1] - 5)^2 + x[2]^2 - 25       # <= 0
    g2 = 7.7 - (x[1] - 8)^2 - (x[2] - 3)^2 # <= 0
    return [f1, f2], [g1, g2], [0.0]
end

# ──────────────────────────────────────────────────────────────
# Fonseca-Fleming problem definitions
# ──────────────────────────────────────────────────────────────
const FF_D = 3

function ff_objective_pe(x)
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

function ff_objective_nsga(x)
    sum1 = sum((x[i] - 1 / sqrt(FF_D))^2 for i in 1:FF_D)
    sum2 = sum((x[i] + 1 / sqrt(FF_D))^2 for i in 1:FF_D)
    return [1 - exp(-sum1), 1 - exp(-sum2)], [0.0], [0.0]
end

# ──────────────────────────────────────────────────────────────
# Reference fronts
# ──────────────────────────────────────────────────────────────
# Fonseca-Fleming: analytical front t ∈ [-1, 1]
ff_ref_t = range(-1, 1, length=1000)
ff_ref_front = hcat(
    [1 - exp(-(t - 1)^2) for t in ff_ref_t],
    [1 - exp(-(t + 1)^2) for t in ff_ref_t]
)'  # 2 x 1000

# Binh-Korn: sample approximate front (from a high-budget run)
println("Generating Binh-Korn reference front...")
bk_starts = [[x1, x2] for (x1, x2) in zip(range(0.1, 4.5, length=20), range(0.1, 2.8, length=20))]
(EC_ref, _, RA_ref) = estimate_ensemble_parallel(
    bk_objective_pe, bk_neighbor, accept_fn, cool_fn, bk_starts;
    rank_cutoff=12, maximum_number_of_iterations=100,
    show_trace=false, rng_seed=999)
bk_ref_front, _ = pe_pareto_front(EC_ref, EC_ref, RA_ref)
println("  BK reference front: $(size(bk_ref_front, 2)) points")

# Reference points for hypervolume
const BK_HV_REF = [200.0, 50.0]
const FF_HV_REF = [1.05, 1.05]

# ──────────────────────────────────────────────────────────────
# ParetoEnsembles settings (matched to paper)
# ──────────────────────────────────────────────────────────────
const PE_N_CHAINS = 10
const PE_N_ITER = 60
const PE_RANK_CUTOFF = 12
const PE_ALPHA = 0.95

# Approximate function evaluations per chain:
# temp steps ≈ ceil(log(0.0001)/log(0.95)) = 181
# evals per chain ≈ 181 * 61 ≈ 11,041
# total ≈ 110,410 for 10 chains
const APPROX_EVALS = 110_000

# NSGA-II: match budget with pop=200, gen=550 → 110,000
const NSGA_POP = 200
const NSGA_GEN = APPROX_EVALS ÷ NSGA_POP

println("\nFunction evaluation budget: ~$(APPROX_EVALS)")
println("NSGA-II: pop=$(NSGA_POP), gen=$(NSGA_GEN)")

# ──────────────────────────────────────────────────────────────
# Run comparison
# ──────────────────────────────────────────────────────────────
results = Dict{String, Vector{NamedTuple}}()

for (problem_name, pe_obj, pe_neigh, nsga_obj, bounds_lo, bounds_hi, hv_ref, ref_front) in [
    ("Binh-Korn", bk_objective_pe, bk_neighbor, bk_objective_nsga,
     [0.0, 0.0], [5.0, 3.0], BK_HV_REF, bk_ref_front),
    ("Fonseca-Fleming", ff_objective_pe, ff_neighbor, ff_objective_nsga,
     fill(-4.0, FF_D), fill(4.0, FF_D), FF_HV_REF, ff_ref_front),
]
    println("\n=== $problem_name ===")
    reps = NamedTuple[]

    for rep in 1:N_REPS
        println("  Rep $rep/$N_REPS...")

        # --- ParetoEnsembles ---
        if problem_name == "Binh-Korn"
            starts = [[x1, x2] for (x1, x2) in zip(
                range(0.1, 4.5, length=PE_N_CHAINS),
                range(0.1, 2.8, length=PE_N_CHAINS))]
        else
            starts = [randn(MersenneTwister(hash((rep, s))), FF_D) for s in 1:PE_N_CHAINS]
        end

        t_pe = @elapsed (EC_pe, PC_pe, RA_pe) = estimate_ensemble_parallel(
            pe_obj, pe_neigh, accept_fn, cool_fn, starts;
            rank_cutoff=PE_RANK_CUTOFF, maximum_number_of_iterations=PE_N_ITER,
            show_trace=false, rng_seed=rep)

        front_pe, _ = pe_pareto_front(EC_pe, PC_pe, RA_pe)
        hv_pe = pe_hypervolume(front_pe, hv_ref)
        igd_pe = igd(front_pe, ref_front)

        # --- NSGA-II ---
        bounds = BoxConstrainedSpace(bounds_lo, bounds_hi)
        nsga_alg = NSGA2(N=NSGA_POP, p_m=0.1)
        nsga_alg.options.iterations = NSGA_GEN
        nsga_alg.options.seed = rep
        t_nsga = @elapsed result_nsga = Metaheuristics.optimize(
            nsga_obj, bounds, nsga_alg)

        # Extract NSGA-II front (returns n_solutions × n_objectives matrix)
        nsga_pf = nsga_pareto_front(result_nsga)
        nsga_objs = collect(nsga_pf')  # transpose to 2 x n
        hv_nsga = pe_hypervolume(nsga_objs, hv_ref)
        igd_nsga = igd(nsga_objs, ref_front)

        push!(reps, (
            hv_pe=hv_pe, hv_nsga=hv_nsga,
            igd_pe=igd_pe, igd_nsga=igd_nsga,
            t_pe=t_pe, t_nsga=t_nsga,
            n_front_pe=size(front_pe, 2), n_front_nsga=size(nsga_objs, 2),
            EC_pe=EC_pe, RA_pe=RA_pe, nsga_objs=nsga_objs
        ))
    end
    results[problem_name] = reps
end

# ──────────────────────────────────────────────────────────────
# Print comparison table
# ──────────────────────────────────────────────────────────────
println("\n" * "="^80)
println("COMPARISON RESULTS (median ± IQR over $N_REPS replicates)")
println("="^80)

for problem_name in ["Binh-Korn", "Fonseca-Fleming"]
    reps = results[problem_name]
    println("\n--- $problem_name ---")

    for (label, hv_key, igd_key, t_key, n_key) in [
        ("ParetoEnsembles", :hv_pe, :igd_pe, :t_pe, :n_front_pe),
        ("NSGA-II", :hv_nsga, :igd_nsga, :t_nsga, :n_front_nsga)]

        hvs = [r[hv_key] for r in reps]
        igds = [r[igd_key] for r in reps]
        ts = [r[t_key] for r in reps]
        ns = [r[n_key] for r in reps]

        println("  $label:")
        println("    HV:   $(round(median(hvs), digits=2)) [$(round(quantile(hvs, 0.25), digits=2)) – $(round(quantile(hvs, 0.75), digits=2))]")
        println("    IGD:  $(round(median(igds), sigdigits=3)) [$(round(quantile(igds, 0.25), sigdigits=3)) – $(round(quantile(igds, 0.75), sigdigits=3))]")
        println("    Time: $(round(median(ts), digits=2))s")
        println("    Front size: $(round(Int, median(ns)))")
    end
end

# ──────────────────────────────────────────────────────────────
# Figure: 1x2 comparison — both solvers overlaid on same axes
# ──────────────────────────────────────────────────────────────
include("paper_theme.jl")
set_paper_theme!()

mkpath(CACHEDIR)
@save CACHE_FILE results bk_ref_front ff_ref_front BK_HV_REF FF_HV_REF N_REPS

println("\nGenerating comparison figure...")
let
    fig = Figure(size = (900, 420))
    bk = results["Binh-Korn"][end]
    ff = results["Fonseca-Fleming"][end]

    C_PE_PT = RGBAf(0.20, 0.45, 0.78, 0.35)
    C_NSGA_PT = RGBAf(0.85, 0.35, 0.10, 0.60)

    # --- (a) Binh-Korn: both solvers overlaid ---
    ax_a = Axis(fig[1, 1], xlabel = "f\u2081", ylabel = "f\u2082",
        title = "(a)  Binh-Korn")

    # PE near-optimal cloud (light blue)
    p_idx = bk.RA_pe .== 0
    scatter!(ax_a, bk.EC_pe[1, .!p_idx], bk.EC_pe[2, .!p_idx],
        color = C_PE_PT, markersize = 2)
    # PE front (solid blue)
    scatter!(ax_a, bk.EC_pe[1, p_idx], bk.EC_pe[2, p_idx],
        color = C_PE, markersize = 3)
    # NSGA-II front (orange)
    scatter!(ax_a, bk.nsga_objs[1,:], bk.nsga_objs[2,:],
        color = C_NSGA, markersize = 5, marker = :diamond)

    # --- (b) Fonseca-Fleming: both solvers overlaid ---
    ax_b = Axis(fig[1, 2], xlabel = "f\u2081", ylabel = "f\u2082",
        title = "(b)  Fonseca-Fleming")

    p_idx_ff = ff.RA_pe .== 0
    scatter!(ax_b, ff.EC_pe[1, .!p_idx_ff], ff.EC_pe[2, .!p_idx_ff],
        color = C_PE_PT, markersize = 2)
    scatter!(ax_b, ff.EC_pe[1, p_idx_ff], ff.EC_pe[2, p_idx_ff],
        color = C_PE, markersize = 3)
    scatter!(ax_b, ff.nsga_objs[1,:], ff.nsga_objs[2,:],
        color = C_NSGA, markersize = 5, marker = :diamond)

    # Theoretical front (plotted last so it's visible on top)
    lines!(ax_b, ff_ref_front[1,:], ff_ref_front[2,:],
        color = (C_THEORY, 0.5), linewidth = 2, linestyle = :dash)

    # Shared legend
    elem_pe_front = MarkerElement(color = C_PE, marker = :circle, markersize = 5)
    elem_pe_near = MarkerElement(color = C_PE_PT, marker = :circle, markersize = 4)
    elem_nsga = MarkerElement(color = C_NSGA, marker = :diamond, markersize = 6)
    elem_theo = LineElement(color = (C_THEORY, 0.5), linewidth = 2, linestyle = :dash)
    Legend(fig[2, 1:2],
        [elem_pe_front, elem_pe_near, elem_nsga, elem_theo],
        ["ParetoEnsembles (front)", "ParetoEnsembles (near-optimal)", "NSGA-II (front)", "Theoretical front"],
        orientation = :horizontal, tellwidth = false, tellheight = true)

    save(joinpath(FIGDIR, "fig_comparison.pdf"), fig)
    println("  Saved fig_comparison.pdf")
end

println("\nDone!")
