# Cell-free gene expression ensemble estimation — REAL experimental data
# P70-deGFP circuit from Adhikari et al. (2020), Frontiers in Bioengineering
#
# Two ODEs (mRNA, protein), two objectives (mRNA error, protein error).
# Demonstrates how ParetoEnsembles.jl generates parameter ensembles
# that trade off between fitting different experimental measurements.
#
# Run from the paper/code directory:
#   julia --project -t4 example_cellfree.jl

using ParetoEnsembles
using CairoMakie
using CSV
using DataFrames
using Random
using Statistics

const FIGDIR = joinpath(@__DIR__, "..", "figures")

# ──────────────────────────────────────────────────────────────
# Load experimental data (Adhikari et al. 2020)
# ──────────────────────────────────────────────────────────────
data = CSV.read(joinpath(@__DIR__, "data", "P70_deGFP_data.csv"), DataFrame)
const T_OBS = data.time_hr
const m_data = data.mean_mRNA_nM
const m_err = data.stdev_mRNA_nM
const p_data_uM = data.mean_GFP_uM
const p_err_uM = data.stdev_GFP_uM

# Convert protein to nM for model consistency
const p_data = p_data_uM .* 1000.0
const p_err = p_err_uM .* 1000.0

println("Experimental data loaded: $(length(T_OBS)) time points")
println("  mRNA range: $(round(minimum(m_data), digits=1)) – $(round(maximum(m_data), digits=1)) nM")
println("  Protein range: $(round(minimum(p_data_uM), digits=2)) – $(round(maximum(p_data_uM), digits=2)) μM")

# ──────────────────────────────────────────────────────────────
# Model: simplified cell-free TX/TL for deGFP
#
#   dm/dt = α · u(σ70) − δm · m           (mRNA balance)
#   dp/dt = κ · m · w(t) − δp · p          (protein balance)
#
#   u(σ70) = σ70^n / (K^n + σ70^n)         (promoter activity)
#   w(t)   = exp(−ln2 · t / τ_half)        (translation capacity decay)
#
# State: [m, p] in nM.  Time in hours.
# ──────────────────────────────────────────────────────────────
const σ70    = 35.0    # nM (endogenous, fixed)
const N_HILL = 1.5     # Hill coefficient (fixed)
const δp     = 0.005   # h⁻¹, protein degradation (very slow, fixed)

# Parameter vector: [α, κ, δm, K, τ_half]
const PARAM_NAMES = ["α", "κ", "δm", "K", "τ_half"]
const P_LOWER = [500.0,  0.5,  0.1,  1.0,  1.0]
const P_UPPER = [10000.0, 30.0, 15.0, 150.0, 12.0]
const N_PARAMS = 5

# ──────────────────────────────────────────────────────────────
# ODE right-hand side and simple RK4 integrator
# ──────────────────────────────────────────────────────────────
function rhs(m, p, params, t)
    α, κ, δm, K, τ_half = params
    u = σ70^N_HILL / (K^N_HILL + σ70^N_HILL)
    w = exp(-0.693 * t / τ_half)
    dm = α * u - δm * m
    dp = κ * m * w - δp * p
    return dm, dp
end

function simulate(params; tspan=(0.0, 16.0), dt=0.005)
    ts, te = tspan
    times = collect(ts:dt:te)
    n = length(times)
    m = zeros(n)
    p = zeros(n)
    for i in 1:(n-1)
        t = times[i]
        mi, pi = m[i], p[i]
        k1m, k1p = rhs(mi, pi, params, t)
        k2m, k2p = rhs(mi + 0.5dt*k1m, pi + 0.5dt*k1p, params, t + 0.5dt)
        k3m, k3p = rhs(mi + 0.5dt*k2m, pi + 0.5dt*k2p, params, t + 0.5dt)
        k4m, k4p = rhs(mi + dt*k3m, pi + dt*k3p, params, t + dt)
        m[i+1] = max(0.0, mi + (dt/6)*(k1m + 2k2m + 2k3m + k4m))
        p[i+1] = max(0.0, pi + (dt/6)*(k1p + 2k2p + 2k3p + k4p))
    end
    return times, m, p
end

function sim_at_times(params, t_obs)
    times, m_sim, p_sim = simulate(params)
    m_out = zeros(length(t_obs))
    p_out = zeros(length(t_obs))
    for (k, t_target) in enumerate(t_obs)
        idx = argmin(abs.(times .- t_target))
        m_out[k] = m_sim[idx]
        p_out[k] = p_sim[idx]
    end
    return m_out, p_out
end

# ──────────────────────────────────────────────────────────────
# ParetoEnsembles callbacks
# ──────────────────────────────────────────────────────────────
function objective_function(x)
    f = zeros(2, 1)
    try
        m_sim, p_sim = sim_at_times(x, T_OBS)
        for i in eachindex(T_OBS)
            # stdev-weighted SSE for mRNA
            if m_err[i] > 0
                f[1] += ((m_sim[i] - m_data[i]) / m_err[i])^2
            end
            # stdev-weighted SSE for protein
            if p_err[i] > 0
                f[2] += ((p_sim[i] - p_data[i]) / p_err[i])^2
            end
        end
    catch
        f[1] = 1e6
        f[2] = 1e6
    end
    return f
end

function neighbor_function(x)
    new_x = x .* (1.0 .+ 0.08 .* randn(N_PARAMS))
    return clamp.(new_x, P_LOWER, P_UPPER)
end

acceptance_probability_function(R, T) = exp(-R[end] / T)
cooling_function(T) = 0.9 * T

# ──────────────────────────────────────────────────────────────
# Run ensemble estimation (10 parallel chains)
# ──────────────────────────────────────────────────────────────
println("\nRunning ensemble estimation (10 chains)...")
initial_states = [P_LOWER .+ rand(MersenneTwister(s), N_PARAMS) .* (P_UPPER .- P_LOWER) for s in 1:10]

(EC, PC, RA) = estimate_ensemble_parallel(
    objective_function, neighbor_function,
    acceptance_probability_function, cooling_function,
    initial_states;
    rank_cutoff = 8,
    maximum_number_of_iterations = 50,
    maximum_archive_size = 2000,
    show_trace = false,
    rng_seed = 42
)

n_total = size(EC, 2)
n_pareto = count(RA .== 0)
println("  Total solutions: $n_total")
println("  Pareto-optimal: $n_pareto")
println("  Near-optimal: $(n_total - n_pareto)")

# Compute hypervolume
ref_point = [maximum(EC[1,:]) * 1.1, maximum(EC[2,:]) * 1.1]
front_E, _ = pareto_front(EC, PC, RA)
hv = hypervolume(front_E, ref_point)
println("  Hypervolume (front): $(round(hv, digits=2))")

# ──────────────────────────────────────────────────────────────
# Select ensemble: rank ≤ 1 (Pareto front + first dominated layer)
# ──────────────────────────────────────────────────────────────
ensemble_idx = findall(RA .<= 1)
n_ens = length(ensemble_idx)
println("  Ensemble size (rank ≤ 1): $n_ens")

# Simulate all ensemble members on a fine time grid
t_fine = collect(0.0:0.05:16.0)
n_fine = length(t_fine)
M_ens = zeros(n_ens, n_fine)
P_ens = zeros(n_ens, n_fine)

for (k, idx) in enumerate(ensemble_idx)
    params_k = PC[:, idx]
    times, m_sim, p_sim = simulate(params_k)
    for j in 1:n_fine
        ii = argmin(abs.(times .- t_fine[j]))
        M_ens[k, j] = m_sim[ii]
        P_ens[k, j] = p_sim[ii]
    end
end

m_mean = vec(mean(M_ens, dims=1))
m_lo   = vec(mapslices(x -> quantile(x, 0.025), M_ens, dims=1))
m_hi   = vec(mapslices(x -> quantile(x, 0.975), M_ens, dims=1))

p_mean = vec(mean(P_ens, dims=1))
p_lo   = vec(mapslices(x -> quantile(x, 0.025), P_ens, dims=1))
p_hi   = vec(mapslices(x -> quantile(x, 0.975), P_ens, dims=1))

# ──────────────────────────────────────────────────────────────
# Figure: 3-panel cell-free ensemble results
# ──────────────────────────────────────────────────────────────
include("paper_theme.jl")
set_paper_theme!()

println("\nGenerating cell-free figure...")
let
    fig = Figure(size = (1100, 400))

    # --- (a) mRNA ---
    ax_a = Axis(fig[1, 1],
        xlabel = "Time (h)", ylabel = "mRNA deGFP (nM)",
        title = "(a)  mRNA")

    band!(ax_a, t_fine, m_lo, m_hi, color = C_ENSEMBLE_FILL)
    lines!(ax_a, t_fine, m_mean, color = C_MEAN, linewidth = 2, linestyle = :dash)
    errorbars!(ax_a, T_OBS, m_data, m_err, color = C_DATA, whiskerwidth = 5)
    scatter!(ax_a, T_OBS, m_data, color = C_DATA, markersize = 7)

    # --- (b) protein ---
    ax_b = Axis(fig[1, 2],
        xlabel = "Time (h)", ylabel = "Protein deGFP (\u00b5M)",
        title = "(b)  Protein")

    band!(ax_b, t_fine, p_lo ./ 1000, p_hi ./ 1000, color = C_ENSEMBLE_FILL)
    lines!(ax_b, t_fine, p_mean ./ 1000, color = C_MEAN, linewidth = 2, linestyle = :dash)
    errorbars!(ax_b, T_OBS, p_data_uM, p_err_uM, color = C_DATA, whiskerwidth = 5)
    scatter!(ax_b, T_OBS, p_data_uM, color = C_DATA, markersize = 7)

    # --- (c) Pareto front (log scale to prevent smashing into origin) ---
    ax_c = Axis(fig[1, 3],
        xlabel = "log\u2081\u2080(\u03b5 mRNA)", ylabel = "log\u2081\u2080(\u03b5 protein)",
        title = "(c)  Pareto front")

    # Log-transform objectives for visibility
    log_e1 = log10.(EC[1, :] .+ 1e-10)
    log_e2 = log10.(EC[2, :] .+ 1e-10)

    p_idx = RA .== 0
    n_idx = .!p_idx
    scatter!(ax_c, log_e1[n_idx], log_e2[n_idx],
        color = C_ENSEMBLE, markersize = 4)
    scatter!(ax_c, log_e1[p_idx], log_e2[p_idx],
        color = C_FRONT, markersize = 6)

    # legend
    elem_band = PolyElement(color = C_ENSEMBLE_FILL)
    elem_mean = LineElement(color = C_MEAN, linewidth = 2, linestyle = :dash)
    elem_data = MarkerElement(color = C_DATA, marker = :circle, markersize = 7)
    elem_pareto = MarkerElement(color = C_FRONT, marker = :circle, markersize = 6)
    elem_near = MarkerElement(color = C_ENSEMBLE, marker = :circle, markersize = 4)
    Legend(fig[2, 1:3],
        [elem_data, elem_mean, elem_band, elem_pareto, elem_near],
        ["Experimental data", "Ensemble mean", "95% CI", "Rank = 0", "Near-optimal"],
        orientation = :horizontal, tellwidth = false, tellheight = true)

    save(joinpath(FIGDIR, "fig_cellfree.pdf"), fig)
    println("  Saved fig_cellfree.pdf")
end

println("\nDone!")
