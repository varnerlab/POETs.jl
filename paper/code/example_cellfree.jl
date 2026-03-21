# Cell-free gene expression ensemble estimation
# Simplified σ70 → P70 → deGFP circuit (C1) from Adhikari et al. (2020)
#
# Two ODEs (mRNA, protein), two objectives (mRNA error, protein error).
# Demonstrates how ParetoEnsembles.jl generates parameter ensembles
# that trade off between fitting different experimental measurements.
#
# Run from the paper/code directory:
#   julia --project -t4 example_cellfree.jl

using ParetoEnsembles
using CairoMakie
using Random
using Statistics

const FIGDIR = joinpath(@__DIR__, "..", "figures")

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
#   α      : effective max transcription rate (nM/h)
#   κ      : effective translation rate constant (h⁻¹)
#   δm     : mRNA degradation rate (h⁻¹)
#   K      : promoter dissociation constant (nM)
#   τ_half : translation capacity half-life (h)
const PARAM_NAMES = ["α", "κ", "δm", "K", "τ_half"]
const P_LOWER = [500.0,  1.0,  0.5,  5.0,  1.0]
const P_UPPER = [8000.0, 20.0, 10.0, 100.0, 10.0]
const N_PARAMS = 5

# "True" parameters (calibrated to match Adhikari et al. Figure 2)
const P_TRUE = [2800.0, 5.3, 3.0, 25.0, 4.0]

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

# Interpolate simulation at specific time points
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
# Generate synthetic experimental data
# Mimic realistic cell-free measurements: samples taken at
# discrete time points with ~10-15% coefficient of variation.
# ──────────────────────────────────────────────────────────────
const T_OBS = [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]

Random.seed!(2020)  # reproducible data
m_true, p_true = sim_at_times(P_TRUE, T_OBS)

# Add multiplicative Gaussian noise (CV ≈ 12%)
const CV = 0.12
m_data = m_true .* (1.0 .+ CV .* randn(length(T_OBS)))
p_data = p_true .* (1.0 .+ CV .* randn(length(T_OBS)))
m_data[1] = 0.0  # enforce zero IC
p_data[1] = 0.0
m_data .= max.(m_data, 0.0)
p_data .= max.(p_data, 0.0)

# Data error bars (standard error, 3 replicates)
m_err = max.(m_data .* CV ./ sqrt(3), 5.0)   # floor at 5 nM
p_err = max.(p_data .* CV ./ sqrt(3), 50.0)   # floor at 50 nM

println("Synthetic data generated at $(length(T_OBS)) time points")
println("  mRNA range: $(round(minimum(m_data), digits=1)) – $(round(maximum(m_data), digits=1)) nM")
println("  Protein range: $(round(minimum(p_data), digits=1)) – $(round(maximum(p_data)/1000, digits=1)) μM")

# ──────────────────────────────────────────────────────────────
# ParetoEnsembles callbacks
# ──────────────────────────────────────────────────────────────
# Objective: 2 objectives — normalized SSE for mRNA, normalized SSE for protein
function objective_function(x)
    f = zeros(2, 1)
    try
        m_sim, p_sim = sim_at_times(x, T_OBS)
        # Normalized sum of squared errors
        for i in eachindex(T_OBS)
            if m_data[i] > 0
                f[1] += ((m_sim[i] - m_data[i]) / max(m_data[i], 1.0))^2
            end
            if p_data[i] > 0
                f[2] += ((p_sim[i] - p_data[i]) / max(p_data[i], 1.0))^2
            end
        end
    catch
        f[1] = 1e6
        f[2] = 1e6
    end
    return f
end

# Neighbor: multiplicative Gaussian perturbation with bounds
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

# ──────────────────────────────────────────────────────────────
# Select ensemble: rank ≤ 1 (Pareto front + first dominated layer)
# ──────────────────────────────────────────────────────────────
ensemble_idx = findall(RA .<= 1)
n_ens = length(ensemble_idx)
println("  Ensemble size (rank ≤ 1): $n_ens")

# Simulate all ensemble members on a fine time grid
t_fine = collect(0.0:0.05:16.0)
n_fine = length(t_fine)
M_ens = zeros(n_ens, n_fine)  # mRNA trajectories
P_ens = zeros(n_ens, n_fine)  # protein trajectories

for (k, idx) in enumerate(ensemble_idx)
    params_k = PC[:, idx]
    times, m_sim, p_sim = simulate(params_k)
    for j in 1:n_fine
        ii = argmin(abs.(times .- t_fine[j]))
        M_ens[k, j] = m_sim[ii]
        P_ens[k, j] = p_sim[ii]
    end
end

# Compute statistics
m_mean = vec(mean(M_ens, dims=1))
m_lo   = vec(mapslices(x -> quantile(x, 0.025), M_ens, dims=1))
m_hi   = vec(mapslices(x -> quantile(x, 0.975), M_ens, dims=1))

p_mean = vec(mean(P_ens, dims=1))
p_lo   = vec(mapslices(x -> quantile(x, 0.025), P_ens, dims=1))
p_hi   = vec(mapslices(x -> quantile(x, 0.975), P_ens, dims=1))


# ──────────────────────────────────────────────────────────────
# Figure: 3-panel cell-free ensemble results
#   (a) mRNA ensemble vs data
#   (b) protein ensemble vs data
#   (c) Pareto front (ε_mRNA vs ε_protein)
# ──────────────────────────────────────────────────────────────
println("\nGenerating cell-free figure...")
let
    fig = Figure(size = (1100, 380), fontsize = 13)

    # --- (a) mRNA ---
    ax_a = Axis(fig[1, 1],
        xlabel = "Time (h)",
        ylabel = "mRNA deGFP (nM)",
        title = "(a)  mRNA")

    # 95% CI band
    band!(ax_a, t_fine, m_lo, m_hi, color = (:dodgerblue, 0.2))
    # ensemble mean
    lines!(ax_a, t_fine, m_mean, color = :dodgerblue, linewidth = 2, linestyle = :dash)
    # data with error bars
    errorbars!(ax_a, T_OBS, m_data, m_err, color = :black, whiskerwidth = 5)
    scatter!(ax_a, T_OBS, m_data, color = :black, markersize = 7)

    # --- (b) protein ---
    ax_b = Axis(fig[1, 2],
        xlabel = "Time (h)",
        ylabel = "Protein deGFP (μM)",
        title = "(b)  Protein")

    band!(ax_b, t_fine, p_lo ./ 1000, p_hi ./ 1000, color = (:dodgerblue, 0.2))
    lines!(ax_b, t_fine, p_mean ./ 1000, color = :dodgerblue, linewidth = 2, linestyle = :dash)
    errorbars!(ax_b, T_OBS, p_data ./ 1000, p_err ./ 1000, color = :black, whiskerwidth = 5)
    scatter!(ax_b, T_OBS, p_data ./ 1000, color = :black, markersize = 7)

    # --- (c) Pareto front (zoomed to region of interest) ---
    # Use 95th percentile of rank ≤ 1 solutions to set axis limits
    ens_e1 = EC[1, ensemble_idx]
    ens_e2 = EC[2, ensemble_idx]
    xlim_hi = quantile(ens_e1, 0.95) * 1.3
    ylim_hi = quantile(ens_e2, 0.95) * 1.3

    ax_c = Axis(fig[1, 3],
        xlabel = L"\varepsilon_{\mathrm{mRNA}}",
        ylabel = L"\varepsilon_{\mathrm{protein}}",
        title = "(c)  Pareto front",
        limits = ((-xlim_hi * 0.05, xlim_hi), (-ylim_hi * 0.05, ylim_hi)))

    # plot only solutions within the zoomed view
    p_idx = RA .== 0
    n_idx = .!p_idx
    scatter!(ax_c, EC[1, n_idx], EC[2, n_idx],
        color = (:gray65, 0.4), markersize = 4)
    scatter!(ax_c, EC[1, p_idx], EC[2, p_idx],
        color = :black, markersize = 5)

    # legend for time-series panels
    elem_band = PolyElement(color = (:dodgerblue, 0.2))
    elem_mean = LineElement(color = :dodgerblue, linewidth = 2, linestyle = :dash)
    elem_data = MarkerElement(color = :black, marker = :circle, markersize = 7)
    Legend(fig[2, 1:2],
        [elem_data, elem_mean, elem_band],
        ["Synthetic data", "Ensemble mean", "95% CI"],
        orientation = :horizontal, framevisible = false,
        tellwidth = false, tellheight = true)

    # legend for Pareto front
    elem_pareto = MarkerElement(color = :black, marker = :circle, markersize = 5)
    elem_near = MarkerElement(color = (:gray65, 0.4), marker = :circle, markersize = 4)
    Legend(fig[2, 3],
        [elem_pareto, elem_near],
        ["Rank = 0", "Near-optimal"],
        orientation = :horizontal, framevisible = false,
        tellwidth = false, tellheight = true)

    save(joinpath(FIGDIR, "fig_cellfree.pdf"), fig)
    println("  Saved fig_cellfree.pdf")
end

println("\nDone!")
