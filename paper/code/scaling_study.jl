# Scaling study: wall clock time vs thread count for cell-free model
# Run from paper/code directory with varying thread counts:
#   julia --project -t1 scaling_study.jl
#   julia --project -t2 scaling_study.jl
#   julia --project -t4 scaling_study.jl
#   julia --project -t8 scaling_study.jl
#
# Outputs a single CSV line: n_threads, median_time_s, n_trials

using ParetoEnsembles
using Random
using Statistics

# ── Cell-free model (same as example_cellfree.jl) ──────────────
const σ70    = 35.0
const N_HILL = 1.5
const δp     = 0.005
const P_LOWER = [500.0,  1.0,  0.5,  5.0,  1.0]
const P_UPPER = [8000.0, 20.0, 10.0, 100.0, 10.0]
const P_TRUE  = [2800.0, 5.3, 3.0, 25.0, 4.0]
const N_PARAMS = 5
const T_OBS = [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]

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

# Generate synthetic data
Random.seed!(2020)
m_true, p_true = sim_at_times(P_TRUE, T_OBS)
const CV = 0.12
m_data = m_true .* (1.0 .+ CV .* randn(length(T_OBS)))
p_data = p_true .* (1.0 .+ CV .* randn(length(T_OBS)))
m_data[1] = 0.0; p_data[1] = 0.0
m_data .= max.(m_data, 0.0)
p_data .= max.(p_data, 0.0)

function objective_function(x)
    f = zeros(2, 1)
    try
        m_sim, p_sim = sim_at_times(x, T_OBS)
        for i in eachindex(T_OBS)
            if m_data[i] > 0
                f[1] += ((m_sim[i] - m_data[i]) / max(m_data[i], 1.0))^2
            end
            if p_data[i] > 0
                f[2] += ((p_sim[i] - p_data[i]) / max(p_data[i], 1.0))^2
            end
        end
    catch
        f[1] = 1e6; f[2] = 1e6
    end
    return f
end

neighbor_function(x) = clamp.(x .* (1.0 .+ 0.08 .* randn(N_PARAMS)), P_LOWER, P_UPPER)
acceptance_probability_function(R, T) = exp(-R[end] / T)
cooling_function(T) = 0.9 * T

# ── Timing ─────────────────────────────────────────────────────
const N_CHAINS = 10
const N_ITER   = 50
const N_REPS   = 5  # number of timed repetitions

function make_starts(seed)
    [P_LOWER .+ rand(MersenneTwister(seed + s), N_PARAMS) .* (P_UPPER .- P_LOWER) for s in 1:N_CHAINS]
end

# Warmup (JIT compilation)
warmup_starts = make_starts(999)
estimate_ensemble_parallel(
    objective_function, neighbor_function,
    acceptance_probability_function, cooling_function,
    warmup_starts;
    rank_cutoff = 8, maximum_number_of_iterations = 10,
    maximum_archive_size = 500, show_trace = false, rng_seed = 0
)

# Timed runs
times_s = Float64[]
for rep in 1:N_REPS
    starts = make_starts(rep * 100)
    t = @elapsed estimate_ensemble_parallel(
        objective_function, neighbor_function,
        acceptance_probability_function, cooling_function,
        starts;
        rank_cutoff = 8, maximum_number_of_iterations = N_ITER,
        maximum_archive_size = 2000, show_trace = false, rng_seed = rep
    )
    push!(times_s, t)
end

med = median(times_s)
n_threads = Threads.nthreads()
println("$n_threads,$med,$N_REPS")
