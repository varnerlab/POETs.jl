# Run the model for the ensemble of solutions -
include("SolveBalances.jl")
using DelimitedFiles
using Statistics

# Simple linear interpolation -
function linear_interp(xq, x, y)
  n = length(xq)
  result = similar(xq, Float64)
  for i in 1:n
    xi = xq[i]
    if xi <= x[1]
      result[i] = y[1]
    elseif xi >= x[end]
      result[i] = y[end]
    else
      j = searchsortedlast(x, xi)
      t = (xi - x[j]) / (x[j+1] - x[j])
      result[i] = y[j] + t * (y[j+1] - y[j])
    end
  end
  return result
end

# Load the ensemble files from disk -
pc_array_full = readdlm("./data/pc_array.dat")
rank_array = readdlm("./data/rank_array.dat")

# Select the desired rank -
idx_rank = findall(vec(rank_array) .<= 0.0)

# Setup time scale -
tStart = 0.0
tStop = 100.0
tStep = 0.1
number_of_timesteps = 200
time_experimental = collect(range(tStart, stop=tStop, length=number_of_timesteps))

# initialize data array -
data_array = zeros(length(time_experimental), 1)

# Run the simulation for this rank -
number_of_samples = 1
for (index, rank_index_value) in enumerate(idx_rank)

  if mod(index, 20) == 0

    # Grab the parameter set from the cache -
    parameter_array = pc_array_full[:, rank_index_value]

    # Run the model -
    (t, x) = SolveBalances(tStart, tStop, tStep, parameter_array)

    # Need to interpolate the simulation onto the experimental time scale -
    CI = linear_interp(time_experimental, t, x[:, 9])

    # grab -
    data_array = hcat(data_array, CI)

    # update the sample count -
    number_of_samples += 1
  end
end

# calculate mean, and std
mean_value = mean(data_array[:, 2:end], dims=2)
std_value = std(data_array[:, 2:end], dims=2)
SF = 2.58 / sqrt(number_of_samples)

UB = mean_value .+ SF .* std_value
LB = mean_value .- SF .* std_value
LB[LB .< 0] .= 0.0

# Plot results (requires Plots.jl or PyPlot.jl)
# using Plots
# plot(time_experimental, mean_value, ribbon=(mean_value .- LB, UB .- mean_value),
#      fillalpha=0.3, color=:gray, linewidth=2, label="Ensemble mean ± 99% CI")
