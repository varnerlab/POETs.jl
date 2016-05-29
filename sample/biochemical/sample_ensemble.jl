# Run the model for the ensemble of solutions -
include("SolveBalances.jl")
using PyPlot
using PyCall
@pyimport numpy as np


# Load the ensemble files from disk -
pc_array_full = readdlm("./data/pc_array.dat")
rank_array = readdlm("./data/rank_array.dat")

# Select the desired rank -
idx_rank = find(rank_array .<= 0.0)

# Setup time scale -
tStart = 0.0;
tStop = 100.0;
tStep = 0.1;
number_of_timesteps = 200
time_experimental = linspace(tStart,tStop,number_of_timesteps)

# initialize data array -
data_array = zeros(length(time_experimental),1)

# Run the simulation for this rank -
number_of_samples = 1
for (index,rank_index_value) in enumerate(idx_rank)

  if (mod(index,20)==0)

    # Grab the parameter set from the cache -
    parameter_array = pc_array_full[:,rank_index_value]

    # Run the model -
    (t,x) = SolveBalances(tStart,tStop,tStep,parameter_array)

    # Need to interpolate the simulation onto the experimental time scale -
    AI = np.interp(time_experimental[:],t,x[:,7])
    BI = np.interp(time_experimental[:],t,x[:,8])
    CI = np.interp(time_experimental[:],t,x[:,9])
    XI = np.interp(time_experimental[:],t,x[:,10])

    # grab -
    data_array = [data_array CI]

    # update the sample count -
    number_of_samples = number_of_samples + 1
  end
end

# calculate mean, and std
mean_value = mean(data_array,2)
std_value = std(data_array,2)
SF = (2.58/sqrt(number_of_samples))

UB = mean_value + (SF)*std_value
LB = mean_value - (SF)*std_value
idx_z = find(LB.<0)
LB[idx_z] = 0.0

# Make the plot -
#plot(time_experimental,mean_value,"k")
fill_between(vec(time_experimental),vec(LB),vec(UB),color="gray",lw=2)
