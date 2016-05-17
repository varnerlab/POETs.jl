using POETs
using PyPlot
include("binh_korn_function.jl")

(EC,PC,RA) = estimate_ensemble(objective_function,neighbor_function,acceptance_probability_function,cooling_function,2.0.*(1+0.5*randn(2));rank_cutoff=4,maximum_number_of_iterations=40)
