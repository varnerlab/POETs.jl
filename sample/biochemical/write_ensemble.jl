using POETs

# Script to write the model ensemble to disk -
ec_array = ec_array[:,2:end]
pc_array = pc_array[:,2:end]

# Re-rank -
rank_array = rank_function(ec_array)

# Write -
writedlm("./data/ec_array.dat",ec_array)
writedlm("./data/pc_array.dat",pc_array)
writedlm("./data/rank_array.dat",rank_array)

# Take the *best* value from the current ensemble, refine it and go around again -
total_error = sum(ec_array[:,2:end],1)

# Which col is the min error?
min_index = indmin(total_error)
initial_parameter_array = pc_array[:,min_index]
writedlm("./data/initial_parameter_array.dat",initial_parameter_array)
