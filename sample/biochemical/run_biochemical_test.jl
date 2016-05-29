# Estimates model parameters for proof-of-concept biochemical model -
using POETs
include("hcmem_lib.jl")

function test_biochemical_model()

  number_of_subdivisions = 10
  number_of_parameters = 13
  number_of_objectives = 4

  # generate parameter guess -
  #initial_parameter_array = readdlm("./data/initial_parameter_array.dat")
  initial_parameter_array = zeros(13)
  initial_parameter_array[1:6]  = ones(6).*(1+0.25*randn(6))
  initial_parameter_array[7:12] = 2*ones(6).*(1+0.25*randn(6))
  initial_parameter_array[13] = (1+0.25*randn())

  ec_array = zeros(number_of_objectives)
  pc_array = zeros(number_of_parameters)
  for index in collect(1:number_of_subdivisions)

    # Run JuPOETs -
    (EC,PC,RA) = estimate_ensemble(objective_function,neighbor_function,acceptance_probability_function,cooling_function,initial_parameter_array;rank_cutoff=4,maximum_number_of_iterations=20,show_trace=true)

    # Package -
    ec_array = [ec_array EC]
    pc_array = [pc_array PC]

    # Take the *best* value from the current ensemble, refine it and go around again -
    total_error = sum(ec_array[:,2:end],1)

    # Which col is the min error?
    min_index = indmin(total_error)
    @show index,total_error[min_index]

    # Refine the best solution -
    initial_parameter_array = pc_array[:,min_index].*(1+0.15*randn(number_of_parameters))
    initial_parameter_array = local_refienment_step(initial_parameter_array)
  end

  return (ec_array,pc_array)
end
