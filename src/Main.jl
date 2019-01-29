#=
 * Copyright (c) 2016. Varnerlab,
 * School of Chemical and Biomolecular Engineering,
 * Cornell University, Ithaca NY 14853
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * Created by jeffreyvarner on 5/4/2016
 =#

function estimate_ensemble(objective_function::Function,neighbor_function::Function,acceptance_probability_function::Function,cooling_function::Function,
  initial_state::Array{Float64,1};maximum_number_of_iterations=20,rank_cutoff=5.0,temperature_min=0.0001,show_trace=true)

  # internal parameters -
  temperature = 1.0

  # Grab the initial parameters -
  parameter_array_best = initial_state

  # initialize error cache -
  error_cache = objective_function(parameter_array_best)

  # initialize parameter_cache -
  parameter_cache = parameter_array_best

  # initialize the Pareto rank array from the error_cache -
  pareto_rank_array = rank_function(error_cache)

  # how many objectives do we have?
  number_of_objectives = length(error_cache)

  # main loop -
  while (temperature>temperature_min)

    should_loop_continue::Bool = true;
    iteration_index = 1
    while (should_loop_continue)

      # generate a new solution -
      test_parameter_array = neighbor_function(parameter_array_best)

      # evaluate the new solution -
      test_error = objective_function(test_parameter_array)

      # Add the test error to the error cache -
      error_cache = [error_cache test_error]

      # Add parameters to parameter cache -
      parameter_cache = [parameter_cache test_parameter_array]

      # compute the Pareto rank for the error_cache -
      pareto_rank_array = rank_function(error_cache)

      # do we accept the new solution?
      acceptance_probability = acceptance_probability_function(pareto_rank_array,temperature)
      if (acceptance_probability>rand())

        # Select the rank -
        #archive_select_index = find(pareto_rank_array.<rank_cutoff)
	archive_select_index = findall(pareto_rank_array.<rank_cutoff)

        # update the caches -
        error_cache = error_cache[:,archive_select_index]
        parameter_cache = parameter_cache[:,archive_select_index]

        # update the parameters -
        parameter_array_best = test_parameter_array

        if (show_trace == true)
          @show iteration_index,temperature
        end
      end

      # check - should we go around again?
      if (iteration_index>maximum_number_of_iterations)
        should_loop_continue = false;
      end;

      # update the loop index -
      iteration_index+=1;
    end # end: inner while-loop (iterations)

    # update the temperature -
    temperature = cooling_function(temperature)

  end # end: outer while-loop (temperature)

  # return the caches -
  return (error_cache,parameter_cache,pareto_rank_array)
end

# Pareto ranking function -
function rank_function(error_cache)

  # Get the size of the error cache -
  (number_of_objectives,number_of_trials) = size(error_cache)

  # initialize -
  rank_array = zeros(number_of_trials)

  # Setup the index vectors -
  trial_index_array = collect(1:number_of_trials)
  objective_index_array = collect(1:number_of_objectives)

  # main rank loop -
  for trial_index in trial_index_array

    # initial dominated population -
    dominated_population_array = collect(1:number_of_trials);
    for objective_index in objective_index_array

      # index of dominated -
	#index_of_dominated_sets = find(error_cache[objective_index,dominated_population_array].<=error_cache[objective_index,trial_index])
      index_of_dominated_sets = findall(error_cache[objective_index,dominated_population_array].<=error_cache[objective_index,trial_index])

      # update -
      dominated_population_array = dominated_population_array[index_of_dominated_sets]
    end

    # update -
    rank_array[trial_index] = length(dominated_population_array) - 1
  end

  return rank_array
end
