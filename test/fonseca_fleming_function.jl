using Debug

# some global parameters -
BIG = 1e10

# Evaluates the objective function values -
function objective_function(parameter_array)

  # Alias the species -
  number_of_parameters = length(parameter_array)

  # Calculate the objective function array -
  obj_array = BIG*ones(2,1)

  # calculate the objectibe array -
  # calculate the sums
  sum_1 = 0.0
  sum_2 = 0.0
  for index in collect(1:number_of_parameters)

    sum_1 = sum_1 + (parameter_array[index] - 1/sqrt(number_of_parameters))^2
    sum_2 = sum_2 + (parameter_array[index] + 1/sqrt(number_of_parameters))^2
  end

  # objectives -
  obj_array[1] = 1 - exp(-1*sum_1)
  obj_array[2] = 1 - exp(-1*sum_2)

  # return -
  return obj_array
end

# Generates new parameter array, given current array -
function neighbor_function(parameter_array)

  SIGMA = 0.05
  number_of_parameters = length(parameter_array)

  # calculate new parameters -
  new_parameter_array = parameter_array.*(1+SIGMA*randn(number_of_parameters))

  # Check the bound constraints -
  LOWER_BOUND = -4.0
  UPPER_BOUND = 4.0

  # return the corrected parameter arrays -
  return parameter_bounds_function(new_parameter_array,LOWER_BOUND*ones(number_of_parameters),UPPER_BOUND*ones(number_of_parameters))
end

function acceptance_probability_function(rank_array,temperature)
  return (exp(-rank_array[end]/temperature))
end

# Helper functions -
function parameter_bounds_function(parameter_array,lower_bound_array,upper_bound_array)

  # reflection_factor -
  epsilon = 0.01

  # iterate through and fix the parameters -
  new_parameter_array = copy(parameter_array)
  for (index,value) in enumerate(parameter_array)

    lower_bound = lower_bound_array[index]
    upper_bound = upper_bound_array[index]

    if (value<lower_bound)
      new_parameter_array[index] = lower_bound
    elseif (value>upper_bound)
      new_parameter_array[index] = upper_bound
    end
  end

  return new_parameter_array
end
