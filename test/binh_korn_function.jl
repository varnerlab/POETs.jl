# some global parameters -
BIG = 1e10

# Evaluates the objective function values -
function objective_function(parameter_array)

  # Alias the species -
  x = parameter_array[1]
  y = parameter_array[2]

  # Calculate the objective function array -
  obj_array = BIG*ones(2,1)

  # f1 and f2 -
  obj_array[1] = 4.0*(x^2)+4.0*(y^2)
  obj_array[2] = (x - 5)^2 + (y - 5)^2

  # Constraints are implemented as a penaltly on obj value
  lambda_value = 100.0

  # How much do we violate the constraints?
  violation_constraint_1 = 25 - (x-5.0)^2 - y^2
  violation_constraint_2 = (x-8.0)^2 + (y-3.0)^2 - 7.7
  penaltly_array = zeros(2)
  penaltly_array[1] = lambda_value*(min(0,violation_constraint_1))^2
  penaltly_array[2] = lambda_value*(min(0,violation_constraint_2))^2

  # return the obj_array -
  return obj_array+penaltly_array
end

# Generates new parameter array, given current array -
function neighbor_function(parameter_array)

  SIGMA = 0.05
  number_of_parameters = length(parameter_array)

  # calculate new parameters -
  new_parameter_array = parameter_array.*(fill(1, number_of_parameters)+SIGMA*randn(number_of_parameters))

  # Check the bound constraints -
  LOWER_BOUND = [0,0]
  UPPER_BOUND = [5,3]

  # return the corrected parameter arrays -
  return parameter_bounds_function(new_parameter_array,LOWER_BOUND,UPPER_BOUND)
end

function acceptance_probability_function(rank_array,temperature)

  return (exp(-rank_array[end]/temperature))

end

function cooling_function(temperature)

  # define my new temperature -
  alpha = 0.9
  return alpha*temperature
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
      new_parameter_array[index] = lower_bound+epsilon*upper_bound
    elseif (value>upper_bound)
      new_parameter_array[index] = upper_bound - epsilon*lower_bound
    end
  end

  return new_parameter_array
end
