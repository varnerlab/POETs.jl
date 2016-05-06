using Debug

# some global parameters -
BIG = 1e10

# Evaluates the objective function values -
@debug function objective_function(parameter_array)

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
  return parameter_array.*(1+SIGMA*randn(number_of_parameters))
end

@debug function acceptance_probability_function(rank_array,temperature)

  return (exp(-rank_array[end]/temperature))

end
