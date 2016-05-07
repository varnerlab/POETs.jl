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
  return bounds(new_parameter_array,LOWER_BOUND*ones(number_of_parameters),UPPER_BOUND*ones(number_of_parameters))
end

function acceptance_probability_function(rank_array,temperature)
  return (exp(-rank_array[end]/temperature))
end

# Helper functions -
function bounds(x,MINJ,MAXJ)

	JMIN_NEW = find(x.<MINJ);
	x[JMIN_NEW] = MINJ[JMIN_NEW]+(MINJ[JMIN_NEW]-x[JMIN_NEW]);

	JTEMP1 = find(x[JMIN_NEW].>MAXJ[JMIN_NEW]);
	x[JTEMP1] = MINJ[JTEMP1];

	JMAX_NEW = find(x.>MAXJ);
	x[JMAX_NEW] = MAXJ[JMAX_NEW]-(x[JMAX_NEW]-MAXJ[JMAX_NEW]);

	JTEMP2 = find(x[JMAX_NEW].<MINJ[JMAX_NEW]);
	x[JTEMP2 = MAXJ[JTEMP2];

	CHKMAX = find(x.>MAXJ);
	x[CHKMAX] = MINJ[CHKMAX];

	CHKMIN = find(x.<MINJ);
	x[CHKMIN] = MAXJ[CHKMIN];

  return x
end
