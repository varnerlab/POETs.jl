# include model data -
include("Data_Cybernetic.jl")
include("SolveBalances.jl")

using PyCall
@pyimport numpy as np

using Debug

# some global parameters -
BIG = 1e10
SMALL = 1e-6

# Globally load data so we don't have load on each iteration -
MEASURED_ARRAY_1 = readdlm("./data/MEASUREMENT_SET_1.dat")
MEASURED_ARRAY_2 = readdlm("./data/MEASUREMENT_SET_2.dat")
MEASURED_ARRAY_3 = readdlm("./data/MEASUREMENT_SET_3.dat")
MEASURED_ARRAY_4 = readdlm("./data/MEASUREMENT_SET_4.dat")

# Evaluates the objective function values -
function local_refienment_step(parameter_array)

  SIGMA = 0.05

  # initialize -
  number_of_parameters = length(parameter_array)

  # Setup the bound constraints -
  LOWER_BOUND = SMALL
  UPPER_BOUND = [
    1.00  ; # 1 k1
    1.00  ; # 2 k2
    1.00  ; # 3 k3
    1.00  ; # 4 k4
    1.00  ; # 5 k5
    1.00  ; # 6 k6
    3.00  ; # 7 K1
    3.00  ; # 8 K2
    3.00  ; # 9 K3
    3.00  ; # 10 K4
    3.00  ; # 11 K5
    3.00  ; # 12 K6
    1.00  ; # 13 ke
  ]

  # calculate the starting error -
  parameter_array_best = parameter_array
  error_array = BIG*ones(4)
  error_array[1] = sum(objective_function(parameter_array_best))

  # main refinement loop -
  iteration_counter = 1
  iteration_max = 1000
  while (iteration_counter<iteration_max)

    # take a step up -
    parameter_up = parameter_array_best.*(1+SIGMA*rand(number_of_parameters))
    parameter_up = parameter_bounds_function(parameter_up,LOWER_BOUND*ones(number_of_parameters),UPPER_BOUND)

    # take a step down -
    parameter_down = parameter_array_best.*(1-SIGMA*rand(number_of_parameters))
    parameter_down = parameter_bounds_function(parameter_down,LOWER_BOUND*ones(number_of_parameters),UPPER_BOUND)

    # Evaluate the obj function -
    error_array[2] = sum(objective_function(parameter_up))
    error_array[3] = sum(objective_function(parameter_down))

    # Calculate a correction factor -
    a = error_array[2]+error_array[3] - 2.0*error_array[1]
    parameter_corrected = parameter_array_best
    if (a>0.0)
      amda = -0.5*(error_array[3] - error_array[2])/a
      parameter_corrected = parameter_array_best+amda*rand(number_of_parameters)
      parameter_corrected = parameter_bounds_function(parameter_corrected,LOWER_BOUND*ones(number_of_parameters),UPPER_BOUND)
      error_array[4] = sum(objective_function(parameter_corrected))
    end

    # Which step has the min error?
    min_index = indmin(error_array)
    if (min_index == 1)
			parameter_array_best = parameter_array_best
	 	elseif (min_index == 2)
		 	parameter_array_best = parameter_up
	 	elseif (min_index == 3)
		 	parameter_array_best = parameter_down
	 	elseif (min_index == 4)
		 parameter_array_best = parameter_corrected
	 	end

		# Update the local error
		error_array[1] = error_array[min_index]

    @show iteration_counter,error_array[min_index]

    # update local counter -
    iteration_counter = iteration_counter + 1
  end

  return parameter_array_best
end


function calculate_error_m_1(t,x)

  tStart = 0.0;
  tStop = 100.0;
  tStep = 0.1;

  obj_array = BIG*ones(4)

  # Need to interpolate the simulation onto the experimental time scale -
  number_of_measurements = 20
  time_experimental = linspace(tStart,tStop,number_of_measurements)
  AI = np.interp(time_experimental[:],t,x[:,7])
  BI = np.interp(time_experimental[:],t,x[:,8])
  CI = np.interp(time_experimental[:],t,x[:,9])
  XI = np.interp(time_experimental[:],t,x[:,10])

  # interpolate the e data onto the same timescale -
  AMI = np.interp(time_experimental[:],MEASURED_ARRAY_1[:,1],MEASURED_ARRAY_1[:,2])
  BMI = np.interp(time_experimental[:],MEASURED_ARRAY_1[:,1],MEASURED_ARRAY_1[:,3])
  CMI = np.interp(time_experimental[:],MEASURED_ARRAY_1[:,1],MEASURED_ARRAY_1[:,4])
  XMI = np.interp(time_experimental[:],MEASURED_ARRAY_1[:,1],MEASURED_ARRAY_1[:,5])

  # Compute the error values -
  error_A = sum((AMI - AI).^2)
  error_B = sum((BMI - BI).^2)
  error_C = sum((CMI - CI).^2)
  error_X = sum((XMI - XI).^2)

  # Compute the objective array -
  obj_array[1] = error_A
  obj_array[2] = error_B
  obj_array[3] = error_C
  obj_array[4] = error_X

  return (sum(obj_array))
end

function calculate_error_m_2(t,x)

  tStart = 0.0;
  tStop = 100.0;
  tStep = 0.1;

  obj_array = BIG*ones(4)

  # Need to interpolate the simulation onto the experimental time scale -
  number_of_measurements = 20
  time_experimental = linspace(tStart,tStop,number_of_measurements)
  AI = np.interp(time_experimental[:],t,x[:,7])
  BI = np.interp(time_experimental[:],t,x[:,8])
  CI = np.interp(time_experimental[:],t,x[:,9])
  XI = np.interp(time_experimental[:],t,x[:,10])

  # interpolate the e data onto the same timescale -
  AMI = np.interp(time_experimental[:],MEASURED_ARRAY_2[:,1],MEASURED_ARRAY_2[:,2])
  BMI = np.interp(time_experimental[:],MEASURED_ARRAY_2[:,1],MEASURED_ARRAY_2[:,3])
  CMI = np.interp(time_experimental[:],MEASURED_ARRAY_2[:,1],MEASURED_ARRAY_2[:,4])
  XMI = np.interp(time_experimental[:],MEASURED_ARRAY_2[:,1],MEASURED_ARRAY_2[:,5])

  # Compute the error values -
  error_A = sum((AMI - AI).^2)
  error_B = sum((BMI - BI).^2)
  error_C = sum((CMI - CI).^2)
  error_X = sum((XMI - XI).^2)

  # Compute the objective array -
  obj_array[1] = error_A
  obj_array[2] = error_B
  obj_array[3] = error_C
  obj_array[4] = error_X

  return (sum(obj_array))
end

function calculate_error_m_3(t,x)

  tStart = 0.0;
  tStop = 100.0;
  tStep = 0.1;

  obj_array = BIG*ones(4)

  # Need to interpolate the simulation onto the experimental time scale -
  number_of_measurements = 20
  time_experimental = linspace(tStart,tStop,number_of_measurements)
  AI = np.interp(time_experimental[:],t,x[:,7])
  BI = np.interp(time_experimental[:],t,x[:,8])
  CI = np.interp(time_experimental[:],t,x[:,9])
  XI = np.interp(time_experimental[:],t,x[:,10])

  # interpolate the e data onto the same timescale -
  AMI = np.interp(time_experimental[:],MEASURED_ARRAY_3[:,1],MEASURED_ARRAY_3[:,2])
  BMI = np.interp(time_experimental[:],MEASURED_ARRAY_3[:,1],MEASURED_ARRAY_3[:,3])
  CMI = np.interp(time_experimental[:],MEASURED_ARRAY_3[:,1],MEASURED_ARRAY_3[:,4])
  XMI = np.interp(time_experimental[:],MEASURED_ARRAY_3[:,1],MEASURED_ARRAY_3[:,5])

  # Compute the error values -
  error_A = sum((AMI - AI).^2)
  error_B = sum((BMI - BI).^2)
  error_C = sum((CMI - CI).^2)
  error_X = sum((XMI - XI).^2)

  # Compute the objective array -
  obj_array[1] = error_A
  obj_array[2] = error_B
  obj_array[3] = error_C
  obj_array[4] = error_X

  return (sum(obj_array))
end

function calculate_error_m_4(t,x)

  tStart = 0.0;
  tStop = 100.0;
  tStep = 0.1;

  obj_array = BIG*ones(4)

  # Need to interpolate the simulation onto the experimental time scale -
  number_of_measurements = 20
  time_experimental = linspace(tStart,tStop,number_of_measurements)
  AI = np.interp(time_experimental[:],t,x[:,7])
  BI = np.interp(time_experimental[:],t,x[:,8])
  CI = np.interp(time_experimental[:],t,x[:,9])
  XI = np.interp(time_experimental[:],t,x[:,10])

  # interpolate the e data onto the same timescale -
  AMI = np.interp(time_experimental[:],MEASURED_ARRAY_4[:,1],MEASURED_ARRAY_4[:,2])
  BMI = np.interp(time_experimental[:],MEASURED_ARRAY_4[:,1],MEASURED_ARRAY_4[:,3])
  CMI = np.interp(time_experimental[:],MEASURED_ARRAY_4[:,1],MEASURED_ARRAY_4[:,4])
  XMI = np.interp(time_experimental[:],MEASURED_ARRAY_4[:,1],MEASURED_ARRAY_4[:,5])

  # Compute the error values -
  error_A = sum((AMI - AI).^2)
  error_B = sum((BMI - BI).^2)
  error_C = sum((CMI - CI).^2)
  error_X = sum((XMI - XI).^2)

  # Compute the objective array -
  obj_array[1] = error_A
  obj_array[2] = error_B
  obj_array[3] = error_C
  obj_array[4] = error_X

  return (sum(obj_array))
end

function objective_function(parameter_array)


  # Calculate the objective function array -
  obj_array = BIG*ones(4,1)

  # Solve the model with the update parameters -
  tStart = 0.0;
  tStop = 100.0;
  tStep = 0.1;
  (t,x) = SolveBalances(tStart,tStop,tStep,parameter_array)

  # call the experiment functions -
  obj_array[1] = calculate_error_m_1(t,x)
  obj_array[2] = calculate_error_m_2(t,x)
  obj_array[3] = calculate_error_m_3(t,x)
  obj_array[4] = calculate_error_m_4(t,x)

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
  LOWER_BOUND = SMALL
  UPPER_BOUND = [
    1.00  ; # 1 k1
    1.00  ; # 2 k2
    1.00  ; # 3 k3
    1.00  ; # 4 k4
    1.00  ; # 5 k5
    1.00  ; # 6 k6
    3.00  ; # 7 K1
    3.00  ; # 8 K2
    3.00  ; # 9 K3
    3.00  ; # 10 K4
    3.00  ; # 11 K5
    3.00  ; # 12 K6
    1.00  ; # 13 ke
  ]

  # return the corrected parameter arrays -
  return parameter_bounds_function(new_parameter_array,LOWER_BOUND*ones(number_of_parameters),UPPER_BOUND)
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
      new_parameter_array[index] = lower_bound
    elseif (value>upper_bound)
      new_parameter_array[index] = upper_bound
    end
  end

  return new_parameter_array
end
