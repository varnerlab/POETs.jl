using DelimitedFiles

function Data_Cybernetic(tStart, tStop, tStep)

  #Load Modes
  mode_matrix = readdlm("./data/EM_modes.csv", ',', Float64)

  #Load Stoichiometric Matrix
  stm_matrix = readdlm("./data/Network.csv", ',', Float64)

  #============================#
  # EnzymeRateParameters
  ke = 0.5
  alpha = 0.004
  beta = 0.05

  #============================#
  #kcat
  kmax = [
    0.565;
    0.337;
    0.282;
    0.353;
    0.276;
    0.0955;
  ]

  #============================#
  #Ksat
  K = [
    1.552;
    1.612;
    1.431;
    1.587;
    1.563;
    1.42;
  ]

  #=============================#
  #Initial Conditions
  x0 = [
    0.5;  # 1 em1
    0.5;  # 2 em2
    0.5;  # 3 em3
    0.5;  # 4 em4
    0.5;  # 5 em5
    0.5;  # 6 em6
    2.0;  # 7 Ax
    0.0;  # 8 Bx
    0.0;  # 9 Cx
    0.01; # 10 Biox
  ]

  #=======================================#
  Data_dict = Dict{String,Any}()
  Data_dict["ReactionRateVector"] = kmax
  Data_dict["SaturationConstantVector"] = K
  Data_dict["EnzymeRate"] = ke
  Data_dict["EnzymeSynthesis"] = alpha
  Data_dict["Degradation"] = beta
  Data_dict["ModeMatrix"] = mode_matrix
  Data_dict["MetaboliteMatrix"] = stm_matrix
  Data_dict["InitialConditions"] = x0
  return Data_dict
end
