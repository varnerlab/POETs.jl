include("Data_Cybernetic.jl")
include("Balances.jl")
using Sundials

function SolveBalances(tStart, tStop, tStep, Param)

  #Define time vector
  t = collect(tStart:tStep:tStop)

  # Load Initial Conditions
  Data_dict = Data_Cybernetic(tStart, tStop, tStep)
  IC = Data_dict["InitialConditions"]

  #Define Parameter values
  if isempty(Param)
    DF = Data_dict
  else
    kmax = Param[1:6]
    K = Param[7:12]
    ke = Param[13]

    # Rewrite Data Dictionary
    Data_dict["ReactionRateVector"] = kmax
    Data_dict["SaturationConstantVector"] = K
    Data_dict["EnzymeRate"] = ke
    DF = Data_dict
  end

  #RunSolver
  f(t, x, dxdt) = Balances(t, x, dxdt, DF)
  x = Sundials.cvode(f, IC, t, reltol=1e-3, abstol=1e-6)

  return t, x
end
