include("Data_Cybernetic.jl")
include("Balances.jl")
using Sundials

function SolveBalances(tStart,tStop,tStep,Param)

  #Define time vector
  t = collect(tStart:tStep:tStop)

  # Load Initial Conditions
  Data_dict = Data_Cybernetic(tStart,tStop,tStep)
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
  f(t,x,dxdt) = Balances(t,x,dxdt,DF)
  x = Sundials.cvode(f,IC,t,reltol=1e-3,abstol=1e-6)
  #t,y = ode23(f,IC,t;reltol=1e-3,abstol=1e-6)

  # # compute x array -
  # e1 = map(y -> y[1], y);
  # e2 = map(y -> y[2], y);
  # e3 = map(y -> y[3], y);
  # e4 = map(y -> y[4], y);
  # e5 = map(y -> y[5], y);
  # e6 = map(y -> y[6], y);
  # Ax = map(y -> y[7], y);
  # Bx = map(y -> y[8], y);
  # Cx = map(y -> y[9], y);
  # Biox = map(y -> y[10], y);
  #
  # x = [e1 e2 e3 e4 e5 e6 Ax Bx Cx Biox]
  return t,x
end
