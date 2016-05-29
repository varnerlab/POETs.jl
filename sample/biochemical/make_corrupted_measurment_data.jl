include("SolveBalances.jl")

#DefineTime
tStart = 0.0;
tStop = 100.0;
tStep = 0.1;

#loadParameterValues
Param = Float64[]

#SolverResults
(t,x) = SolveBalances(tStart,tStop,tStep,Param)

#Define species from SolverResults
Ax = x[:,7]
Bx = x[:,8]
Cx = x[:,9]
Biox = x[:,10]

# write data files -
Ax = Ax.*(1.0)
Bx = Bx.*(1.0)
Cx = Cx.*(1.0)
Biox = Biox.*(1.0)

data = [t Ax Bx Cx Biox]
writedlm("./data/MEASUREMENT_SET_4.dat",data)
