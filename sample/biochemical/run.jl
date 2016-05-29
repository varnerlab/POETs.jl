include("SolveBalances.jl")

#DefineTime
tStart = 0.0;
tStop = 100.0;
tStep = 1;

#loadParameterValues
Param = Float64[]

#SolverResults
(t,x) = SolveBalances(tStart,tStop,tStep,Param)

#Define species from SolverResults
Ax = x[:,7]
Bx = x[:,8]
Cx = x[:,9]
Biox = x[:,10]


using PyPlot
figure(1)
plot(t,Ax,color="k",label="HCM EM",linewidth=2)
plot(t,Bx, color="r",linewidth=2)
plot(t,Cx, color="b",linewidth=2)
plot(t,Biox, color="g",linewidth=2)
legend(fontsize=18)
xlabel("Time (hr)",fontsize=20)
ylabel("Abundance (mM)",fontsize=20)
axis([0,100,0,2.1;]);
