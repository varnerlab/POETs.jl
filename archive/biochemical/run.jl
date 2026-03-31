include("SolveBalances.jl")

# Define Time
tStart = 0.0
tStop = 100.0
tStep = 1.0

# Load Parameter Values
Param = Float64[]

# Solver Results
(t, x) = SolveBalances(tStart, tStop, tStep, Param)

# Define species from Solver Results
Ax = x[:, 7]
Bx = x[:, 8]
Cx = x[:, 9]
Biox = x[:, 10]

# Plot results (requires Plots.jl or PyPlot.jl)
# using Plots
# plot(t, [Ax Bx Cx Biox], label=["A" "B" "C" "Biomass"],
#      xlabel="Time (hr)", ylabel="Abundance (mM)", linewidth=2)
