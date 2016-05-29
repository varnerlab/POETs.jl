function Kinetics(t,x,DF)

  #Initialize rate_vectors
  rM = Float64[]
  rE = Float64[]
  rG = Float64[]

  # Alias the species vector -
  e1 = x[1];
  e2 = x[2];
  e3 = x[3];
  e4 = x[4];
  e5 = x[5];
  e6 = x[6];
  Ax = x[7];
  Bx = x[8];
  Cx = x[9];
  Biox = x[10];

  #Parametrs from DF
  kmax = abs(DF["ReactionRateVector"])
  K = abs(DF["SaturationConstantVector"])
  ke = abs(DF["EnzymeRate"])
  alpha = abs(DF["EnzymeSynthesis"])
  beta = abs(DF["Degradation"])
  Z = DF["ModeMatrix"]
  S = DF["MetaboliteMatrix"]

  #Metabolite Reaction
  fill!(rM,0.0)
  push!(rM,kmax[1]*x[1]*Ax/(K[1]+Ax))
  push!(rM,kmax[2]*x[2]*Ax/(K[2]+Ax))
  push!(rM,kmax[3]*x[3]*Ax/(K[3]+Ax))
  push!(rM,kmax[4]*x[4]*Ax/(K[4]+Ax))
  push!(rM,kmax[5]*x[5]*Ax/(K[5]+Ax))
  push!(rM,kmax[6]*x[6]*Ax/(K[6]+Ax))

  #EnzymeReactionRate
  fill!(rE,0.0)
  push!(rE,ke*Ax/(K[1]+Ax))
  push!(rE,ke*Ax/(K[2]+Ax))
  push!(rE,ke*Ax/(K[3]+Ax))
  push!(rE,ke*Ax/(K[4]+Ax))
  push!(rE,ke*Ax/(K[5]+Ax))
  push!(rE,ke*Ax/(K[6]+Ax))

  #GrowthRate
  fill!(rG,0.0)
  push!(rG,Z[end,1]*rM[1])
  push!(rG,Z[end,2]*rM[2])
  push!(rG,Z[end,3]*rM[3])
  push!(rG,Z[end,4]*rM[4])
  push!(rG,Z[end,5]*rM[5])
  push!(rG,Z[end,6]*rM[6])

  #===========================================#
  kinetics_dict = Dict()
  kinetics_dict["rM_vector"] = rM
  kinetics_dict["rE_vector"] = rE
  kinetics_dict["rG_vector"] = rG
  return kinetics_dict
end
