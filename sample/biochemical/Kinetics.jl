function Kinetics(t, x, DF)

  # Alias the species vector -
  Ax = x[7]

  # Parameters from DF
  kmax = abs.(DF["ReactionRateVector"])
  K = abs.(DF["SaturationConstantVector"])
  ke = abs(DF["EnzymeRate"])
  Z = DF["ModeMatrix"]

  num_modes = size(Z, 2)

  # Metabolite Reaction rates: kmax * enzyme * substrate / (K + substrate)
  rM = [kmax[i] * x[i] * Ax / (K[i] + Ax) for i in 1:num_modes]

  # Enzyme Reaction rates
  rE = [ke * Ax / (K[i] + Ax) for i in 1:num_modes]

  # Growth Rate
  rG = [Z[end, i] * rM[i] for i in 1:num_modes]

  kinetics_dict = Dict{String,Any}()
  kinetics_dict["rM_vector"] = rM
  kinetics_dict["rE_vector"] = rE
  kinetics_dict["rG_vector"] = rG
  return kinetics_dict
end
