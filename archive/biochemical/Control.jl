function Control(t, x, rM, rE, DF)

  #Get the ModeMatrix
  Z = DF["ModeMatrix"]
  (num_reactions, num_modes) = size(Z)

  # Cybernetic variables: resource allocation based on first reaction row
  cybernetic_var = [Z[1, i] * rM[i] for i in 1:num_modes]

  # Initialize cybernetic variables u and v
  cv_sum = sum(cybernetic_var)
  cv_max = maximum(cybernetic_var)
  u = cybernetic_var ./ cv_sum
  v = cybernetic_var ./ cv_max

  return u, v
end
