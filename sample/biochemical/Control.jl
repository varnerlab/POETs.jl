function Control(t,x,rM,rE,DF)

  #Initialize Control Vector
  cybernetic_var = Float64[]

  #Get the ModeMatrix
  Z = DF["ModeMatrix"]
  (num_reactions,num_modes) = size(Z)

  #Cybernetic_Variables
  fill!(cybernetic_var,0.0)
  push!(cybernetic_var,Z[1,1]*rM[1])
  push!(cybernetic_var,Z[1,2]*rM[2])
  push!(cybernetic_var,Z[1,3]*rM[3])
  push!(cybernetic_var,Z[1,4]*rM[4])
  push!(cybernetic_var,Z[1,5]*rM[5])
  push!(cybernetic_var,Z[1,6]*rM[6])


  #Initialize cybernetic variables u and v
  u = zeros(num_modes)
  v = zeros(num_modes)
  for i = 1:num_modes
    u[i] = cybernetic_var[i]/sum(cybernetic_var)
    v[i] = cybernetic_var[i]/maximum(cybernetic_var)
  end

  return u, v
end
