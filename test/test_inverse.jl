using LinearAlgebra
using ITensors

cutoff = 1E-14
maxdim = 256

m = 4  # number of timesteps
n = 2  # number of coordinates
p = 2 # number of functions

Dim = [n+1 for q in range(1, p)]
sites_Theta_row = siteinds(n+1, p)
push!(sites_Theta_row, Index(m, "RowDim"))

Theta_row = rand((n+1)^p, m)
Theta_rshp_row = reshape(Theta_row, (Tuple(Dim)..., m))


Θ_row = MPS(Theta_rshp_row, sites_Theta_row; cutoff=cutoff, maxdim=maxdim)

Θ_inv_row = pinv(Θ_row)

Theta_inv_row = pinv(Theta_row)

@test Theta_inv_row ≈ Matrix(Θ_inv_row)
