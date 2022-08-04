using ITensors

cutoff = 1E-14
maxdim = 256

m = 4  # number of timesteps
n = 2  # number of coordinates
p = 2 # number of functions

Dim = [n+1 for q in range(1, p)]
sites_Theta_row = siteinds(n+1, p)
sites_Theta_col = deepcopy(sites_Theta_row)
push!(sites_Theta_row, Index(m, "RowDim"))
pushfirst!(sites_Theta_col, Index(m, "ColDim"))

Theta_row = rand((n+1)^p, m)
Theta_col = rand(m, (n+1)^p)
Theta_rshp_row = reshape(Theta_row, (Tuple(Dim)..., m))
Theta_rshp_col = reshape(Theta_col, (m, Tuple(Dim)...))

Θ_row = MPS(Theta_rshp_row, sites_Theta_row; cutoff=cutoff, maxdim=maxdim);
Θ_col = MPS(Theta_rshp_col, sites_Theta_col; cutoff=cutoff, maxdim=maxdim);

@test Theta_rshp_row ≈ Array(Θ_row)
@test Theta_rshp_col ≈ Array(Θ_col)
@test Matrix(Θ_row) ≈ Theta_row
@test Matrix(Θ_col) ≈ Theta_col
