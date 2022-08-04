function full(A::MPS)
    M = A[1]
    for q in range(2, length(A))
        M = M*A[q]
    end
    return Array(M, inds(M))
end


function init_H_Itensor(U::MPS, A::MPS)
	order = length(U.data)
	H = Array{ITensor}(undef, order)
	H[order] = A[order+1]
	for q = order:-1:2
		H[q-1] = update_H_Itensor(U.data[q], A.data[q], H[q])
	end
	return H
end


function update_H_Itensor(U_c::ITensor, A_c::ITensor, Hi::ITensor)
	return Hi*U_c*A_c
end

cutoff = 1E-14
maxdim = 256

m = 100  # number of timesteps
n = 10  # number of coordinates
p = 5 # number of functions

dim = [n+1 for q in range(1, p)]
sites_xi = siteinds(n+1, p)
sites_Theta = deepcopy(sites_xi)
push!(sites_Theta, Index(m))

xi_start = rand((n+1)^p)
xi_start_rshp = reshape(xi_start, Tuple(dim))
Theta = rand(m, (n+1)^p)
Theta_rshp = reshape(Theta, (Tuple(dim)..., m))

ξ_start_mps = MPS(xi_start_rshp, sites_xi; cutoff=cutoff, maxdim=maxdim)
Θ_mps = MPS(Theta_rshp, sites_Theta; cutoff=cutoff, maxdim=maxdim)

init_H_Itensor($ξ_start_mps, $Θ_mps)

