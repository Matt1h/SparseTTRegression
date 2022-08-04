function Core.Array(A::MPS)
    M = A[1]
    for q in range(2, length(A))
        M = M*A[q]
    end
    return Array(M, inds(M))
end


function Base.Matrix(A::MPS)
    M = A[1]
    for q in range(2, length(A))
        M = M*A[q]
    end
    sites = siteinds(A)
    single_ind, single_dim_loc, non_single_dim_loc = find_matrix_size(sites)
    single_dim = ITensors.dim(single_ind)

    if hastags(single_ind, "ColDim")
        M_array = permutedims(Array(M, inds(M)), [single_dim_loc; non_single_dim_loc])
        return reshape(M_array, single_dim, :)
    elseif hastags(single_ind, "RowDim")
        M_array = permutedims(Array(M, inds(M)), [non_single_dim_loc; single_dim_loc])
        return reshape(M_array, :, single_dim)
    else 
        return reshape(Array(M, inds(M)), ITensors.dim(sites))
    end
end


function find_matrix_size(A_inds)
    single_ind = filter(x -> hastags(x, "ColDim") || hastags(x, "RowDim"), A_inds)
    single_dim_loc = findall(x -> hastags(x, "ColDim") || hastags(x, "RowDim"), A_inds)
    non_single_dim_loc = findall(x -> !hastags(x, "ColDim") && !hastags(x, "RowDim"), A_inds)

    !isempty(single_ind) ? single_ind = single_ind[1] : nothing
    !isempty(single_dim_loc) ? single_dim_loc = single_dim_loc[1] : nothing
    
    return single_ind, single_dim_loc, non_single_dim_loc
end


function random_matrix_vector_mps(m::Integer, n::Integer, p::Integer, maxdim::Integer; cutoff=1e-14)
	xi_dim = [n+1 for _ in range(1, p)]
	sites_xi = siteinds(n+1, p)
	sites_Theta = deepcopy(sites_xi)
	push!(sites_Theta, Index(m, "ColDim"))

	xi = rand((n+1)^p)
	xi_rshp = reshape(xi, Tuple(xi_dim))
	Theta = rand(m, (n+1)^p)
	Theta_rshp = reshape(Theta, (Tuple(xi_dim)..., m))

	ξ = MPS(xi_rshp, sites_xi; cutoff=cutoff, maxdim=maxdim)
	Θ = MPS(Theta_rshp, sites_Theta; cutoff=cutoff, maxdim=maxdim)
	return Θ, ξ
end


function random_row_matrix(m::Integer, n::Integer, p::Integer, maxdim::Integer; cutoff=1e-14)
    Dim = [n+1 for _ in range(1, p)]
    sites_Theta = siteinds(n+1, p)
    push!(sites_Theta, Index(m, "RowDim"))

    Theta = rand((n+1)^p, m)
    Theta_rshp = reshape(Theta, (Tuple(Dim)..., m))

    Θ = MPS(Theta_rshp, sites_Theta; cutoff=cutoff, maxdim=maxdim)
    return Θ
end


function random_col_matrix(m::Integer, n::Integer, p::Integer, maxdim::Integer; cutoff=1e-14)
    Dim = [n+1 for _ in range(1, p)]
    sites_Theta = deepcopy(sites_Theta_row)
    pushfirst!(sites_Theta, Index(m, "ColDim"))
    
    Theta = rand(m, (n+1)^p)
    Theta_rshp = reshape(Theta_col, (m, Tuple(Dim)...))
    
    Θ = MPS(Theta_rshp, sites_Theta; cutoff=cutoff, maxdim=maxdim)
    return Θ
end


function print_error(Θ::MPS, ξ::MPS, ξ_start::MPS, b::AbstractVector, λ::Number)
    error = norm(Matrix(Θ)*Matrix(ξ) - b)/norm(b)*100
    error_start = norm(Matrix(Θ)*Matrix(ξ_start) - b)/norm(b)*100
    @printf("Relative error = %.2f %% \n", error)
    @printf("Relative error with starting coefficiants = %.2f %% \n", error_start)
    sparsity = sum(abs.(Matrix(ξ)))
    sparsity_start = sum(abs.(Matrix(ξ_start)))
    @printf("Sparsity(1-Norm) = %.2f \n", sparsity)
    @printf("Sparsity(1-Norm) with starting coefficiants = %.2f \n", sparsity_start)
    cost_function = norm(Matrix(Θ)*Matrix(ξ) - b) + λ*sum(abs.(Matrix(ξ)))
    cost_function_start = norm(Matrix(Θ)*Matrix(ξ_start) - b) + λ*sum(abs.(Matrix(ξ_start)))
    @printf("Lasso cost function = %.2f \n", cost_function)
    @printf("Lasso cost function with starting coefficiants = %.2f \n", cost_function_start)
end


function plot_one_dgl(trace_forward::AbstractVector, trace_backward::AbstractVector,
    n_sweeps::Integer, name::AbstractString)
    n_core_opt = Int(length(trace_forward)/n_sweeps)
    step = 1/(2*n_core_opt)
    x_forward = vcat([collect(i+step:step:i+1/2) for i in range(0, n_sweeps-1)]...)
    x_backward = vcat([collect(i+1/2+step:step:i+1) for i in range(0, n_sweeps-1)]...)

    scatter(x_forward, trace_forward, lab="Forward sweep")
    scatter!(x_backward, trace_backward, lab="Backward sweep")
    xlabel!("Number of sweeps")
    ylabel!("Lasso cost function")
    savefig(name*".png")
end