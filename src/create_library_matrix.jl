function function_major(X::AbstractArray{T}, psi::AbstractArray; add_one::Bool=true,  # TODO: should construct cores not the hole array
    dims::Union{Tuple, Nothing}=nothing, maxdim=100, cutoff=10e-14) where{T}
    m, n = size(X)  
    p = length(psi)
    Theta = Array{T}(undef, m, (n + add_one)^p)
    for i in range(1, m)
        Theta[i, :] = kron([[add_one; psi[q].(X[i, :])] for q in range(1, p)]...)
    end
    if isnothing(dims)
        Theta = reshape(Theta, (m, Tuple(repeat([n+add_one], p))...))
        sites = siteinds(n + add_one, p)
        push!(sites, Index(m, "ColDim"))
    else
        Theta = reshape(Theta, dims)
        sites = [Index(dims[q]) for q in range(1, length(dims))]
        push!(sites, Index(m, "ColDim"))
    end
    
    return MPS(Theta, sites; cutoff=cutoff, maxdim=maxdim)
end


function coordinate_major(X::AbstractMatrix{T}, psi::AbstractVector; 
    site_inds=nothing::Union{AbstractVector, Nothing}, maxdim=100, cutoff=10e-14) where{T}
    m, order = size(X)
    p = length(psi)

    link_inds = [Index(m) for _ in range(1, order)]
    if isnothing(site_inds)
        site_inds = [Index(p) for _ in range(1, order)]
        push!(site_inds, Index(m, "ColDim"))
    end

    Θ = MPS(order+1)
    for q in range(1, order+1)
        if q == 1  # first core
            core = zeros(T, p, m)
            for i in range(1, m)
                core[:, i] = [psi[j](X[i, q]) for j in range(1, p)]
            end
            Θ[1] = ITensor(core, [site_inds[q], link_inds[q]])
        elseif q == order + 1 # last core (extra core for matrix column)
            core = zeros(T, m, m)
            for i in range(1, m)
                core[i, i] = 1
            end
            Θ[order+1] = ITensor(core, link_inds[q-1], site_inds[q]) 
        else  # middel cores
            core = zeros(T, m, p, m)
            for i in range(1, m)
                core[i, :, i] = [psi[j]((X)[i, q]) for j in range(1,p)]
            end
            Θ[q] = ITensor(core, [link_inds[q-1], site_inds[q], link_inds[q]])
        end
    end
    return Θ
end