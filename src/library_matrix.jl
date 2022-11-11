"""
    coordinate_major(X::AbstractMatrix, ψ::AbstractVector)

Compute a coordinate-major decomposition as a `MPS`.

`ψ` is a vector of different basis functions, stored as `Function`. `X` is a ``m × n`` Matrix with data, where 
``m`` is the number of snapshots and ``n`` is the number of ODEs respectively the number of different coordinates. 
`X` is expected to be an trajectory of a system of ``n`` ODEs, that can be constructed with combinations of 
different basis functions applied to different coordinates.

For every coordinate a Tensor Train core is constructed by applying all basis functions in `ψ` to the coordinate.
This is done for every snapshot. An extra core is constructed that works as an index for the different snapshots.

# Examples
```julia-repl
julia> coordinate_major([4.0 1.0 3.0; 0.1 17.0 17.0; 1.0 4.0 3.0], [sin; cos])
ITensors.MPS
[1] ((dim=2|id=108), (dim=3|id=332))
[2] ((dim=3|id=332), (dim=2|id=737), (dim=3|id=245))
[3] ((dim=3|id=245), (dim=2|id=730), (dim=3|id=233))
[4] ((dim=3|id=233), (dim=3|id=713|"ColDim"))
```
"""
function coordinate_major(X::AbstractMatrix{T}, psi::AbstractVector; 
    row_sites::Union{Nothing, Vector{Index{Int64}}}=nothing) where{T}
    m, n = size(X)
    p = length(psi)

    link_inds = [Index(m) for _ in range(1, n)]

    if isnothing(row_sites)
        site_inds = [Index(p) for _ in range(1, n)]
    else
        site_inds = deepcopy(row_sites)
    end
    push!(site_inds, Index(m, "ColDim"))

    Θ = MPS(n+1)
    for q in range(1, n+1)
        if q == 1  # first core
            core = zeros(T, p, m)
            for i in range(1, m)
                core[:, i] = [psi[j](X[i, q]) for j in range(1, p)]
            end
            Θ[1] = ITensor(core, [site_inds[q], link_inds[q]])
        elseif q == n + 1 # last core (extra core for matrix column)
            core = zeros(T, m, m)
            for i in range(1, m)
                core[i, i] = 1
            end
            Θ[n+1] = ITensor(core, link_inds[q-1], site_inds[q]) 
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