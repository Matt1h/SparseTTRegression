function Core.Array(A::MPS)
    M = A[1]
    for q in range(2, length(A))
        M = M*A[q]
    end
    return Array(M, inds(M))
end


function Core.Array(A::MPO)
    M = A[1]
    for q in range(2, length(A))
        M = M*A[q]
    end
    return Array(M, inds(M))
end


"""
    Matrix(A::MPS)

Converts a `MPS` that represents a matrix (the `Index` that represents the column/row has
to be tagged with "ColDim"/"RowDim") to a `Matrix` and other `MPS` to a `Vector`.

# Examples
```julia-repl
julia> Matrix(randomMPS([Index(2), Index(2), Index(3, "ColDim")]))
3×4 Matrix{Float64}:
 -0.268026   0.333602   0.06218    -0.0773931
 -0.224629   0.279587   0.0521122  -0.064862
  0.499952  -0.622271  -0.115985    0.144362
```
"""
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


"""
    pinv(A::MPS)

Compute the pseudoinverse a `MPS`.

The last `Index` of the `MPS` has to represent the column or row of the matrix that is represented by the `MPS`,  
while the other indices represent the the respective opposite. The `Index` that represents the column/row has
to be tagged with "ColDim"/"RowDim".

# Examples
```julia-repl
julia> pinv(randomMPS([Index(4), Index(3), Index(4, "ColDim")]))
MPS
[1] ((dim=2|id=146), (dim=2|id=895))
[2] ((dim=2|id=690), (dim=2|id=895), (dim=3|id=68))
[3] ((dim=2|id=794), (dim=3|id=68), (dim=3|id=275|"Link,u"))
[4] ((dim=3|id=275|"Link,u"), (dim=3|id=140|"RowDim"))
```
"""
function LinearAlgebra.pinv(A::MPS)  # TODO: only works if RowDim/ColDim is the last siteind
    sites = siteinds(A)
    single_ind, single_ind_loc, _ = find_matrix_size(sites)

    W = deepcopy(A)

    # new tag
    if hastags(single_ind, "RowDim")
        W[single_ind_loc] = replaceind(W[single_ind_loc], single_ind, replacetags(single_ind, "RowDim", "ColDim"))
    elseif hastags(single_ind, "ColDim")
        W[single_ind_loc] = replaceind(W[single_ind_loc], single_ind, replacetags(single_ind, "ColDim", "RowDim"))
    end

    idx = single_ind_loc - 1

    orthogonalize!(W, idx)

    if ITensors.order(A[idx]) == 2
        u, s, v = svd(W[idx], inds(W[idx])[1])
    elseif ITensors.order(W[idx]) == 3
        u, s, v = svd(W[idx], inds(W[idx])[1:2])
    end

    W[idx] = u

    W[idx+1] = v*W[idx+1]
    W[idx+1] = (1 ./ s)*W[idx+1]
    return W
end


function find_matrix_size(A_inds)
    single_ind = filter(x -> hastags(x, "ColDim") || hastags(x, "RowDim"), A_inds)
    single_dim_loc = findall(x -> hastags(x, "ColDim") || hastags(x, "RowDim"), A_inds)
    non_single_dim_loc = findall(x -> !hastags(x, "ColDim") && !hastags(x, "RowDim"), A_inds)

    !isempty(single_ind) ? single_ind = single_ind[1] : nothing
    !isempty(single_dim_loc) ? single_dim_loc = single_dim_loc[1] : nothing
    
    return single_ind, single_dim_loc, non_single_dim_loc
end


function Base.:*(Θ::MPS, Ξ::MPS)
    ord = length(Ξ)

    E = Θ[1]*Ξ[1]
    for q in range(2, ord)
        E = E*Θ[q]*Ξ[q]
    end
    # E = E*Θ[end]
    return Array(E, inds(E)) # TODO: transpose makes only sense if "row_dim" ist first dim, like for mandy
end


function Base.:*(Θ::MPS, Ξ::Vector{MPS})
    ord = length(Ξ[1])
    b = ITensor[]
    for n in range(1, length(Ξ))
        E = Θ[1]*Ξ[n][1]
        for q in range(2, ord)
            E = E*Θ[q]*Ξ[n][q]
        end
        E = E*Θ[end]
        push!(b, E)
    end
    return hcat([Array(b[i], inds(b[i])) for i in range(1,length(b))]...)
end


function mae(Theta::AbstractArray, Xi::AbstractArray, dX::AbstractArray)
    return sum(abs.(Theta*Xi - dX))/length(dX)
end


function mape(Theta::AbstractArray, Xi::AbstractArray, dX::AbstractArray)
    return sum(abs.((Theta*Xi - dX)./dX))/length(dX)
end


function rec(Theta::AbstractArray, Xi::AbstractArray, dX::AbstractArray)
    return norm(Theta*Xi - dX)/norm(dX)
end


function mae(Θ::MPS, Ξ::Vector{MPS}, dX::AbstractArray)
    return sum(abs.(Θ*Ξ - dX))/length(dX)
end


function mape(Θ::MPS, Ξ::Vector{MPS}, dX::AbstractArray)
    return sum(abs.((Θ*Ξ - dX)./dX))/length(dX)
end


function rec(Θ::MPS, Ξ::Vector{MPS}, dX::AbstractArray)
    return norm(Θ*Ξ - dX)/norm(dX)
end


function mae(Θ::MPS, Ξ::MPS, dX::AbstractArray)
    return sum(abs.(Θ*Ξ - dX))/length(dX)
end


function mape(Θ::MPS, Ξ::MPS, dX::AbstractArray)
    return sum(abs.((Θ*Ξ - dX)./dX))/length(dX)
end


function rec(Θ::MPS, Ξ::MPS, dX::AbstractArray)

    return norm(Θ*Ξ - dX)/norm(dX)
end


function mae(Xi::AbstractArray, Xi_exact::AbstractArray)
    return sum(abs.(Xi - Xi_exact))/length(Xi_exact)
end


function mape(Xi::AbstractArray, Xi_exact::AbstractArray)
    return sum(abs.((Xi - Xi_exact)./Xi_exact))/length(Xi_exact)
end


function rec(Xi::AbstractArray, Xi_exact::AbstractArray)
    return norm(Xi - Xi_exact)/norm(Xi_exact)
end