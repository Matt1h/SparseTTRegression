"""
    mals_lasso(Θ::MPS, dX::AbstractMatrix,  λ::AbstractVector; 
    init="random"::AbstractString, ξ_start_given=nothing::Union{Nothing, MPS}, kwargs...)

Finds the coefficiants 

# Examples
```julia-repl
Matrix(randomMPS([Index(2), Index(2), Index(3, "ColDim")]))
3×4 Matrix{Float64}:
-0.268026   0.333602   0.06218    -0.0773931
-0.224629   0.279587   0.0521122  -0.064862
0.499952  -0.622271  -0.115985    0.144362
```
"""
function mals_lasso(Θ::MPS, dX::AbstractMatrix,  λ::AbstractVector; 
    init="random"::AbstractString, ξ_start_given=nothing::Union{Nothing, MPS}, cutoff=nothing::Union{Nothing, Number}, 
    kwargs...)

    n = size(dX, 2)

    Ξ = MPS[]

    for i in range(1, n)
        ξ, ξ_start = init_ξ(Θ, init; ξ_start_given, cutoff, dX=dX[:, i])  # start coefficiants
        err_count = 0
        while true
            try
                ξ = mals_lasso(ξ, Θ, dX[:, i], i, λ; kwargs...)             
                break
            catch err
                if isa(err, ErrorException) && err_count < 10
                    print("LASSO error was thrown, init=" ,init, "\n\n")
                    init == "given" ? ξ_start_given = randomMPS(siteinds(ξ_start_given)) : nothing
                    ξ, ξ_start = init_ξ(Θ, init; ξ_start_given=ξ_start_given, cutoff=cutoff, dX=dX[:, i])
                    err_count += 1
                elseif isa(err, ErrorException) && err_count >= 10
                    print("This kind of ErrorException was thrown 100 times:\n\n")
                    throw(err)
                else
                    throw(err)
                end
            end
        end
        push!(Ξ, ξ)
    end
    return Ξ
end


"""
    mals_lasso(Θ::MPS, dX::AbstractMatrix{T}, ψ::AbstractVector, λ::AbstractVector,
    X_test_l::AbstractVector, dX_test_l::AbstractVector; cutoff=nothing,, error_func=rec::Function, kwargs...)

Uses MANDy to initialize the coefficiant TTs.
"""
function mals_lasso(Θ::MPS, dX::AbstractMatrix{T}, ψ::AbstractVector, λ::AbstractVector,
    X_test_l::AbstractVector, dX_test_l::AbstractVector; cutoff=10^-1, error_func=rec::Function, kwargs...) where{T}

    iter_rec = length(X_test_l)

    row_sites = siteinds(Θ)[1:end-1]

    Ξ = mals_lasso(Θ, dX,  λ, init="mandy", cutoff=cutoff)

    # reconstruct derivative for validation
    rec_err_l = Float64[]
    for i in range(1, iter_rec)
        Θ_rec = coordinate_major(X_test_l[i], ψ;  row_sites=row_sites)
        rec_err = error_func(Θ_rec, Ξ, dX_test_l[i])

        push!(rec_err_l, rec_err)
    end
    rec_err_mean = mean(rec_err_l)

    train_err = error_func(Θ, Ξ, dX)

    return Ξ, train_err, rec_err_mean
end

# """
#     mals_lasso(Θ::MPS, dX::AbstractMatrix{T}, ψ::AbstractVector, λ::AbstractVector,
#     X_test_l::AbstractVector, dX_test_l::AbstractVector; n_inits=10::Integer, error_func=rec::Function, kwargs...)

# Finds the coefficiants for n_inits different random initializations and chooses the model with the 
# lowest reconstruction error. Returns the training and reconstruction error for the choosen model.
# """
# function mals_lasso(Θ::MPS, dX::AbstractMatrix{T}, ψ::AbstractVector, λ::AbstractVector,
#     X_test_l::AbstractVector, dX_test_l::AbstractVector; n_inits=10::Integer, error_func=rec::Function, kwargs...) where{T}

#     iter_rec = length(X_test_l)
#     rec_err_old = Inf

#     local Ξ
#     row_sites_ = siteinds(Θ)[1:end-1]

#     for i in range(1, n_inits)
#         Ξ_new = mals_lasso(Θ, dX,  λ)

#         # reconstruct derivative for validation
#         rec_err_l = Float64[]
#         for i in range(1, iter_rec)
#             Θ_rec = coordinate_major(X_test_l[i], ψ;  row_sites=row_sites_)
#             rec_err = error_func(Θ_rec, Ξ_new, dX_test_l[i])

#             push!(rec_err_l, rec_err)
#         end
#         rec_err_mean = mean(rec_err_l)

#         if rec_err_mean  < rec_err_old
#             rec_err_old = rec_err_mean
#             Ξ = Ξ_new
#         end
#     end
#     train_err = error_func(Θ, Ξ, dX)

#     return Ξ, train_err, rec_err_old
# end


function truncated_mals_lasso(Θ, dX, λ, cutoff; sweep_count=1)
    truncate!(Θ, cutoff=cutoff)
    Ξ =  mals_lasso(Θ, dX,  λ, sweep_count=1)
end

function init_ξ(Θ::MPS, init::AbstractString; 
    ξ_start_given=nothing::Union{Nothing, MPS}, 
    cutoff=nothing::Union{Nothing, Number}, dX::Union{Nothing, AbstractArray})
    # START COEFFICIANTS
    if init == "random"
        ξ = randomMPS(siteinds(Θ)[1:end-1])
    elseif init == "mandy"
        ξ = mandy(Θ, dX)
        truncate!(ξ, cutoff=cutoff)
    elseif init == "given"
        ξ = deepcopy(ξ_start_given)
    end
    ξ_start = deepcopy(ξ)
    return ξ, ξ_start
end


function mandy(Θ::MPS, dX::AbstractArray; cutoff=nothing::Union{Nothing, Number})
    n = size(dX, 2)
    order = length(Θ)

    Θ_copy = deepcopy(Θ)
    cutoff != nothing ? truncate!(Θ_copy, cutoff=cutoff) : nothing
    Ξ = pinv(Θ_copy)
    sites = siteinds(Ξ)

    if n > 1 
        dX = ITensor(dX, [sites[end], Index(n, "RowDim")])
        Ξ[order] = Ξ[order]*dX
        return Ξ
    elseif n == 1
        dX = ITensor(dX, [sites[end]])
        ξ = deepcopy(Ξ)
        ξ[order] = ξ[order]*dX
        ξ[end-1] = ξ[end-1]*ξ[end]
        ξ = MPS(ξ[1:end-1])
        return ξ
    end
end


function mandy(Θ::MPS, dX_train::AbstractMatrix, ψ::AbstractVector, λ::AbstractVector,
    X_test_l::AbstractVector, dX_test_l::AbstractVector; error_func=rec::Function)
    n = size(dX_train, 2)
    order = length(Θ)
    Θ_copy = deepcopy(Θ)
    # truncate!(Θ_copy, cutoff=10^-2)

    # training
    Ξ = pinv(Θ_copy)
    sites = siteinds(Ξ)
    row_sites = sites[1:end-1]
    dX_train_it = ITensor(dX_train, [sites[end], Index(n, "RowDim")])
    Ξ[order] = Ξ[order]*dX_train_it
    train_err = error_func(Θ, Ξ, dX_train)

    # reconstruction error
    rec_err_l = Float64[]
    for j in range(1, length(X_test_l))
        Θ_rec = coordinate_major(X_test_l[j], ψ;  row_sites=row_sites)
        rec_err = error_func(Θ_rec, Ξ, dX_test_l[j])
        push!(rec_err_l, rec_err)
    end
    rec_err = mean(rec_err_l)


    return Ξ, train_err, rec_err
end


function sindy(Theta::AbstractArray, dX::AbstractArray, lambda::Real, iter::Int)
    n = size(dX, 2)

    # calculate Xi
    Xi = Theta\dX

    for _ in range(1, iter)
        smallinds = collect(abs.(Xi) .< lambda)
        Xi[smallinds] .= 0
        for idx in range(1, n)
            biginds = collect(.~smallinds[:, idx])
            Xi[biginds, idx] = Theta[:, biginds]\dX[:, idx]
        end
    end
    return Xi
end



function sindy(Θ::MPS, dX_train::AbstractMatrix, ψ::AbstractVector, λ::AbstractVector,
    X_test_l::AbstractVector, dX_test_l::AbstractVector; iter=1::Integer, error_func=rec::Function)
    Theta = Matrix(Θ)

    # training
    Xi = sindy(Theta, dX_train, λ[1], iter)
    train_err = error_func(Theta, Xi, dX_train)

    # reconstruction error
    rec_err_l = Float64[]
    for j in range(1, length(X_test_l))
        Θ_rec = coordinate_major(X_test_l[j], ψ)
        rec_err = error_func(Matrix(Θ_rec), Xi, dX_test_l[j])
        push!(rec_err_l, rec_err)
    end
    rec_err = mean(rec_err_l)

    return Xi, train_err, rec_err
end


function regular_lasso(Theta::AbstractArray, dX::AbstractArray{T}, λ) where{T}
    n = size(dX, 2)
    Xi = Vector{T}[]
    for i in range(1, n)
        f = fit(LassoPath, Theta, dX[:, i], λ=λ, maxncoef=100000, standardize = false, intercept = false)
        xi = Matrix(f.coefs)[:, end]
        push!(Xi, xi)
    end
    return hcat([Xi[i] for i in range(1,n)]...)
end


function regular_lasso(Θ::MPS, dX_train::AbstractArray{T}, ψ::AbstractVector, λ::AbstractVector,
     X_test_l::AbstractVector, dX_test_l::AbstractVector; error_func=rec::Function) where{T}
    Theta = Matrix(Θ)

    # training
    Xi = regular_lasso(Theta, dX_train, λ)
    train_err = error_func(Theta, Xi, dX_train)

    # reconstruction error
    rec_err_l = Float64[]
    for j in range(1, length(X_test_l))
        Θ_rec = coordinate_major(X_test_l[j], ψ)
        rec_err = error_func(Matrix(Θ_rec), Xi, dX_test_l[j])
        push!(rec_err_l, rec_err)
    end
    rec_err = mean(rec_err_l)

    return Xi, train_err, rec_err
end