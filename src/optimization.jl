function sparse_TT_optimization(Θ::MPS, dX::AbstractMatrix,  λ::AbstractVector; 
    init="random"::AbstractString, make_plot=false::Bool, verbose=false::Bool, svd_tol=1e-14::Number, 
    lasso_tol = 1e-7, sweep_count=10::Integer)

    n = size(dX, 2)

    Ξ = MPS[]
    for i in range(1, n)
        # START COEFFICIANTS
        if init == "random"
            ξ = randomMPS(siteinds(Θ)[1:end-1])
        elseif init == "solve_eqs"
            ξ = Matrix(Θ)\b
            ξ = MPS(ξ, siteinds(Θ)[1:end-1])
        end
        ξ_start = deepcopy(ξ)

        # RUN
        while true
            try
                ξ, trace_forward, trace_backward = mals_lasso(ξ, Θ, dX[:, i], λ; verbose=verbose, svd_tol=svd_tol, lasso_tol=lasso_tol, sweep_count=sweep_count)
                break
            catch
                # START COEFFICIANTS
                if init == "random"
                    ξ = randomMPS(siteinds(Θ)[1:end-1])
                elseif init == "solve_eqs"
                    ξ = Matrix(Θ)\b
                    ξ = MPS(ξ, siteinds(Θ)[1:end-1])
                end
                ξ_start = deepcopy(ξ)
            end
        end
        push!(Ξ, ξ)
        verbose ? print_error(Θ, ξ, ξ_start, dX[:, i], λ[end]) : nothing
        make_plot ? plot_one_dgl(trace_forward, trace_backward, Int(sweep_count/2), "test_model1_dgl"*string(i)) : nothing
    end
    return Ξ
end


function mandy(Θ::MPS, dX::AbstractMatrix)
    n = size(dX, 2)
    order = length(Θ)

    Ξ = pinv(Θ)
    sites = siteinds(Ξ)

    dX = ITensor(dX, [sites[end], Index(n, "RowDim")])
    Ξ[order] = Ξ[order]*dX
    return Ξ
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
