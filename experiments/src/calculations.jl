function val_diff_inits(Θ::MPS, dX_train::AbstractArray, ψ::AbstractVector, 
    Xi_exact::AbstractArray, X_test_l::AbstractVector, dX_test_l::AbstractVector, λ::AbstractVector; 
    iter_train=13::Integer, error_func=rec::Function)
    iter_rec = length(X_test_l)

    train_err_l = Float64[]
    rec_err_mean_l = Float64[]
    rec_err_std_l = Float64[]
    coef_err_l = []
    Θ_rec_l = MPS[]
    Ξ_l = Vector{MPS}[]

    row_sites = siteinds(Θ)[1:end-1]

    # construct test library TTs
    for j in range(1, iter_rec)
        Θ_rec = coordinate_major(X_test_l[j], ψ, row_sites=row_sites)
        push!(Θ_rec_l, Θ_rec)
    end

    for i in range(1, iter_train)
        # training
        ξ_start = randomMPS(row_sites)
        if i == 1 
            Ξ = mals_lasso(Θ, dX_train,  λ; init="mandy", cutoff=10^-1)
        # elseif i == 2
        #     Ξ = mals_lasso(Θ, dX_train,  λ; init="mandy", cutoff=10^-14)
        # elseif i == 3
        #     Ξ = mals_lasso(Θ, dX_train,  λ; init="mandy", cutoff=10^-6)
        # elseif i == 4
        #     Ξ = mals_lasso(Θ, dX_train,  λ; init="mandy", cutoff=10^-3)
        else
            Ξ = mals_lasso(Θ, dX_train,  λ; init="given", ξ_start_given=ξ_start)
        end
        # Ξ = mals_lasso(Θ, dX_train,  λ, init="given", ξ_start_given=ξ_start)
        train_err = error_func(Θ, Ξ, dX_train)
        push!(train_err_l, train_err)
        push!(Ξ_l, Ξ)

        # coefficiant error
        Xi = hcat([Matrix(Ξ[i]) for i in range(1, length(Ξ))]...)  # explicit Xi
        push!(coef_err_l, error_func(Xi, Xi_exact))

        # reconstruction error
        rec_err_l = Float64[]
        for j in range(1, iter_rec)
            rec_err = error_func(Θ_rec_l[j], Ξ, dX_test_l[j])
            push!(rec_err_l, rec_err)
        end
        push!(rec_err_mean_l, mean(rec_err_l))
        push!(rec_err_std_l, std(rec_err_l))

    end
    return Ξ_l, train_err_l, rec_err_mean_l, rec_err_std_l, coef_err_l
end
