using ITensors
using LinearAlgebra
using SparseTTRegression
using BenchmarkTools
using Random
using PyPlot
using experiments
# using PyCall
using CSV
using DataFrames
# @pyimport matplotlib.animation as animation


function ddX_fput(X::AbstractMatrix{T}, β::Number) where{T}
    m, n = size(X)
    ddX = zeros(T, m, n)
    for i in range(1, m)
        for j in 2:n-1
            ddX[i, j] = X[i, j+1] - 2*X[i, j] + X[i, j-1] + β*((X[i, j+1] - X[i, j])^2 - (X[i, j] - X[i, j-1])^2)
        end
    end
    return ddX
end


# function fput_animation(sol, amplitude)
#     m = length(sol.t)
#     n = Int(length(sol.u[1])/2)
#     anim = @animate for (i,u) in enumerate(sol.u)
#         Plots.scatter(u[1:n], ylims = (-1.1*amplitude,1.1*amplitude),)
#     end
#     gif(anim, "anim_fps15.gif", fps = 15)
# end


function exact_coef_fput_alpha(ψ::AbstractVector, coef::Number, n_dgl::Integer)
    p = length(ψ)
    b = p^n_dgl
    Xi_exact = zeros(typeof(coef), (b, n_dgl))
    for i in range(1, n_dgl)
        i != 1 ? Xi_exact[p^(i-2)+1, i] = 1 : nothing
        Xi_exact[p^(i-1)+1, i] = -2
        i != n_dgl ? Xi_exact[p^i+1, i] = 1 : nothing
        i != 1 ? Xi_exact[2*p^(i-2)+1, i] = -coef : nothing
        # Xi_exact[2*p^(i-1)+1, i] = 2*coef
        i != n_dgl ? Xi_exact[2*p^i+1, i] = coef : nothing
        i != 1 ? Xi_exact[p^(i-1)+1+p^(i-2), i] = 2*coef : nothing
        i != n_dgl ? Xi_exact[p^i+1+p^(i-1), i] = -2*coef : nothing

    end
    return Xi_exact
end


function exact_coef_fput_beta(ψ::AbstractVector, coef::Number, n_dgl::Integer)
    p = length(ψ)^n_dgl
    Xi_exact = zeros(typeof(coef), (p, n_dgl))
    for q in range(1, n_dgl)
        Xi_exact[4^(q-1)+1, q] = -2
        q != n_dgl ? Xi_exact[4^q+1, q] = 1 : nothing
        q != 1 ? Xi_exact[4^(q-2)+1, q] = 1 : nothing
        q != n_dgl ? Xi_exact[3*4^q+1, q] = coef : nothing
        q != n_dgl ? Xi_exact[2*4^q+1+4^(q-1), q] = -3*coef : nothing
        q != n_dgl ? Xi_exact[4^q+1 + 2*4^(q-1), q] = 3*coef : nothing
        Xi_exact[3*4^(q-1)+1, q] = -2*coef
        q != 1 ? Xi_exact[2*4^(q-1)+1+4^(q-2), q] = -3*coef : nothing
        q != 1 ? Xi_exact[4^(q-1)+1+2*4^(q-2), q] = 3*coef : nothing
        q != 1 ? Xi_exact[3*4^(q-2)+1, q] = -coef : nothing
    end
    return Xi_exact
end


function normal_mode_C(x, mode)
    N = length(x)-1
    return sqrt(2/N)*sum(x.*sin.(((0:N).*pi*mode)./N))
end


# load data
X_train = Matrix(CSV.read(joinpath("experiments","data","fput","diff_inits","rand_rand","train_data","X_train.csv"), DataFrame))[:, 2:end-1]
ddX_train = Matrix(CSV.read(joinpath("experiments","data","fput","diff_inits","rand_rand","train_data","ddX_train.csv"), DataFrame))[:, 2:end-1]

X_test_l = []
ddX_test_l = []

for j in range(1,9)
    push!(X_test_l, Matrix(CSV.read(joinpath("experiments","data","fput","diff_inits","rand_rand","test_data","X_test"*string(j)*".csv"), DataFrame))[:, 2:end-1])
    push!(ddX_test_l, Matrix(CSV.read(joinpath("experiments","data","fput","diff_inits","rand_rand","test_data","ddX_test"*string(j)*".csv"), DataFrame))[:, 2:end-1])
end

# create library matrix
ψ = [x -> 1, x->x, x->x^2]
Θ = coordinate_major(X_train, ψ)

# calculate exact Xi
Xi_exact = exact_coef_fput_alpha(ψ, 0.25, 8)

λ = 10 .^(range(-1, stop=-3, length=10))
Ξ_l, train_err_l, rec_err_mean_l, rec_err_std_l, coef_err_l = val_diff_inits(Θ, ddX_train, ψ, Xi_exact, X_test_l, ddX_test_l, λ)