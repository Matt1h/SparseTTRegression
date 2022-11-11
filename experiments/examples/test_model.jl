using ITensors
using LinearAlgebra
using SparseTTRegression
using TensorComputation
using Plots
using BenchmarkTools
using Statistics
using Noise


function test_model!(dx, x, coeff, t)
    alpha, beta = coeff 
    dx[1] = alpha*sin(x[1] - x[2])
    dx[2] = beta*sin(x[1])
    dx[3] = beta*sin(x[2])
end


function call_derivative_test_model(X::AbstractMatrix{T}, coeff::Tuple) where{T}
    α, β = coeff
    m, n = size(X)
    dX = zeros(T, m, n)
    for i in range(1, m)
        dX[i, 1] = α*sin(X[i, 1] - X[i, 2])
        dX[i, 2] = β*sin(X[i, 1])
        dX[i ,3] = β*sin(X[i, 2])
    end
    return dX
end


function exact_coef_test_model(psi::AbstractVector, coeffs::Tuple, n_dgl::Integer, 
    n_coord::Integer, decomp::AbstractString)
    p = length(psi)
    if decomp == "cm"
        sin_loc = findall(z -> z == sin, psi)
        cos_loc = findall(z -> z == cos, psi)
        one_loc = findall(z -> z == f_one, psi)
        M = zeros(Tuple([n_dgl; repeat([p], n_coord)]))

        α, β = coeffs

        # dgl 1
        idx = CartesianIndex(Tuple([1; sin_loc; cos_loc; repeat(one_loc, n_coord-2)]))
        M[idx] = α
        idx = CartesianIndex(Tuple([1; cos_loc; sin_loc; repeat(one_loc, n_coord-2)]))
        M[idx] = -α

        # dgl 2
        idx = CartesianIndex(Tuple([2; sin_loc; repeat(one_loc, n_coord-1)]))
        M[idx] = β

        # dgl 3
        idx = CartesianIndex(Tuple([3; one_loc; sin_loc; repeat(one_loc, n_coord-2)]))
        M[idx] = β

        return reshape(M, n_dgl, p^n_coord)'
    end
end


f_one(x) = 1


function generate_data_test_model(param::Dict, model::Function, derivative::Function)  
    dt = param["dt"]
    prob = ODEProblem(model, param["x0"], param["tspan"], param["coef"])
    sol = solve(prob, TRBDF2(), saveat=dt)
    X = Array(sol)'
    dX = derivative(X, param["coef"])
    return X, dX
end


# GENERATE TRAIN DATA
param = Dict(
    "tspan" => (0.0, 10),
    "dt" => 0.01,
    "x0" => [1.0; 0.5; 0.1],
    "coef" => (17.0, 5.0),
)

X_train, dX_train = generate_data_test_model(param, test_model1!, call_derivative_test_model)
X_train_noisy = add_gauss(X_train, sqrt(0.0001))

X_train_l = []
X_train_noisy_l = []
dX_train_l = []
timesteps = [10, 20, 50, 100, 200, 300, 400, 600, 800, 1000, 1200, 1400]
for m in timesteps
    param["tspan"] = (0.0, m*param["dt"]-param["dt"])
    local X_train, dX_train = generate_data_test_model(param, test_model!, call_derivative_test_model)
    local X_train_noisy = add_gauss(X_train, 0.001)
    push!(X_train_l, X_train)
    push!(X_train_noisy_l, X_train_noisy)
    push!(dX_train_l, dX_train)
end

# GENERATE TEST DATA
param_rec = Dict(
    "tspan" => (0.0, 10),
    "dt" => 0.05,
    "x0" => rand(3) .- 0.5,
    "coef" => (17.0, 5.0),
)

X_test_l = []
dX_test_l = []
for i in range(1, 9)
    X_test, dX_test = generate_data_test_model(param, test_model1!, call_derivative_test_model)
    push!(X_test_l, X_test)
    push!(dX_test_l, dX_test)
    param["x0"] = rand(3) .- 0.5
end


# Calculations
Xi_exact = exact_coef_test_model1(ψ, (17.0, 5.0), 3, 3, "cm")
ψ = [sin, cos, f_one]  # functionspace
λ = 10 .^(range(2, stop=-4, length=10))  # lasso path

Θ = coordinate_major(X_train_noisy, ψ)


# Ξ, train_err, rec_err_old = mals_lasso(Θ, dX_train, ψ, λ, X_test_l, dX_test_l)


# Ξ_l, train_err_l, rec_err_mean_l, rec_err_std_l, coef_err_l = val_diff_inits(Θ, dX_train, ψ, Xi_exact, X_test_l, dX_test_l, λ)

