using DifferentialEquations
using ITensors
using LinearAlgebra
using SparseTTRegression


function fput!(ddX, x, β, t)
    n = length(x)
    ddX[1] = x[2] - 2*x[1] + β*((x[2] - x[1])^3 - x[1]^3)
    ddX[n] = 2*x[n] + x[n-1] + β*(x[n]^3 - (x[n]- x[n-1])^3)
    for i in 2:n-1
		ddX[i] = x[i+1] - 2*x[i] + x[i-1] + β*((x[i+1] - x[i])^3 - (x[i] - x[i-1])^3)
	end
end


function call_derivative_fput(X::AbstractMatrix{T}, β::Number) where{T}
    m, n = size(X)
    ddX = zeros(T, m, n)
    for i in range(1, m)
        ddX[i, 1] = X[i, 2] - 2*X[i, 1] + β*((X[i, 2] - X[i, 1])^3 - X[i, 1]^3)
        ddX[i, n] = 2*X[i, n] + X[i, n-1] + β*(X[i, n]^3 - (X[i, n]- X[i, n-1])^3)
        for j in 2:n-1
            ddX[i, j] = X[i, j+1] - 2*X[i, j] + X[i, j-1] + β*((X[i, j+1] - X[i, j])^3 - (X[i, j] - X[i, j-1])^3)
        end
    end
    return ddX
end


function generate_data_fput(param::Dict, max_iter::Integer)
    dt = param["dt"]
    n = length(param["x0"])
    X = []
    ddX = []
    iter = 0
    while iter <= max_iter
        param["x0"] = rand(n)
        prob = ODEProblem(fput!, param["x0"], param["tspan"], param["beta"])
        X_n = Array(solve(prob, saveat=dt))'
        ddX_n = call_derivative_fput(X_n, param["beta"])
        push!(X, X_n)
        push!(ddX, ddX_n)
        iter += 1
    end
    X = vcat(X...)
    ddX = vcat(ddX...)
    return X, ddX
end

# MODEL PARAMETER
n = 10
param = Dict(
    "tspan" => (0.0, 0.2),  # TODO: higher t_max doesnt work, why?
    "dt" => 0.002,
    "x0" => rand(n),
    "beta" => 0.7,
)

# GENERATE DATA
X, ddX = generate_data_fput(param, 5)

# FUNCTIONSPACE
ψ = [x -> 1, x->x, x->x^2, x->x^3]
Θ = coordinate_major(X, ψ)

# LAMBDA
λ = [10, 1, 0.1, 0.01, 0.001, 0.0001]

Ξ = sparse_TT_optimization(Θ, ddX,  λ, sweep_count=10)
Ξ_mandy = mandy(Θ, ddX)
Xi_sindy = sindy(Matrix(Θ), ddX, 0.0000001, 20)
Theta = Matrix(Θ)
print("Relative error ALS:", norm(Theta*hcat([Matrix(Ξ[i]) for i in range(1, length(Ξ))]...) - ddX)/norm(ddX)*100, "\n")
print("Relative error MANDy:", norm(Theta*Matrix(Ξ_mandy) - ddX)/norm(ddX)*100, "\n")
print("Relative error SINDy:", norm(Theta*Xi_sindy - ddX)/norm(ddX)*100, "\n")


# MODEL PARAMETER
n = 10
param_rec = Dict(
    "tspan" => (0.0, 0.2),  # TODO: higher t_max doesnt work, why?
    "dt" => 0.002,
    "x0" => rand(n),
    "beta" => 0.7,
)

X_rec, ddX_rec = generate_data_fput(param_rec)
Θ_rec = coordinate_major(X_rec, ψ)
Theta_rec = Matrix(Θ_rec)
ddX_rec_als = Theta_rec*hcat([Matrix(Ξ[i]) for i in range(1, length(Ξ))]...)
ddX_rec_mandy = Theta_rec*Matrix(Ξ_mandy)
ddX_rec_sindy = Theta_rec*Matrix(Xi_sindy)
print("Relative reconstruction error ALS:", norm(ddX_rec_als - ddX_rec)/norm(ddX_rec)*100, "\n")
print("Relative reconstruction error MANDy:", norm(ddX_rec_mandy - ddX_rec)/norm(ddX_rec)*100, "\n")
print("Relative reconstruction error SINDy:", norm(ddX_rec_sindy - ddX_rec)/norm(ddX_rec)*100, "\n")

