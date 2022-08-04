using DifferentialEquations
using ITensors
using LinearAlgebra
using SparseTTRegression
using Plots
using BenchmarkTools


function test_model1!(dx, x, coeff, t)
    alpha, beta = coeff 
    dx[1] = alpha*sin(x[1] - x[2])
    dx[2] = beta*sin(x[1])
    dx[3] = beta*sin(x[2])
end


function test_model2!(dx, x, coeff, t)
    alpha, beta = coeff 
    dx[1] = alpha*sin(x[1] - x[2])
    dx[2] = beta*sin(x[1])
    dx[3] = beta*sin(x[2])
    dx[4] = alpha*sin(x[3] - x[4])
    dx[5] = beta*sin(x[5])
    dx[6] = beta*cos(x[6])
end


function CalDerivative(u, dt)
    #  u: Measurement data we wish to approxiamte the derivative. 
    #  It should be of size m x n, where m is the number of states, n is the number of measurement.
    # dt: Time step
    # return du: The approximated derivative. 
    
    # Define the coeficient for different orders of derivative
    p1=1/12;p2=-2/3;p3=0;p4=2/3;p5=-1/12;
    
    du=(p1*u[1:end-4,:]+p2*u[2:end-3,:]+p3*u[3:end-2,:]+p4*u[4:end-1,:]+p5*u[5:end,:])/dt;
        
    return du
end


# function call_derivative_test_model(X::AbstractMatrix{T}, coeff::Tuple) where{T}
#     α, β = coeff
#     m, n = size(X)
#     dX = zeros(T, m, n)
#     for i in range(1, m)
#         dX[i, 1] = alpha*sin(X[i, 1] - X[i, 2])
#         dX[i, 2] = beta*sin(X[i, 1])
#         dX[i ,3] = beta*sin(X[i, 2])
#     end
#     return dX
# end


function generate_data(param::Dict, model::Function, derivative::Function; max_iter=1::Integer)
    dt = param["dt"]
    n = length(param["x0"])
    X = []
    iter = 0
    while iter < max_iter
        prob = ODEProblem(model, param["x0"], param["tspan"], param["coef"])
        X_n = Array(solve(prob, saveat=dt))'
        push!(X, X_n)
        iter += 1
        param["x0"] = rand(n) .- 0.5
    end
    X = vcat(X...)
    dX = derivative(X, dt)
    return X[3:end-2,:], dX
end


function exact_coef_test_model1(psi::AbstractVector, coeffs::Tuple, n_dgl::Integer, 
    n_coord::Integer, decomp::AbstractString)
    p = length(psi)
    if decomp == "cm"
        sin_loc = findall(z -> z == sin, psi)
        cos_loc = findall(z -> z == cos, psi)
        one_loc = findall(z -> z == f_one, psi)
        M = zeros(typeof(coef), Tuple([n_dgl; repeat([p], n_coord)]))

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

        return reshape(M, p^n_coord)
    end
end


f_one(x) = 1


# test model parameter
param = Dict(
    "tspan" => (0.0, 10),
    "dt" => 0.01,
    "x0" => [1.0; 0.9; 0.3],
    # "x0" => rand(3) .- 0.5,
    "coef" => (17.0, 5.0),
)

X, dX = generate_data(param, test_model1!, CalDerivative; max_iter=1)

# FUNCTIONSPACE
ψ = [sin, cos, f_one]
Θ = coordinate_major(X, ψ)

# LAMBDA
λ = [10, 1, 0.1, 0.01, 0.001, 0.0001]

Ξ = sparse_TT_optimization(Θ, dX,  λ)
Ξ_mandy = mandy(Θ, dX)
Xi_sindy = sindy(Matrix(Θ), dX, 0.01, 100)


# @benchmark Ξ = sparse_TT_optimization(Θ, dX,  λ; tol=1e-6)
# @benchmark Ξ_mandy = mandy(Θ, dX)
# @benchmark Xi_sindy = sindy(Matrix(Θ), dX, 0.1, 100)

Theta = Matrix(Θ)
print("Relative error ALS:", norm(Theta*hcat([Matrix(Ξ[i]) for i in range(1, length(Ξ))]...) - dX)/norm(dX)*100, "\n")
print("Relative error MANDy:", norm(Theta*Matrix(Ξ_mandy) - dX)/norm(dX)*100, "\n")
print("Relative error SINDy:", norm(Theta*Xi_sindy - dX)/norm(dX)*100, "\n")

# RECONSTRUCT DERIVATIVE
param_rec = Dict(
    "tspan" => (0.0, 10),
    "dt" => 0.05,
    "x0" => rand(3) .- 0.5,
    "coef" => (17.0, 5.0),
)

X_rec, dX_rec = generate_data(param_rec, test_model1!, CalDerivative; max_iter=1)
Θ_rec = coordinate_major(X_rec, ψ)
Theta_rec = Matrix(Θ_rec)
dX_rec_mandy = Theta_rec*Matrix(Ξ_mandy)
dX_rec_als = Theta_rec*hcat(Matrix(Ξ[1]), Matrix(Ξ[2]), Matrix(Ξ[3]))
dX_rec_sindy = Theta_rec*Matrix(Xi_sindy)
print("Relative reconstruction error ALS:", norm(dX_rec_als - dX_rec)/norm(dX_rec)*100, "\n")
print("Relative reconstruction error MANDy:", norm(dX_rec_mandy - dX_rec)/norm(dX_rec)*100, "\n")
print("Relative reconstruction error SINDy:", norm(dX_rec_sindy - dX_rec)/norm(dX_rec)*100, "\n")