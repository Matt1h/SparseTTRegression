module SparseTTRegression

using ITensors
using Lasso
using LinearAlgebra

include("library_matrix.jl")
export coordinate_major
include("tt_lasso_mals.jl")
export mals_lasso
include("math.jl")
export Array
export Matrix
export pinv
export *
export rec
export mae
export mape
include("optimization.jl")
export mals_lasso
export mals_lasso
export truncated_mals_lasso
export mandy
export sindy
export regular_lasso
end
