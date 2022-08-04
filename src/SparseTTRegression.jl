module SparseTTRegression

using ITensors
using Lasso
using Printf
using LinearAlgebra
using Plots

include("utils.jl")
export Array
export Matrix
export make_random_matrix_vector_mps
export print_error
export plot_one_dgl
include("create_library_matrix.jl")
export function_major
export coordinate_major
include("tt_lasso_mals.jl")
export mals_lasso
include("math.jl")
export pinv
include("optimization.jl")
export sparse_TT_optimization
export mandy
export sindy

end
