using SparseTTRegression
using Test

@testset "SparseTTRegression.jl" begin
    @testset "Matrix and Array" begin
        include("test_Matrix_Array.jl")
    end
    @testset "Inverse" begin
        include("test_inverse.jl")
    end
end