using TropicalYao.LogLikeNumbers
using Test

@testset "base" begin
    include("base.jl")
end

@testset "blas" begin
    include("blas.jl")
end

@testset "einsum" begin
    include("einsum.jl")
end
