using TropicalYao.Reversible
using Test

@testset "LogLikeNumbers" begin
    include("LogLikeNumbers/LogLikeNumbers.jl")
end

@testset "spinglass_gates" begin
    include("spinglass_gates.jl")
end

@testset "instructions" begin
    include("instructs.jl")
end
