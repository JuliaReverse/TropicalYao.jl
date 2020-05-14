using TropicalYao
using Test

@testset "blocks" begin
    include("blocks.jl")
end

@testset "instructions" begin
    include("instructs.jl")
end

@testset "tropicalgate" begin
    include("tropicalgate.jl")
end
