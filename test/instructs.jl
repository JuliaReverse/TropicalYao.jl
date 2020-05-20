using TropicalYao
using Test
using LinearAlgebra
using Yao, NiLang

@testset "new instr" begin
    g4 = Diagonal(spinglass_g4_tensor(1.5))
    reg = ArrayReg(randn(1<<12) .|> Tropical)
    S = stack4reg(reg, 0)
    s1 = i_instruct!(copy(vec(reg.state)), g4, (3, 7), (), (), S)[1]
    nreg = copy(reg) |> put(12, (3, 7)=>matblock(g4))
    @test statevec(nreg) ≈ s1

    g4 = randn(4, 4) .|> Tropical
    reg = ArrayReg(randn(1<<12) .|> Tropical)
    S = stack4reg(reg, 1)
    s1 = i_instruct!(copy(vec(reg.state)), g4, (3, 7), (), (), S)[1]
    nreg = copy(reg) |> put(12, (3, 7)=>matblock(g4))
    @test statevec(nreg) ≈ s1
end

@testset "stack" begin
    a = one(Tropical{Float64})
    b = Tropical(randn())
    @instr TropicalYao.unsafe_store!(a, b)
    @test a == b
    @instr (~TropicalYao.unsafe_store!)(a, b)
    @test a == one(Tropical{Float64})
    @test check_inv(TropicalYao.unsafe_store!, (a, b))
end
