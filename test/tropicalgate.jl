using TropicalYao, LinearAlgebra
using Test, Random
using Yao
using NiLang

@testset "tropical block" begin
    reg = ArrayReg(Tropical.(randn(1<<10)))
    S = stack4reg(reg, 10)
    reg1, _, _, S = apply_G2!(copy(reg), 2, 0.5, S)
    reg2 = copy(reg) |> put(10, 2=>matblock(spinglass_bond_tensor(0.5)))
    @test reg1 ≈ reg2
    @test check_inv(apply_G2!, (copy(reg), 2, 0.5, S))
    reg1, _, _, S = apply_G4!(copy(reg), (4,2), 0.5, S)
    reg2 = copy(reg) |> put(10, (4,2)=>matblock(spinglass_g4_tensor(0.5)))
    @test reg1 ≈ reg2
    @test check_inv(apply_G4!, (copy(reg), (4,2), 0.5, S))
    Js = randn(16)
    reg1, _, _, S = apply_G16!(copy(reg), (4,1,2,9), Js, S)
    reg2 = copy(reg) |> put(10, (4,1,2,9)=>matblock(spinglass_g16_tensor(Js)))
    @test reg1 ≈ reg2
    @test check_inv(apply_G16!, (copy(reg), (4,1,2,9), Js, S))
    reg1, _, _, S = apply_Gh!(copy(reg), 2, 0.5, S)
    reg2 = copy(reg) |> put(10, 2=>matblock(Diagonal(spinglass_mag_tensor(0.5))))
    @test reg1 ≈ reg2
    @test check_inv(apply_Gh!, (copy(reg), 2, 0.5, S))
    reg1, _, S = apply_Gcp!(copy(reg), (2, 4), S)
    reg2 = copy(reg) |> put(10, (2, 4)=>matblock(copytensor(Tropical{Float64})))
    @test reg1 ≈ reg2
    @test check_inv(apply_Gcp!, (copy(reg), (2, 4), S))
    reg1, _, S = apply_Greset!(copy(reg), 2, S)
    reg2 = copy(reg) |> put(10, 2=>matblock(resettensor(Tropical{Float64})))
    @test reg1 ≈ reg2
    @test check_inv(apply_Greset!, (copy(reg), 2, S))
end
