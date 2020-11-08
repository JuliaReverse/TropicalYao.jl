using TropicalYao.Reversible
using TropicalNumbers
using LinearAlgebra, Test
using Suppressor
using Yao


@testset "i-bond tensor" begin
    mat = ones(Tropical{Float64}, 2, 2)
    J = 0.3
    @test i_bondtensor!(mat, J)[1] ≈ i_bondtensor(J)
    mat = Diagonal(ones(Tropical{Float64}, 4))
    @test i_gvb_tensor!(mat, J)[1] ≈ i_gvb_tensor(J)
    mat = ones(Tropical{Float64}, 16, 16)
    Js = randn(16)
    @test i_g16_tensor!(mat, Js)[1] ≈ i_g16_tensor(Js)
end

@testset "tropical block" begin
    reg = @suppress ArrayReg(Tropical.(randn(1<<10)))
    S = stack4reg(reg, 10)
    reg1, _, _, S = apply_Ghb!(copy(reg), 2, 0.5, copy(S))
    reg2 = copy(reg) |> put(10, 2=>matblock(i_bondtensor(0.5)))
    @test reg1 ≈ reg2
    @test check_inv(apply_Ghb!, (copy(reg), 2, 0.5, S))
    reg1, _, _, S = apply_Gvb!(copy(reg), (2,4), 0.5, S)
    reg2 = copy(reg) |> put(10, (2,4)=>matblock(i_gvb_tensor(0.5)))
    @test reg1 ≈ reg2
    @test check_inv(apply_Gvb!, (copy(reg), (2,4), 0.5, S))
    Js = randn(16)
    reg1, _, _, S = apply_G16!(copy(reg), (1,2,4,9), Js, S)
    reg2 = copy(reg) |> put(10, (1,2,4,9)=>matblock(i_g16_tensor(Js)))
    @test reg1 ≈ reg2
    @test check_inv(apply_G16!, (copy(reg), (1,2,4,9), Js, S))
    reg1, _, _, S = apply_Gh!(copy(reg), 2, 0.5, S)
    reg2 = copy(reg) |> put(10, 2=>matblock(Diagonal(i_magtensor(0.5))))
    @test reg1 ≈ reg2
    @test check_inv(apply_Gh!, (copy(reg), 2, 0.5, S))
    reg1, _, S = apply_Gcp!(copy(reg), (2, 4), S)
    reg2 = copy(reg) |> put(10, (2, 4)=>matblock(copytensor(Tropical{Float64})))
    @test reg1 ≈ reg2
    @test check_inv(apply_Gcp!, (copy(reg), (2, 4), S))
    reg1, _, S = apply_Greset!(copy(reg), 2, copy(S))
    reg2 = copy(reg) |> put(10, 2=>matblock(resettensor(Tropical{Float64})))
    @test reg1 ≈ reg2
    @test check_inv(apply_Greset!, (copy(reg), 2, S))
end
