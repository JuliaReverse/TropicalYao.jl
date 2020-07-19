using TropicalNumbers, TropicalYao.Reversible.LogLikeNumbers
using Test, NiLang.AD, NiLang, Random
using ForwardDiff

@testset "GVar" begin
    x = Tropical(0.4)
	@test (~Tropical)(x) == 0.4
    @test GVar(x) isa Tropical{<:GVar}
    @test grad(x) isa Tropical{Float64}
    @test (~GVar)(GVar(Tropical(0.4))) == x

    x = GVar(Tropical(0))
    @test zero(x) isa Tropical{GVar{Int, Int}}

    x = ULog(0.4)
	@test (~ULog)(x) == 0.4
    @test GVar(x) isa ULog{<:GVar}
    @test grad(x) isa ULog{Float64}
    @test (~GVar)(GVar(ULog(0.4))) == x
end

@testset "basic instructions" begin
	x, y = Tropical(1), Tropical(2)
	@instr x *= y
	@test x == Tropical(3)
	@test y == Tropical(2)
	x, y, bk = Tropical(3), Tropical(2), false
	@instr i_unsafe_addto(y, x, bk)
	@test x == Tropical(2)
	@test y == Tropical(3)
	@test bk == true
end

@testset "basic instructions, ULog" begin
	x, y = ULog(1), ULog(2)
	@instr x *= y
	@test x == ULog(3)
	@test y == ULog(2)

	@test PlusEq(gaussian_log)(1.0, 2.0) == (1.0+log(1+exp(2.0)), 2.0)
	@test check_grad(PlusEq(gaussian_log), (1.0, 2.0); iloss=1)

	x, y,z = ULog(7.0), ULog(2.0), ULog(3.0)
	@instr x *= y + z
	@test check_inv(MulEq(+), (x, y, z))
	@test x.log ≈ log(exp(7.0) * (exp(2.0) + exp(3.0)))
	x, y,z = ULog(7.0), ULog(5.0), ULog(3.0)
	@instr x *= y - z
	@test x.log ≈ log(exp(7.0) * (exp(5.0) - exp(3.0)))
	function muleq(f, x, y, z)
		x = ULog(x)
		y = ULog(y)
		z = ULog(z)
		x *= f(y, z)
		x.log
	end
	g1 = ForwardDiff.gradient(arr->muleq(+, arr...), [7.0, 5.0, 3.0])
	x, y,z = ULog(7.0), ULog(5.0), ULog(3.0)
	@instr (MulEq(+))(x, y, z)
	@instr GVar(x)
	@instr GVar(y)
	@instr GVar(z)
	@instr x.log.g += 1
	@instr (~MulEq(+))(x, y, z)
	@test grad(x.log) ≈ g1[1]
	@test grad(y.log) ≈ g1[2]
	@test grad(z.log) ≈ g1[3]

	g2 = ForwardDiff.gradient(arr->muleq(-, arr...), [7.0, 5.0, 3.0])
	x, y,z = ULog(2.0), ULog(5.0), ULog(3.0)
	@instr (MulEq(-))(x, y, z)
	@instr GVar(x)
	@instr GVar(y)
	@instr GVar(z)
	@instr x.log.g += 1
	@instr (~MulEq(-))(x, y, z)
	@test grad(x.log) ≈ g2[1]
	@test grad(y.log) ≈ g2[2]
	@test grad(z.log) ≈ g2[3]
end
