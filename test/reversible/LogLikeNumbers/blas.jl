using TropicalYao.Reversible.LogLikeNumbers

@testset "Tropical algebra" begin
	Random.seed!(4)
	@test check_inv(i_muladd, (Tropical(2.0), Tropical(3.0), Tropical(4.0), false))
	@test check_inv(i_muladd, (Tropical(2.0), Tropical(3.0), Tropical(-5.0), false))
	x = Tropical.(randn(100, 100))
	y = Tropical.(randn(100, 100))
	v = Tropical.(randn(100))
	out1 = x * y
	out2 = one.(out1)
	i_gemm!(out2, x, y)
	out4 = Tropical.(zeros(100))
	out3 = x * v
	bk = zeros(Bool, 100)
	i_gemv!(out4, x, v, bk)
	@test out1 ≈ out2
	@test out3 ≈ out4
end

@testset "i_sum" begin
	i_sum(TropicalF32(0), TropicalF32.([3.0,10.0,2.0]))[1].n == 10
end
