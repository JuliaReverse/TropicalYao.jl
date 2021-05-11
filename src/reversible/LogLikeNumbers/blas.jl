# out initialized to one(Tropical)
@i function i_gemm!(out!::AbstractMatrix{T}, x::AbstractMatrix{T}, y::AbstractMatrix{T}) where T<:Tropical
	@safe size(x, 2) == size(y, 1) || throw(DimensionMismatch())
	@invcheckoff branch_keeper ← zeros(Bool, size(x,2))
	@invcheckoff for j=1:size(y,2)
		for i=1:size(x,1)
			el ← zero(T)
			@routine @inbounds for k=1:size(x,2)
				x[i,k] *= y[k,j]
				if (el.n < x[i,k].n, branch_keeper[k])
					FLIP(branch_keeper[k])
					NiLang.SWAP(el, x[i,k])
				end
			end
			@inbounds out![i,j] *= el
			~@routine
			el → zero(T)
		end
	end
	@invcheckoff branch_keeper → zeros(Bool, size(x,2))
end

@i function i_gemv!(out!::AbstractVector{T}, x::AbstractMatrix{T}, y::AbstractVector{T}, branch_keeper::AbstractVector{Bool}) where T<:Tropical
	@safe size(x, 2) == size(y, 1) || throw(DimensionMismatch())
	@invcheckoff for i=1:size(x,1)
		el ← zero(T)
		@routine @inbounds for k=1:size(x,2)
			x[i,k] *= y[k]
			if (el.n < x[i,k].n, branch_keeper[k])
				FLIP(branch_keeper[k])
				NiLang.SWAP(el, x[i,k])
			end
		end
		@inbounds out![i] *= el
		~@routine
		el → zero(T)
	end
end

@i function i_sum(out!::T, v::AbstractArray{T}) where T<:Tropical
	@routine @invcheckoff begin
		branch_keeper ← zeros(Bool, length(v))
		anc ← zero(T)
		for i = 1:length(v)
			@inbounds if (anc.n < v[i].n, branch_keeper[i])
				FLIP(branch_keeper[i])
				NiLang.SWAP(anc, v[i])
			end
		end
	end
	out!.n += anc.n
	~@routine
end
