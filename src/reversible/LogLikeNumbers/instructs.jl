@i @inline function :(*=)(+)(z::T, x::T, y::T) where T<:Tropical
    @invcheckoff if (content(x) > content(y), ~)
        (z |> content) += (x |> content)
    else
        (z |> content) += (y |> content)
    end
end

@i @inline function (:*=(identity))(x::T, y::T) where T<:LogLikeNumber
    (x |> content) += (y |> content)
end

@i @inline function (:*=(*))(out!::T, x::T, y::T) where T<:LogLikeNumber
    (out! |> content) += (x |> content) + (y |> content)
end


"""
add a number to out!.
"""
@i @inline function i_unsafe_addto(out!::T, x!::T, branch!::Bool) where T<:Tropical
	@invcheckoff if (content(out!) < content(x!), branch!)
		FLIP(branch!)
		NiLang.SWAP(out!, x!)
	end
end

# branch should be initialized to false.
@i @inline function i_muladd(out!::T, x::T, y::T, branch::Bool) where T<:Tropical
	x *= y
	@invcheckoff if (out! < x, branch)
		FLIP(branch)
		NiLang.SWAP(out!, x)
	end
end

maxloc(v::AbstractVector) = findmax(v)[2]

gaussian_log(x) = log1p(exp(x))
gaussian_nlog(x) = log1p(-exp(x))
@i function (:-=)(gaussian_log)(y!::GVar{T}, x::GVar{T}) where T
	y!.x -= gaussian_log(x.x)
	@routine @invcheckoff begin
		exp_x ← zero(x)
		jac ← zero(x)
		exp_x += exp(-x)
		exp_x += identity(T(1))
		jac += T(1)/exp_x
	end
	x.g += y!.g * jac
	~@routine
end

@i function (:-=)(gaussian_nlog)(y!::GVar{T}, x::GVar{T}) where T
	y!.x -= gaussian_nlog(x.x)
	@routine @invcheckoff begin
		exp_x ← zero(x)
		jac ← zero(x)
		exp_x += exp(-x)
		exp_x -= identity(T(1))
		jac -= T(1)/exp_x
	end
	x.g += y!.g * jac
	~@routine
end

@i function (:*=)(+)(out!::ULog{T}, x::ULog{T}, y::ULog{T}) where {T}
	@invcheckoff if (x.log == y.log, ~)
		out!.log += x.log
		out!.log += identity(T(log(2)))
	elseif (x.log ≥ y.log, ~)
		out!.log += x.log
		y.log -= x.log
		out!.log += gaussian_log(y.log)
		y.log += x.log
	else
		out!.log += y.log
		x.log -= y.log
		out!.log += gaussian_log(x.log)
		x.log += y.log
	end
end

@i function (:*=)(-)(out!::ULog{T}, x::ULog{T}, y::ULog{T}) where {T}
	@safe @assert x.log ≥ y.log
	@invcheckoff if (!iszero(x), ~)
		out!.log += x.log
		y.log -= x.log
		out!.log += gaussian_nlog(y.log)
		y.log += x.log
	end
end
