@i @inline function :(*=)(+)(z::Tropical, x::Tropical, y::Tropical)
    @invcheckoff if (content(x) > content(y), ~)
        (z |> content) += (x |> content)
    else
        (z |> content) += (y |> content)
    end
end

@i @inline function (:*=(identity))(x::Tropical, y::Tropical)
    (x |> content) += (y |> content)
end

@i @inline function (:*=(*))(out!::Tropical, x::Tropical, y::Tropical)
    (out! |> content) += (x |> content) + (y |> content)
end


"""
add a number to out!.
"""
@i @inline function i_unsafe_addto(out!::Tropical, x!::Tropical, branch!::Bool)
	@invcheckoff if (content(out!) < content(x!), branch!)
		FLIP(branch!)
		NiLang.SWAP(out!, x!)
	end
end

# branch should be initialized to false.
@i @inline function i_muladd(out!::Tropical, x::Tropical, y::Tropical, branch::Bool)
	x *= y
	@invcheckoff if (out! < x, branch)
		FLIP(branch)
		NiLang.SWAP(out!, x)
	end
end

maxloc(v::AbstractVector) = findmax(v)[2]
