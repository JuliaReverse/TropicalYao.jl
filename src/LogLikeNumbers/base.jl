export i_gemm!, i_gemv!, i_sum, i_unsafe_addto, i_muladd
export maxloc, gaussian_log

export ULog
const ULog{T} = ULogarithmic{T}
const LogLikeNumber{T} = Union{ULog{T}, Tropical{T}}

const TropicalG{T,TG} = Tropical{GVar{T,TG}}
const ULogG{T,TG} = ULogarithmic{GVar{T,TG}}

# Happy Pirate!!!!!
ULog{T}(gv::T) where T<:Real = exp(ULogarithmic{T}, gv)
ULog(gv::T) where T<:Real = exp(ULogarithmic, gv)

import TropicalNumbers: content
@fieldview content(x::ULogarithmic) = x.log
NiLang.chfield(x::Tropical{T}, ::typeof(content), val::T) where T = Tropical{T}(val)

# AD wrappers
for T in [:Tropical, :ULogarithmic]
    @eval NiLang.AD.GVar(x::$T) = $T(GVar(content(x), zero(content(x))))
    @eval (_::Type{Inv{$T}})(x::$T) = content(x)
    @eval NiLang.AD.grad(x::$T{<:GVar}) = $T(grad(content(x)))
    @eval (_::Type{Inv{GVar}})(x::$T{<:GVar}) = $T((~GVar)(content(x)))

    @eval Base.one(x::$T{GVar{T,GT}}) where {T, GT} = one($T{GVar{T,GT}})
    @eval Base.one(::Type{$T{GVar{T,GT}}}) where {T,GT} = $T(GVar(zero(T), zero(GT)))
    @eval Base.zero(x::$T{GVar{T,GT}}) where {T,GT} =zero($T{GVar{T,GT}})
    @eval Base.zero(::Type{$T{GVar{T,T}}}) where T = $T(GVar(zero(T), zero(T)))
end

function NiLang.loaddata(::Type{Array{<:LogLikeNumber{GVar{T,T}}}}, data::Array{<:LogLikeNumber{T}}) where {T}
    GVar.(data)
end
import NiLang.NiLangCore: deanc

function deanc(x::T, v::T) where T<:LogLikeNumber
    x === v || deanc(content(x), content(v))
end
