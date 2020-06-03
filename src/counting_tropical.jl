struct CountingTropical{T} <: Number
    n::T
    c::Int32
end
CountingTropical(x::Real) = CountingTropical(x, Int32(1))
CountingTropical{T1}(x::T2) where {T1, T2} = CountingTropical{T1}(T1(x), Int32(1))

Base.:*(a::CountingTropical, b::CountingTropical) = CountingTropical(a.n + b.n, a.c * b.c)
function Base.:+(a::CountingTropical, b::CountingTropical)
    n = max(a.n, b.n)
    if a.n > b.n
        c = a.c
    elseif a.n == b.n
        c =  a.c + b.c
    else
        c = b.c
    end
    CountingTropical(n, c)
end
Base.zero(::Type{CountingTropical{T}}) where T<:Integer = CountingTropical(T(-999999), Int32(1))
Base.zero(::Type{CountingTropical{T}}) where T<:AbstractFloat = CountingTropical(typemin(T), Int32(1))
Base.one(::Type{CountingTropical{T}}) where T<:Integer = CountingTropical(zero(T), Int32(1))
Base.one(::Type{CountingTropical{T}}) where T<:AbstractFloat = CountingTropical(zero(T), Int32(1))

TropicalNumbers.content(x::CountingTropical) = x.n

using NiLang
(_::Type{Inv{CountingTropical}})(x::CountingTropical) = content(x)

TropicalTypes{T} = Union{CountingTropical{T}, Tropical{T}}