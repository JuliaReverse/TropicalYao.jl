function bondtensor(::Type{TT}, J) where TT
    TT.([J -J; -J J])
end

function vertextensor(::Type{TT}, h) where TT
    TT.([h, -h])
end

function copytensor(::Type{T}) where T
    Diagonal([one(T), T(-999999), T(-999999), one(T)])
end

function resettensor(::Type{T}) where T
    [one(T) one(T); zero(T) zero(T)]
end

"""
    hypercubicI([T], ndim::Int, D::Int)

Get a `ndim`-dimensional identity hypercubic, with bound dimension `D`.
"""
function hypercubicI(::Type{T}, ndim::Int, D::Int) where T
    res = zeros(T, fill(D, ndim)...)
    @inbounds for i=1:D
        res[fill(i,ndim)...] = one(T)
    end
    return res
end

hypercubicI(ndim::Int, D::Int) = hypercubicI(Float64, ndim, D)

Gh(vertex_tensor::Vector{T}) where T = tropicalblock(Diagonal(vertex_tensor) |> LuxurySparse.staticize)
Gvb(bond_tensor::Matrix{T}) where T = tropicalblock(Diagonal([bond_tensor...]) |> LuxurySparse.staticize)
Ghb(bond_tensor::Matrix{T}) where T = tropicalblock(transpose(bond_tensor) |> LuxurySparse.staticize)
function G16(::Type{TT}, Js) where TT<:TropicalTypes
    xs = map(x->_bondtensor(TT, x), Js)
    mat = zeros(TT, 2, 2, 2, 2, 2, 2, 2, 2)
    for a=1:2, α=1:2, β=1:2, b=1:2, γ=1:2, c=1:2, d=1:2, δ=1:2
        mat[a,b,c,d,α,β,γ,δ] += xs[1][a,α] * xs[2][a,β] * xs[3][a,γ] * xs[4][a,δ] *
        xs[5][b,α] * xs[6][b,β] * xs[7][b,γ] * xs[8][b,δ] * xs[9][c,α] * xs[10][c,β] *
        xs[11][c,γ] * xs[12][c,δ] * xs[13][d,α] * xs[14][d,β] * xs[15][d,γ] * xs[16][d,δ]
    end
    tropicalblock(reshape(mat, 16, 16) |> LuxurySparse.staticize)
end

function _bondtensor(::Type{TT}, J) where TT
    TT.([J -J; -J J])
end

"""
    Gcp(T)

copy state of qubit 2 -> 1.
"""
function Gcp(::Type{TT}) where TT<:TropicalTypes
    tropicalblock(copytensor(TT))
end

function Greset(::Type{TT}) where TT<:TropicalTypes
    tropicalblock([one(TT) one(TT); zero(TT) zero(TT)])
end

function Gcut(::Type{TT}) where TT<:TropicalTypes
    tropicalblock([one(TT) one(TT); one(TT) one(TT)])
end
