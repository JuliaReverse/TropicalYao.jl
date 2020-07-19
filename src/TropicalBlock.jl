export tropicalblock, TropicalMatrixBlock

"""
    TropicalMatrixBlock{N, MT} <: PrimitiveBlock{N}

Tropical matrix gate wraps a tropical matrix operator to Yao blocks.
`N` is the hilbert dimension of current block.
"""
struct TropicalMatrixBlock{N,T,MT<:AbstractMatrix{T}} <: PrimitiveBlock{N}
    mat::MT
end

YaoBlocks.PutBlock{N,C,GT}(g::GT, locs) where {N, C, GT} = PutBlock{N}(g, locs)

function TropicalMatrixBlock{N}(m::MT) where {N,T,MT<:AbstractMatrix{T}}
    (1 << N, 1 << N) == size(m) ||
    throw(DimensionMismatch("expect a $(1<<N) x $(1<<N) matrix, got $(size(m))"))
    return TropicalMatrixBlock{N,T,MT}(m)
end

TropicalMatrixBlock(m::AbstractMatrix) = TropicalMatrixBlock{Int(log2(size(m,1)))}(m)

tropicalblock(m::AbstractMatrix{<:TropicalTypes}) = TropicalMatrixBlock(m)

YaoBlocks.mat(A::TropicalMatrixBlock) where T = A.mat

function YaoBlocks.mat(::Type{T}, A::TropicalMatrixBlock) where {T<:TropicalTypes}
    if eltype(A.mat) == T
        return A.mat
    else
        @warn "converting $(eltype(A.mat)) to eltype $T, consider create another matblock with eltype $T"
        return copyto!(similar(A.mat, T), A.mat)
    end
end

Base.:(==)(A::TropicalMatrixBlock, B::TropicalMatrixBlock) = A.mat == B.mat
Base.copy(A::TropicalMatrixBlock) = TropicalMatrixBlock(copy(A.mat))
Base.adjoint(x::TropicalMatrixBlock) = Daggered(x)
function Base.show(io::IO, b::TropicalMatrixBlock{N}) where N
    println(io, "TropicalMatrixBlock{$N}")
    for i=1:size(b.mat, 1)
        print(io, " ")
        for j=1:size(b.mat, 1)
            print(io, b.mat[i,j].n, "  ")
        end
        println(io)
    end
end
Base.show(io::IO, ::MIME"text/plain", b::TropicalMatrixBlock) = Base.show(io, b)
