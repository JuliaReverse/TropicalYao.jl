export spinglass_mag_tensor!, spinglass_g4_tensor!, spinglass_bond_tensor!, spinglass_g16_tensor!
export spinglass_mag_tensor, spinglass_g4_tensor, spinglass_bond_tensor, spinglass_g16_tensor
export copytensor, resettensor

"""
    spinglass_mag_tensor!(v, Jij)

`v` should be a length 2 vector.
"""
@i function spinglass_mag_tensor!(v::AbstractVector, h::Real)
    @safe @assert size(v) == (2,)
    Tropical(h)
    v[1] *= identity(h)
    v[2] /= identity(h)
    (~Tropical)(h)
end

"""
    spinglass_bond_tensor!(mat, Jij)

`mat` should be a `one` tensor.
"""
@i function spinglass_bond_tensor!(mat::AbstractMatrix, Jij::Real)
    @safe @assert size(mat) == (2,2)
    Tropical(Jij)
    mat[1,1] *= identity(Jij)
    mat[2,2] *= identity(Jij)
    mat[1,2] /= identity(Jij)
    mat[2,1] /= identity(Jij)
    (~Tropical)(Jij)
end

@i function spinglass_g4_tensor!(mat::Diagonal, Jij::Real)
    @safe @assert size(mat) == (4,4)
    Tropical(Jij)
    mat[1,1] *= identity(Jij)
    mat[2,2] /= identity(Jij)
    mat[3,3] /= identity(Jij)
    mat[4,4] *= identity(Jij)
    (~Tropical)(Jij)
end

@i function spinglass_g16_tensor!(out!::AbstractMatrix{T}, Js) where T<:Tropical
    @safe @assert length(Js) == 16
    @safe @assert size(out!) == (16, 16)
    @routine begin
        y ← ones(T,2,2,2,2,2,2,2,2)
        xs ← ([ones(T,2,2) for i=1:16]...,)
        for i = 1:length(Js)
            spinglass_bond_tensor!(tget(xs,i), Js[i])
        end
        ixs ← ([(ix...,) for ix in split("aα,aβ,aγ,aδ,bα,bβ,bγ,bδ,cα,cβ,cγ,cδ,dα,dβ,dγ,dδ", ',')]...,)
        iy ← ("abcdαβγδ"...,)
        NiLogLikeNumbers.einsum!(ixs, xs, iy, y)
    end
    for i=1:length(out!)
        out![i] *= identity(y[i])
    end
    ~@routine
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

spinglass_bond_tensor(Jij::T) where T<:Real = spinglass_bond_tensor!(ones(Tropical{T}, 2, 2), Jij)[1]
spinglass_mag_tensor(h::T) where T<:Real = spinglass_mag_tensor!(ones(Tropical{T}, 2), h)[1]
spinglass_g4_tensor(Jij::T) where T<:Real = spinglass_g4_tensor!(Diagonal(ones(Tropical{T}, 4)), Jij)[1]
spinglass_g16_tensor(Js::Vector{T}) where T<:Real = spinglass_g16_tensor!(ones(Tropical{T}, 16, 16), Js)[1]

"""
    apply_G2!(reg::ArrayReg, i::Int, J::Real, REG_STACK)

Apply a spin glass bond tensor (parametrized by coupling `J`) on site `i`.
This instruct will increase the stack top of `REG_STACK` by 1.
"""
@i function apply_G2!(reg::ArrayReg{1,T}, i::Int, J::Real, REG_STACK) where T<:Tropical
    @routine @invcheckoff begin
        nbit ← nqubits(reg)
        blk ← put(nbit, i=>tropicalblock(MMatrix{2,2}(ones(T, 2, 2))))
        spinglass_bond_tensor!(blk.content.mat, J)
    end
    apply!(reg, blk, REG_STACK)
    ~@routine
end

"""
    apply_Gh!(reg::ArrayReg, i::Int, h::Real, REG_STACK)

Apply a spin glass magnetic tensor (parametrized by coupling `h`) on site `i`.
This instruct will increase the stack top of `REG_STACK` by 0.
"""
@i function apply_Gh!(reg::ArrayReg{1,T}, i::Int, h::Real, REG_STACK) where T<:Tropical
    @routine @invcheckoff begin
        nbit ← nqubits(reg)
        blk ← put(nbit, i=>tropicalblock(Diagonal(MVector{2}(ones(T, 2)))))
        spinglass_mag_tensor!(blk.content.mat.diag, h)
    end
    apply!(reg, blk, REG_STACK)
    ~@routine
end

"""
    apply_G4!(reg::ArrayReg, i::NTuple{2,Int}, J::Real, REG_STACK)

Apply a vertical spin glass bond tensor (parametrized by coupling `J`) on sites `i[1]` and `i[2]`.
This instruct will increase the stack top of `REG_STACK` by 0.
"""
@i function apply_G4!(reg::ArrayReg{1,T}, i::NTuple{2,Int}, J::Real, REG_STACK) where T<:Tropical
    @routine @invcheckoff begin
        nbit ← nqubits(reg)
        blk ← put(nbit, i=>tropicalblock(Diagonal(MVector{4}(ones(T, 4)))))
        spinglass_g4_tensor!(blk.content.mat, J)
    end
    apply!(reg, blk, REG_STACK)
    ~@routine
end

"""
    apply_G16!(reg::ArrayReg, i::NTuple{4,Int}, Js::AbstractVector, REG_STACK)

Apply a spin glass tensor of K(4,4) graph (the intra-block coupling term in the Chimera graph that parametrized by 16 coupling terms `Js`) on sites `i[1:4]`.
This instruct will increase the stack top of `REG_STACK` by 1.
"""
@i function apply_G16!(reg::ArrayReg{1,T}, i::NTuple{4,Int}, Js::AbstractVector{<:Real}, REG_STACK) where T<:Tropical
    @routine @invcheckoff begin
        nbit ← nqubits(reg)
        blk ← put(nbit, i=>tropicalblock(MMatrix{16,16}(ones(T, 16, 16))))
        spinglass_g16_tensor!(blk.content.mat, Js)
    end
    apply!(reg, blk, REG_STACK)
    ~@routine
end

"""
    apply_Gcp!(reg::ArrayReg, i::NTuple{2,Int}, REG_STACK)

Apply a copy gate (or CNOT).
This instruct will increase the stack top of `REG_STACK` by 1.
"""
@i function apply_Gcp!(reg::ArrayReg{1,T}, i::NTuple{2,Int}, REG_STACK) where T<:Tropical
    @routine @invcheckoff g ← put(nqubits(reg), i=>tropicalblock(copytensor(T)))
    apply!(reg, g, REG_STACK)
    ~@routine
end

"""
    apply_Greset!(reg::ArrayReg, i::Int, REG_STACK)

Apply a reset gate (or SUM).
This instruct will increase the stack top of `REG_STACK` by 1.
"""
@i function apply_Greset!(reg::ArrayReg{1,T}, i::Int, REG_STACK) where T<:Tropical
    @routine @invcheckoff g ← put(nqubits(reg), i=>tropicalblock(MMatrix{2,2}(resettensor(T))))
    apply!(reg, g, REG_STACK)
    ~@routine
end

"""
    apply_Gcut!(reg::ArrayReg, i::Int, REG_STACK)

Apply a cut gate.
This instruct will increase the stack top of `REG_STACK` by 1.
"""
@i function apply_Gcut!(reg::ArrayReg{1,T}, i::Int, REG_STACK) where T<:Tropical
    @routine @invcheckoff g ← put(nqubits(reg), i=>tropicalblock(MMatrix{2,2}(one(T), one(T), one(T), one(T))))
    apply!(reg, g, REG_STACK)
    ~@routine
end
