export i_magtensor!, i_gvb_tensor!, i_bondtensor!, i_g16_tensor!
export i_magtensor, i_gvb_tensor, i_bondtensor, i_g16_tensor
export apply_Ghb!, apply_Gvb!, apply_G16!, apply_Gh!, apply_Gcp!, apply_Greset!, apply_Gcut!

(f::Type{Inv{TropicalMatrixBlock}})(m::TropicalMatrixBlock) = m.mat

"""
    i_magtensor!(v, Jij)

`v` should be a length 2 vector.
"""
@i function i_magtensor!(v::AbstractVector{TT}, h::Real) where TT<:TropicalTypes
    @safe @assert size(v) == (2,)
    Tropical(h)
    v[1] *= h
    v[2] /= h
    (~Tropical)(h)
end

"""
    i_bondtensor!(mat, Jij)

`mat` should be a `one` tensor.
"""
@i function i_bondtensor!(mat::AbstractMatrix{TT}, Jij::Real) where TT<:TropicalTypes
    @safe @assert size(mat) == (2,2)
    Tropical(Jij)
    mat[1,1] *= Jij
    mat[2,2] *= Jij
    mat[1,2] /= Jij
    mat[2,1] /= Jij
    (~Tropical)(Jij)
end

@i function i_gvb_tensor!(mat::Diagonal{TT}, Jij::Real) where TT<:TropicalTypes
    @safe @assert size(mat) == (4,4)
    Tropical(Jij)
    mat[1,1] *= Jij
    mat[2,2] /= Jij
    mat[3,3] /= Jij
    mat[4,4] *= Jij
    (~Tropical)(Jij)
end

@i function i_g16_tensor!(out!::AbstractMatrix{T}, Js) where T<:TropicalTypes
    @safe @assert length(Js) == 16
    @safe @assert size(out!) == (16, 16)
    @routine begin
        y ← ones(T,2,2,2,2,2,2,2,2)
        xs ← ([ones(T,2,2) for i=1:16]...,)
        for i = 1:length(Js)
            i_bondtensor!(xs |> tget(i), Js[i])
        end
        ixs ← ([(ix...,) for ix in split("aα,aβ,aγ,aδ,bα,bβ,bγ,bδ,cα,cβ,cγ,cδ,dα,dβ,dγ,dδ", ',')]...,)
        iy ← ("abcdαβγδ"...,)
        LogLikeNumbers.i_einsum!(ixs, xs, iy, y)
    end
    for i=1:length(out!)
        out![i] *= y[i]
    end
    ~@routine
end

i_bondtensor(::Type{TT}, Jij::T) where {T<:Real, TT} = i_bondtensor!(ones(TT, 2, 2), Jij)[1]
i_magtensor(::Type{TT}, h::T) where {T<:Real, TT} = i_magtensor!(ones(TT, 2), h)[1]
i_gvb_tensor(::Type{TT}, Jij::T) where {T<:Real, TT} = i_gvb_tensor!(Diagonal(ones(TT, 4)), Jij)[1]
i_g16_tensor(::Type{TT}, Js::Vector{T}) where {T<:Real, TT} = i_g16_tensor!(ones(TT, 16, 16), Js)[1]

i_bondtensor(Jij::T) where T<:Real = i_bondtensor(Tropical{T}, Jij)
i_magtensor(h::T) where T<:Real = i_magtensor(Tropical{T}, h)
i_gvb_tensor(Jij::T) where T<:Real = i_gvb_tensor(Tropical{T}, Jij)
i_g16_tensor(Js::Vector{T}) where T<:Real = i_g16_tensor(Tropical{T}, Js)

@i function YaoBlocks.apply!(reg::ArrayReg{1}, pb::PutBlock{N,C,<:TropicalMatrixBlock}, REG_STACK) where {N, C}
    i_instruct!(vec(reg.state), pb.content.mat, pb.locs, (), (), REG_STACK)
end
@i function YaoBlocks.apply!(reg::ArrayReg{1}, cb::ControlBlock{N,<:TropicalMatrixBlock}, REG_STACK) where {N, C}
    i_instruct!(vec(reg.state), cb.content.mat, cb.locs, cb.ctrl_locs, cb.ctrl_config, REG_STACK)
end
@i function YaoBlocks.apply!(reg::ArrayReg{1,<:Tropical}, cb::ControlBlock{N,<:XGate}, REG_STACK) where {N, C}
    i_instruct!(vec(reg.state), Val(:X), cb.locs, cb.ctrl_locs, cb.ctrl_config, REG_STACK)
end
YaoBlocks.apply!(reg::ArrayReg{B}, b::TropicalMatrixBlock, REG_STACK) where B = throw(NotImplementedError(:apply!, typeof((reg, b))))

"""
    apply_Ghb!(reg::ArrayReg, i::Int, J::Real, REG_STACK)

Apply a horizontal  spinglassbond tensor (parametrized by coupling `J`) on site `i`.
This instruct will increase the stack top of `REG_STACK` by 1.
"""
@i function apply_Ghb!(reg::ArrayReg{1,T}, i::Int, J::Real, REG_STACK) where T<:Tropical
    @routine @invcheckoff begin
        nbit ← nqubits(reg)
        blk ← put(nbit, i=>tropicalblock(MMatrix{2,2}(ones(T, 2, 2))))
        i_bondtensor!(blk.content.mat, J)
    end
    apply!(reg, blk, REG_STACK)
    ~@routine
end

"""
    apply_Gh!(reg::ArrayReg, i::Int, h::Real, REG_STACK)

Apply a spinglass magnetic tensor (parametrized by coupling `h`) on site `i`.
This instruct will increase the stack top of `REG_STACK` by 0.
"""
@i function apply_Gh!(reg::ArrayReg{1,T}, i::Int, h::Real, REG_STACK) where T<:Tropical
    @routine @invcheckoff begin
        nbit ← nqubits(reg)
        blk ← put(nbit, i=>tropicalblock(Diagonal(MVector{2}(ones(T, 2)))))
        i_magtensor!(blk.content.mat.diag, h)
    end
    apply!(reg, blk, REG_STACK)
    ~@routine
end

"""
    apply_Gvb!(reg::ArrayReg, i::NTuple{2,Int}, J::Real, REG_STACK)

Apply a vertical spinglass bond tensor (parametrized by coupling `J`) on sites `i[1]` and `i[2]`.
This instruct will increase the stack top of `REG_STACK` by 0.
"""
@i function apply_Gvb!(reg::ArrayReg{1,T}, i::NTuple{2,Int}, J::Real, REG_STACK) where T<:Tropical
    @routine @invcheckoff begin
        nbit ← nqubits(reg)
        blk ← put(nbit, i=>tropicalblock(Diagonal(MVector{4}(ones(T, 4)))))
        i_gvb_tensor!(blk.content.mat, J)
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
        i_g16_tensor!(blk.content.mat, Js)
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
