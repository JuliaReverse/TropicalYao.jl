using BitBasis
using YaoArrayRegister: SDDiagonal, ArrayReg, staticize
using TupleTools
using StaticArrays
using NiLang.AD: GVar

export i_instruct!, VecStack, incstack!, stack4reg

"""
    VecStack{T}


A vector stack, its data is initialized as an empty Matrix.
It has a stack `top` that points to one column of the matrix.
The `VecStack` then behaves likes a normal vector.
One can use `incstack!` to increase the stack top.
"""
struct VecStack{T}
	data::Matrix{T}
	top::Int
end

Base.getindex(x::VecStack, i::Int) = @inbounds x.data[i,x.top]
Base.setindex!(x::VecStack, val, i::Int) = @inbounds setindex!(x.data, val, i, x.top)

"""
    stack4reg(reg, n)

Build a `VecStack` of size (`n` × the register size).
"""
function stack4reg(reg::ArrayReg{1,T}, n::Int) where T<:Tropical
	VecStack(ones(T, length(reg.state), n), 0)
end

"""
    incstack!(vs::VecStack)

Increase the stack size for 1.
"""
@i function incstack!(vs::VecStack)
	@safe @assert size(vs.data, 2) > vs.top
	INC(vs.top)
end

"""
    i_instruct!(state, U0, locs, clocs, cvals, REG_STACK)

Reversible instruction function.

* `state`is a register with tropical numbers as elements.
* `U0` is the gate matrix, also with tropical numbers as elements.
* `locs` is the locations (tuple of ints) to apply gate `U0`.
* `clocs` is the locations of control bits.
* `cvals` is the values of control bits, should be a tuple of `0` and `1`.
* `REG_STACK` is the stack for caching intermediate state.
    If U0 is dense, one-register-sized space is required,
    if `U0` is diagonal, it can be empty due to the reversibility.
    For stack constructor, see `stack4reg` for details.
"""
@i function i_instruct!(state::StridedVector{T},
        U0::AbstractMatrix{T},
        locs::NTuple{M, Int},
        clocs::NTuple{C, Int},
        cvals::NTuple{C, Int},
		REG_STACK::VecStack{T}) where {T<:Tropical,C, M}
    @routine @invcheckoff begin
        nbit ← log2i(length(state))
        U ← (all(TupleTools.diff(locs).>0) ? U0 : reorder(U0, collect(locs)|>sortperm))
        MM ← size(U0, 1)
        locked_bits ← [clocs..., locs...]
        locked_vals ← [cvals..., zeros(Int, M)...]
        locs_raw ← [i+1 for i in itercontrol(nbit, setdiff(1:nbit, locs), zeros(Int, nbit-M))] |> staticize
        configs ← itercontrol(nbit, locked_bits, locked_vals)
    end
    loop_kernel(state, configs, U, locs_raw, REG_STACK)
    ~@routine
end

@i function i_instruct!(state::StridedVector{T},
		val::Val{:X},
        locs::NTuple{M, Int},
        clocs::NTuple{C, Int},
        cvals::NTuple{C, Int},
		REG_STACK::VecStack{T}) where {T<:Tropical,C, M}
	@routine @invcheckoff begin
	    configs ← itercontrol(log2dim1(state), [clocs..., locs[1]], [cvals..., 0])
	    mask ← bmask(locs...)
	end
	@invcheckoff for i=1:length(configs)
		@invcheckoff b ← configs[i]
        @inbounds NiLang.SWAP(state[b+1], state[flip(b, mask) + 1])
		@invcheckoff b → configs[i]
    end
	~@routine
end

@i function loop_kernel(state::StridedVector{<:Tropical}, configs, U::SDDiagonal, locs_raw, REG_STACK)
    @invcheckoff @inbounds for i=1:length(configs)
        x ← configs[i]
        for i=1:length(U.diag)
            state[x+locs_raw[i]] *= identity(U.diag[i])
        end
    end
end

_scache(u::MMatrix{N,N,T}) where {N,T} = MVector{N}(ones(T, N))
_scache(u::AbstractMatrix{T}) where {T} = MVector{size(u,1)}(ones(T, size(u, 1)))

@i function loop_kernel(state::StridedVector{T}, configs, U::AbstractMatrix, locs_raw, REG_STACK) where {T<:Tropical}
    @invcheckoff begin
        branch_keeper ← zeros(Bool, size(U, 2))
		scache ← _scache(U)
		incstack!(REG_STACK)
        @inbounds for i=1:length(configs)
			@routine begin
            	x ← configs[i]
				for l=1:length(locs_raw)
					scache[l] *= identity(state[x + locs_raw[l]])
				end
			end
			for l=1:size(U,1)
				el ← zero(T)
				@routine for k=1:size(U,2)
					scache[k] *= identity(U[l,k])
					if (el.n < scache[k].n, branch_keeper[k])
						FLIP(branch_keeper[k])
						NiLang.SWAP(el, scache[k])
					end
				end
				REG_STACK[x+locs_raw[l]] *= identity(el)
				~@routine
			end
			~@routine
        end
		for i=1:length(state)
        	NiLang.SWAP(state[i], REG_STACK[i])
		end
        branch_keeper → zeros(Bool, size(U, 2))
    end
end

@i function loop_kernel(state::StridedVector{T}, configs, U::MMatrix{2,2}, locs_raw, REG_STACK) where T<:Tropical
    @invcheckoff begin
		@routine @inbounds begin
			a ← one(T)
			b ← one(T)
			c ← one(T)
			d ← one(T)
			a *= identity(U[1])
			b *= identity(U[2])
			c *= identity(U[3])
			d *= identity(U[4])
		end
		incstack!(REG_STACK)
        @inbounds for i=1:length(configs)
        	x ← configs[i]
			is1 ← x + locs_raw[1]
			is2 ← x + locs_raw[2]
			s1 ← one(T)
			s2 ← one(T)
			NiLang.SWAP(s1, state[is1])
			NiLang.SWAP(s2, state[is2])
			@routine begin
				s3 ← one(T)
				s4 ← one(T)
				s3 *= s1 * b
				s4 *= s2 * d
				s1 *= identity(a)
				s2 *= identity(c)
				bk1 ← s1 > s2
				bk2 ← s3 > s4
			end
			if (bk1, ~)
				state[is1] *= identity(s1)
			else
				state[is1] *= identity(s2)
			end
			if (bk2, ~)
				state[is2] *= identity(s3)
			else
				state[is2] *= identity(s4)
			end
			~@routine
        	NiLang.SWAP(s1, REG_STACK[is1])
        	NiLang.SWAP(s2, REG_STACK[is2])
    	end
		~@routine
	end
end

NiLang.AD.GVar(x::ArrayReg{B}) where B = ArrayReg{B}(GVar(x.state))
NiLang.AD.GVar(x::VecStack) where B = VecStack(GVar(x.data), x.top)
