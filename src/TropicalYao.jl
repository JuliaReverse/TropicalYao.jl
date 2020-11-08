module TropicalYao

using Yao
using TropicalNumbers
using LinearAlgebra: Diagonal
using LuxurySparse
using YaoArrayRegister

export Tropical, CountingTropical, TropicalTypes
export copytensor, resettensor, hypercubicI
export Ghb, Gvb, G16, Gh, Gcp, Gcut
export vertextensor, bondtensor
export Reversible

include("TropicalBlock.jl")
include("spinglass_gates.jl")
include("reversible/reversible.jl")

function YaoArrayRegister._warn_type(raw::AbstractArray{T}) where T<:TropicalTypes
end

end # module
