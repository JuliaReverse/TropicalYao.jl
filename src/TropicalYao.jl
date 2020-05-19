module TropicalYao

using Yao
using NiLang
using NiLogLikeNumbers
using LinearAlgebra: Diagonal
using LuxurySparse

export Tropical

include("instructs.jl")
include("tropicalgate.jl")
include("spinglass_gates.jl")

end # module