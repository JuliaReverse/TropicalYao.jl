module TropicalYao

using Yao
using NiLang
using NiLogLikeNumbers
using TropicalNumbers
using LinearAlgebra: Diagonal
using LuxurySparse

export Tropical, CountingTropical, TropicalTypes

include("counting_tropical.jl")
include("instructs.jl")
include("tropicalgate.jl")
include("spinglass_gates.jl")

end # module
