module LogLikeNumbers
using TropicalNumbers
using LogarithmicNumbers
using NiLang, NiLang.AD

import NiLang: i_sum
using ..Reversible: TropicalTypes

export Tropical

include("base.jl")
include("instructs.jl")
include("blas.jl")
include("einsum.jl")

end # module
