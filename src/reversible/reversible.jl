module Reversible

using Yao
using NiLang
using TropicalNumbers
using StaticArrays
using LinearAlgebra: Diagonal
using LuxurySparse
using Suppressor
using ..TropicalYao: TropicalMatrixBlock, TropicalTypes, tropicalblock, copytensor, resettensor

export Tropical, CountingTropical, TropicalTypes
export LogLikeNumbers

include("LogLikeNumbers/LogLikeNumbers.jl")
using .LogLikeNumbers

include("instructs.jl")
include("spinglass_gates.jl")

end
