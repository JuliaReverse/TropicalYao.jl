module TropicalYao

using Yao
using NiLang
using NiLog
using LinearAlgebra: Diagonal

include("instructs.jl")
include("tropicalgate.jl")
include("blocks.jl")

end # module
