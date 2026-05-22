using Test
using WaveToySecondOrder

@testset "WaveToySecondOrder" begin
    include("test_operators.jl")
    include("test_kernels1d.jl")
    include("test_mesh.jl")
    include("test_kernels3d.jl")
end
