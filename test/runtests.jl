using Test
using WaveToySecondOrder

# Print a short progress banner before each test file / nested testset.
# Tests are compile-dominated; the banner tells the user what they're
# waiting on instead of staring at a silent terminal.
function _section(label)
    printstyled(stderr, "── ", label, " ──\n"; color = :cyan, bold = true)
    flush(stderr)
end

@testset verbose = true "WaveToySecondOrder" begin
    _section("test_operators.jl");   include("test_operators.jl")
    _section("test_kernels1d.jl");   include("test_kernels1d.jl")
    _section("test_mesh.jl");        include("test_mesh.jl")
    _section("test_kernels3d.jl");   include("test_kernels3d.jl")
end
