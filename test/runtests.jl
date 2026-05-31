# On Apple Silicon (developer machines and `macos-latest` ARM CI
# runners), add Metal to the test sandbox so the GPU testset in
# `test_kernels3d.jl` actually executes. Metal.jl only installs cleanly
# on `apple/aarch64`; on every other platform we leave the sandbox
# alone and the `HAS_METAL` gate inside `test_kernels3d.jl` causes the
# GPU testset to skip silently. This is the cheapest way to get CI to
# exercise the Metal path without forcing Metal into the cross-platform
# test deps (which would break Linux/x86 CI).
if Sys.isapple() && Sys.ARCH === :aarch64
    using Pkg
    Pkg.add("Metal")
end

using Test
using WaveToySecondOrder

# Print a short progress banner before each test file / nested testset.
# Tests are compile-dominated; the banner tells the user what they're
# waiting on instead of staring at a silent terminal.
function _section(label)
    printstyled(stderr, "── ", label, " ──\n"; color = :cyan, bold = true)
    flush(stderr)
end

# Single shared `_progress` helper used by all `test_*.jl` files. Defined
# here so the per-file `include`s don't redefine the same method in the
# top-level scope (which used to trigger `Method definition … overwritten`
# warnings when `Pkg.test()` ran every file in the same process).
_progress(msg) = (printstyled(stderr, "  • ", msg, "\n"; color = :cyan);
                  flush(stderr))

@testset verbose = true "WaveToySecondOrder" begin
    _section("test_kernels1d.jl");   include("test_kernels1d.jl")
    _section("test_kernels3d.jl");   include("test_kernels3d.jl")
    _section("test_sommerfeld.jl");  include("test_sommerfeld.jl")
    _section("test_wave_lap_strong.jl"); include("test_wave_lap_strong.jl")
    _section("test_wave_lap_strong_conservative.jl"); include("test_wave_lap_strong_conservative.jl")
    _section("test_wave_lap_strong_mesh.jl"); include("test_wave_lap_strong_mesh.jl")
    _section("test_wave_curved.jl"); include("test_wave_curved.jl")
    # The mesh-topology tests live in `HexMeshes/test/test_mesh.jl`.
    # The operator-level tests (SBP identities, MeshGeometry shape,
    # apply_laplacian! symmetry / spectrum, to_device round-trip)
    # live in `HexSBPSAT/test/`. Run them with
    #   cd HexMeshes  && julia --project=. -e 'using Pkg; Pkg.test()'
    #   cd HexSBPSAT  && julia --project=. -e 'using Pkg; Pkg.test()'
end
