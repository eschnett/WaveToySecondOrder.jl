@testitem "evolve3d_conservative" tags=[:cpu] begin
    _progress(m) = (printstyled(stderr, "  • ", m, "\n"; color = :cyan); flush(stderr))

# End-to-end tests of the conservative first-order 3D driver
# `evolve3d(; formulation = :conservative, …)` on uniform_hex (Milestone
# 1). Mirrors evolve2d_tests.jl.

using Test
using WaveToySecondOrder: evolve3d

@testset "evolve3d conservative driver" begin
    _progress("periodic minkowski: convergence + energy")
    @testset "periodic minkowski: plane-wave convergence + energy" begin
        errs = Float64[]
        for M in (4, 8)
            r = evolve3d(; formulation = :conservative, N = 4, M,
                         background = :minkowski, ic = :exact, bc = :periodic,
                         t1 = 0.3, Nt = 4)
            push!(errs, maximum(r.l2_err))
        end
        @test errs[1] / errs[2] > 4
        r = evolve3d(; formulation = :conservative, N = 4, M = 6,
                     background = :minkowski, ic = :exact, bc = :periodic,
                     t1 = 0.5, Nt = 5)
        @test abs(r.energy[end] / r.energy[1] - 1) < 1e-3
        @test r.integrator_name == :RK4
    end

    _progress("non-periodic :auto Sommerfeld: noise bounded")
    @testset "noise + :auto absorbs (subluminal)" begin
        r = evolve3d(; formulation = :conservative, N = 4, M = 4,
                     background = :minkowski, ic = :noise, bc = :auto,
                     ε_KO = 0.1, noise_amp = sqrt(eps(Float64)),
                     t1 = 0.5, Nt = 5)
        @test all(isfinite, r.Φ_final)
        @test maximum(r.energy) ≤ r.energy[1] * (1 + 1e-6)
    end

    _progress("bad formulation throws")
    @testset "formulation guard" begin
        @test_throws ArgumentError evolve3d(; formulation = :bogus, N = 4, M = 2)
    end
end

end
