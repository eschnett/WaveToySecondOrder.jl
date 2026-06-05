@testitem "evolve3d_curvilinear" tags=[:cpu] begin
    _progress(m) = (printstyled(stderr, "  • ", m, "\n"; color = :cyan); flush(stderr))

# End-to-end tests of the conservative first-order 3D driver on
# CURVILINEAR meshes (Milestone 2): curved-Dirichlet plane-wave
# convergence on the cubed-cube, an absorbing Sommerfeld outer sphere,
# and the BH-excision setup on the radial shell (inner sphere excised
# with a superluminal radial shift, outer sphere Sommerfeld). The 3D
# analog of annulus_tests.jl / evolve2d curvilinear coverage.

using Test
using WaveToySecondOrder: evolve3d

@testset "evolve3d conservative driver (curvilinear)" begin
    _progress("cubed_cube curved Dirichlet: plane-wave convergence + energy")
    @testset "cubed_cube Dirichlet: convergence + energy" begin
        errs = Float64[]
        for M in (2, 4)
            r = evolve3d(; formulation = :conservative, N = 4, M,
                         mesh_kind = :cubed_cube, R = 0.3,
                         background = :minkowski, ic = :exact, bc = :dirichlet,
                         t0 = 0, t1 = 0.2, Nt = 4, ε_KO = 0.0)
            push!(errs, maximum(r.l2_err))
            @test all(isfinite, r.Φ_final)
        end
        @test errs[1] / errs[2] > 4          # ~3rd-order over a 2× refinement
        # Energy stays bounded under the radiative (Dirichlet) boundary.
        r = evolve3d(; formulation = :conservative, N = 4, M = 4,
                     mesh_kind = :cubed_cube, R = 0.3,
                     background = :minkowski, ic = :exact, bc = :dirichlet,
                     t0 = 0, t1 = 0.3, Nt = 4, ε_KO = 0.0)
        @test r.energy[end] ≤ r.energy[1] * 1.1
    end

    _progress("cubed_cube curved Sommerfeld: noise bounded")
    @testset "cubed_cube Sommerfeld: noise absorbed" begin
        r = evolve3d(; formulation = :conservative, N = 4, M = 1,
                     mesh_kind = :cubed_cube, R = 0.3,
                     background = :minkowski, ic = :noise, bc = :sommerfeld,
                     ε_KO = 0.1, noise_amp = sqrt(eps(Float64)),
                     t0 = 0, t1 = 0.5, Nt = 5)
        @test all(isfinite, r.Φ_final)
        @test maximum(r.energy) ≤ r.energy[1] * (1 + 1e-6)
    end

    _progress("radial_shell BH-excision: superluminal-inflow shift drains noise")
    @testset "radial_shell excision: inner outflow + outer Sommerfeld" begin
        # Radial shift superluminal at the inner sphere (excised, tag 8)
        # and subluminal at the outer sphere (Sommerfeld). Noise drains
        # out through both boundaries; energy is non-increasing.
        r = evolve3d(; formulation = :conservative, N = 4, M = 2,
                     mesh_kind = :radial_shell, R1 = 0.5, R2 = 1.0,
                     background = :radial_shift, shift = (2.0, 0.0, 0.0),
                     ic = :noise, bc = :sommerfeld, ε_KO = 0.1,
                     noise_amp = sqrt(eps(Float64)),
                     t0 = 0, t1 = 1.0, Nt = 5)
        @test all(isfinite, r.Φ_final)
        @test maximum(r.energy) ≤ r.energy[1] * (1 + 1e-6)
        @test r.energy[end] < r.energy[1]     # noise actually leaves
    end
end

end
