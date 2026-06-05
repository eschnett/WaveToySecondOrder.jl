@testitem "evolve2d_gpu" tags=[:gpu] begin
    _progress(m) = (printstyled(stderr, "  • ", m, "\n"; color = :cyan); flush(stderr))

# The `evolve2d` driver on GPU (Metal): periodic AND non-periodic /
# curvilinear evolutions run entirely on-device through the same driver
# the CPU uses (OrdinaryDiffEq RK4 on device ArrayPartition; all boundary
# passes are GPU kernels). Deterministic IC + fixed dt ⇒ GPU and CPU take
# identical steps, so the final fields agree to Float32 reduction order.
# Auto-skips without Metal.

using KernelAbstractions
using Test
using WaveToySecondOrder: evolve2d

if !@isdefined(HAS_METAL)
    const HAS_METAL = try
        @eval using Metal
        Metal.functional()
    catch
        false
    end
end

if HAS_METAL
    @testset "evolve2d on Metal vs CPU (Float32)" begin
        T = Float32
        # GPU-vs-CPU final-field agreement for one driver configuration.
        function _agree(label; kw...)
            _progress(label)
            rc = evolve2d(; T, backend = KernelAbstractions.CPU(), kw...)
            rg = evolve2d(; T, backend = MetalBackend(), kw...)
            @test all(isfinite, rg.Φ_final)
            rel = maximum(abs, rg.Φ_final .- rc.Φ_final) /
                  max(maximum(abs, rc.Φ_final), eps(T))
            @test rel ≤ 1e-3
        end

        # Periodic affine — the integrator-on-device baseline.
        _agree("periodic cubical (minkowski)";
               mesh_kind = :cubical, M = 8, N = 4, background = :minkowski,
               ic = :exact, bc = :periodic, t1 = 0.2, Nt = 4, cfl = 0.1)
        # Rectangular non-periodic with :auto classification — exercises
        # the host-side face classification on GPU (no device scalar index).
        _agree("cubical bc=:auto (minkowski, host classification)";
               mesh_kind = :cubical, M = 6, N = 4, background = :minkowski,
               ic = :exact, bc = :auto, t1 = 0.2, Nt = 4, cfl = 0.1)
        # Curvilinear non-periodic, absorbing outer (the core capability).
        _agree("cubed-square Sommerfeld (gaussian)";
               mesh_kind = :cubed_square, M = 4, N = 4, R = 0.3,
               background = :minkowski, ic = :gaussian, ic_width = 0.15,
               bc = :sommerfeld, ε_KO = 0.1, t1 = 0.3, Nt = 4, cfl = 0.1)
        # Annulus with inner excision + outer Sommerfeld, radial shift.
        _agree("annulus radial-shift (inner excision)";
               mesh_kind = :annulus, R1 = 0.5, R2 = 2.0, M = 4, N = 4,
               background = :radial_shift, A = 1.2, ic = :gaussian,
               ic_width = 0.3, ε_KO = 0.1, t1 = 0.3, Nt = 4, cfl = 0.1)
        # Curved Dirichlet — exercises the on-device boundary-data buffers
        # and the per-stage analytic-closure fill.
        _agree("cubed-square curved Dirichlet (exact)";
               mesh_kind = :cubed_square, M = 4, N = 4, R = 0.3,
               background = :minkowski, ic = :exact, bc = :dirichlet,
               ε_KO = 0.0, t1 = 0.3, Nt = 4, cfl = 0.1)
    end
end

end
