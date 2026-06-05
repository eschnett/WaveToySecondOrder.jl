@testitem "evolve3d_gpu" tags=[:gpu] begin
    _progress(m) = (printstyled(stderr, "  • ", m, "\n"; color = :cyan); flush(stderr))

# The conservative 3D driver on GPU (Metal): periodic and non-periodic
# (:auto) evolutions run entirely on-device through the same driver the
# CPU uses. Deterministic IC + fixed dt ⇒ GPU and CPU agree to Float32
# reduction order. Auto-skips without Metal.

using KernelAbstractions
using Test
using WaveToySecondOrder: evolve3d

if !@isdefined(HAS_METAL)
    const HAS_METAL = try
        @eval using Metal
        Metal.functional()
    catch
        false
    end
end

if HAS_METAL
    @testset "evolve3d conservative on Metal vs CPU (Float32)" begin
        T = Float32
        function _agree(label; kw...)
            _progress(label)
            rc = evolve3d(; formulation = :conservative, T,
                          backend = KernelAbstractions.CPU(), kw...)
            rg = evolve3d(; formulation = :conservative, T,
                          backend = MetalBackend(), kw...)
            @test all(isfinite, rg.Φ_final)
            rel = maximum(abs, rg.Φ_final .- rc.Φ_final) /
                  max(maximum(abs, rc.Φ_final), eps(T))
            @test rel ≤ 1e-3
        end
        _agree("periodic minkowski"; mesh_kind = :cubical, M = 4, N = 4,
               background = :minkowski, ic = :exact, bc = :periodic,
               t1 = 0.2, Nt = 4, cfl = 0.1)
        _agree("non-periodic :auto Sommerfeld (gaussian)";
               mesh_kind = :cubical, M = 4, N = 4, background = :minkowski,
               ic = :gaussian, bc = :auto, ε_KO = 0.1, t1 = 0.2, Nt = 4, cfl = 0.1)
        _agree("curvilinear cubed_cube Sommerfeld (gaussian)";
               mesh_kind = :cubed_cube, M = 1, N = 4, R = 0.3,
               background = :minkowski, ic = :gaussian, bc = :sommerfeld,
               ε_KO = 0.1, t1 = 0.2, Nt = 4, cfl = 0.1)
        # Deterministic IC (Gaussian, not noise) so the two runs share
        # the same initial data; exercises the excision-tag skip + radial
        # shift on Metal.
        _agree("curvilinear radial_shell excision (gaussian)";
               mesh_kind = :radial_shell, M = 2, N = 4, R1 = 0.5, R2 = 1.0,
               background = :radial_shift, shift = (2.0f0, 0.0f0, 0.0f0),
               ic = :gaussian, ic_width = 0.5, bc = :sommerfeld, ε_KO = 0.1,
               t1 = 0.3, Nt = 4, cfl = 0.1)
    end
end

end
