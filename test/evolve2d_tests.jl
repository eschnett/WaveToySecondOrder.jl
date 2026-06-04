@testitem "evolve2d" tags=[:cpu] begin
    _progress(m) = (printstyled(stderr, "  • ", m, "\n"; color = :cyan); flush(stderr))

# End-to-end tests of the 2D driver `evolve2d` (conservative
# first-order (Φ,Π) on uniform_quad + ADM backgrounds + field-radiation
# BCs). Kernel physics is covered by test_wave2d.jl; this checks the
# driver wiring.

using Test
using WaveToySecondOrder: evolve2d


@testset "evolve2d driver" begin
    _progress("evolve2d: periodic minkowski convergence")
    @testset "periodic minkowski: plane-wave convergence + energy" begin
        errs = Float64[]
        for M in (4, 8, 16)
            r = evolve2d(; N = 4, M, background = :minkowski,
                         bc = :periodic, t1 = 0.5, Nt = 5)
            push!(errs, maximum(r.l2_err))
        end
        @test (errs[1] / errs[end])^(1/2) > 2.5
        r = evolve2d(; N = 4, M = 12, background = :minkowski,
                     bc = :periodic, t1 = 1.0, Nt = 5)
        @test abs(r.energy[end] / r.energy[1] - 1) < 1e-3
        @test r.integrator_name == :RK4
    end

    _progress("evolve2d: gauge wave (periodic)")
    @testset "gauge-wave periodic bounded error" begin
        r = evolve2d(; N = 4, M = 16, background = :gaugewave, A = 0.1,
                     bc = :periodic, t1 = 0.5, Nt = 5)
        @test all(isfinite, r.Φs)
        @test maximum(r.l2_err) < 1e-2
    end

    _progress("evolve2d: noise + auto Sommerfeld (absorbing)")
    @testset "noise + :auto absorbs energy (subluminal)" begin
        r = evolve2d(; N = 4, M = 12, background = :minkowski, bc = :auto,
                     ic = :noise, noise_amp = 1.0, ε_KO = 0.1,
                     t1 = 2.0, Nt = 8)
        @test all(isfinite, r.Φ_final)
        @test r.energy[end] < r.energy[1]
    end

    _progress("evolve2d: superluminal :auto (excision + full Dirichlet)")
    @testset "superluminal shift :auto stays bounded" begin
        # shift=(2,0): −x outflow (excision), +x inflow (full
        # Dirichlet, exact data), ±y subluminal (Sommerfeld).
        r = evolve2d(; N = 4, M = 12, background = :constant_shift,
                     shift = (2.0, 0.0), bc = :auto, ic = :noise,
                     noise_amp = sqrt(eps(Float64)), ε_KO = 0.1,
                     t1 = 1.0, Nt = 5)
        @test all(isfinite, r.Φ_final)
        @test maximum(abs, r.Φ_final) < 100 * sqrt(eps(Float64))
    end

    _progress("evolve2d: curvilinear cubed-square")
    @testset "cubed-square gaussian pulse: bounded, absorbed" begin
        r = evolve2d(; N = 4, M = 4, mesh_kind = :cubed_square, R = 0.3,
                     ic = :gaussian, ic_width = 0.15, ε_KO = 0.1,
                     t1 = 1.0, Nt = 10)
        @test all(isfinite, r.Φ_final)
        @test r.mesh_kind === :cubed_square
        # Energy non-increasing (Sommerfeld outer boundary absorbs) and
        # the pulse has partly radiated out.
        @test r.energy[end] ≤ r.energy[1] * (1 + 1e-6)
        @test r.energy[end] < r.energy[1]
    end

    _progress("evolve2d: inadmissible bc throws")
    @testset "inadmissible bc combinations throw" begin
        # Sommerfeld requested at the +x superluminal inflow face.
        @test_throws ArgumentError evolve2d(; N = 4, M = 4,
            background = :constant_shift, shift = (2.0, 0.0),
            bc = (:sommerfeld, :sommerfeld, :sommerfeld, :sommerfeld),
            t1 = 0.1, Nt = 2)
        # Excision at a subluminal face.
        @test_throws ArgumentError evolve2d(; N = 4, M = 4,
            background = :minkowski,
            bc = (:excision, :excision, :excision, :excision),
            t1 = 0.1, Nt = 2)
        # Unknown bc spec.
        @test_throws ArgumentError evolve2d(; N = 4, M = 4,
            background = :minkowski, bc = :nonsense, t1 = 0.1, Nt = 2)
    end
end

end
