@testitem "evolve1d" tags=[:cpu] begin
    _progress(m) = (printstyled(stderr, "  • ", m, "\n"; color = :cyan); flush(stderr))

# End-to-end tests of the DiffEq-based 1D driver `evolve1d` (mesh +
# geometry + Background1D + ODEProblem/ArrayPartition + explicit RK).
# The kernel-level physics is covered by `test_wave1d.jl`; this file
# checks the driver wiring: exact-IC accuracy, energy monitoring,
# noise mode, and the background menu.

using Test
using WaveToySecondOrder: evolve1d


@testset "evolve1d driver" begin
    _progress("evolve1d: sine-shift exact IC, one period")
    @testset "sine-shift: exact IC, error + energy" begin
        res = evolve1d(; N = 4, M = 32, background = :sineshift,
                       A = 0.3, d = 1.0, t1 = 1.0, Nt = 21)
        @test all(isfinite, res.Φs) && all(isfinite, res.Πs)
        # L² error stays small over one period at this resolution.
        @test maximum(res.l2_err) < 1e-2
        # Energy returns to its initial value (time-periodic background).
        @test abs(res.energy[end] / res.energy[1] - 1) < 1e-3
        @test res.integrator_name == :RK4
        @test length(res.xs_line) == 4 * 32
        @test issorted(res.xs_line)
    end

    _progress("evolve1d: remaining backgrounds, smoke")
    @testset "backgrounds: minkowski / constant_shift / gaugewave" begin
        for bgkind in (:minkowski, :constant_shift, :gaugewave)
            res = evolve1d(; N = 4, M = 16, background = bgkind,
                           t1 = 0.5, Nt = 6)
            @test all(isfinite, res.Φs)
            @test maximum(res.l2_err) < 5e-2
        end
    end

    _progress("evolve1d: noise IC + KO")
    @testset "noise IC, ε_KO > 0: bounded" begin
        res = evolve1d(; N = 4, M = 16, background = :constant_shift,
                       shift = 2.0,        # superluminal
                       ic = :noise, ε_KO = 0.1, t1 = 2.0, Nt = 6)
        @test all(isfinite, res.Φs) && all(isfinite, res.Πs)
        amp = sqrt(eps(Float64))
        @test maximum(abs, res.Φ_final) < 100 * amp
    end

    _progress("evolve1d: higher order picks Tsit5")
    @testset "N = 6 picks Tsit5" begin
        res = evolve1d(; N = 6, M = 8, background = :minkowski,
                       t1 = 0.25, Nt = 3)
        @test res.integrator_name == :Tsit5
        @test maximum(res.l2_err) < 1e-3
    end

    _progress("evolve1d: boundary conditions")
    @testset "bc = :auto, subluminal Dirichlet (β = 0, converges)" begin
        # The field-radiation BC is exact at β = 0, so Dirichlet data
        # injection on Minkowski converges spectrally.
        errs = Float64[]
        for M in (8, 16, 32)
            res = evolve1d(; N = 4, M, background = :minkowski,
                           bc = :auto, t1 = 1.0, Nt = 5)
            push!(errs, maximum(res.l2_err))
        end
        @test (errs[1] / errs[end])^(1/2) > 2.5
    end

    @testset "bc = :auto, superluminal (excision + full Dirichlet)" begin
        errs = Float64[]
        for M in (8, 16, 32)
            res = evolve1d(; N = 4, M, background = :constant_shift,
                           shift = 2.0, bc = :auto, t1 = 1.0, Nt = 5)
            push!(errs, maximum(res.l2_err))
        end
        @test (errs[1] / errs[end])^(1/2) > 1.8   # state pin: ~2nd order
    end

    @testset "explicit bc tuple on a curved background (small shift)" begin
        # sineshift A = 0.05 ⇒ |β| ≲ 0.05, within the small-shift
        # policy for radiative BCs; the field-radiation BC keeps the
        # error bounded at the O(β) reflection level.
        res = evolve1d(; N = 4, M = 16, background = :sineshift, A = 0.05,
                       bc = (left = :dirichlet, right = :sommerfeld),
                       t1 = 1.0, Nt = 5)
        @test all(isfinite, res.Φ_final)
        @test maximum(res.l2_err) < 0.1
    end

    @testset "noise + Sommerfeld: energy absorbed, bounded" begin
        res = evolve1d(; N = 4, M = 16, background = :minkowski,
                       bc = (left = :sommerfeld, right = :sommerfeld),
                       ic = :noise, ε_KO = 0.1, t1 = 3.0, Nt = 5)
        @test all(isfinite, res.Φ_final)
        @test res.energy[end] < res.energy[1]
    end

    @testset "inadmissible bc combinations throw" begin
        # Sommerfeld at superluminal faces.
        @test_throws ArgumentError evolve1d(; N = 4, M = 8,
            background = :constant_shift, shift = 2.0,
            bc = (left = :sommerfeld, right = :sommerfeld), t1 = 0.1, Nt = 2)
        # Plain Dirichlet at a superluminal inflow face.
        @test_throws ArgumentError evolve1d(; N = 4, M = 8,
            background = :constant_shift, shift = 2.0,
            bc = (left = :excision, right = :dirichlet), t1 = 0.1, Nt = 2)
        # Excision at a subluminal face.
        @test_throws ArgumentError evolve1d(; N = 4, M = 8,
            background = :constant_shift, shift = 0.5,
            bc = (left = :excision, right = :sommerfeld), t1 = 0.1, Nt = 2)
        # Unknown bc spec.
        @test_throws ArgumentError evolve1d(; N = 4, M = 8,
            background = :minkowski, bc = :nonsense, t1 = 0.1, Nt = 2)
    end
end

end
