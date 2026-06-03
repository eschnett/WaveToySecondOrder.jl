# End-to-end tests of the DiffEq-based 1D driver `evolve1d` (mesh +
# geometry + Background1D + ODEProblem/ArrayPartition + explicit RK).
# The kernel-level physics is covered by `test_wave1d.jl`; this file
# checks the driver wiring: exact-IC accuracy, energy monitoring,
# noise mode, and the background menu.

using Test
using WaveToySecondOrder: evolve1d

@isdefined(_progress) ||
    (_progress(msg) = (printstyled(stderr, "  • ", msg, "\n"; color = :cyan);
                       flush(stderr)))

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
                       ic = :noise, ε_KO = 1e-4, t1 = 2.0, Nt = 6)
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
end
