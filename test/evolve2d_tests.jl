@testitem "evolve2d" tags=[:cpu] begin
    _progress(m) = (printstyled(stderr, "  • ", m, "\n"; color = :cyan); flush(stderr))

# End-to-end tests of the 2D driver `evolve2d` (conservative
# first-order (Φ,Π) on uniform_quad + ADM backgrounds + field-radiation
# BCs). Kernel physics is covered by test_wave2d.jl; this checks the
# driver wiring.

using Test
using WaveToySecondOrder: evolve2d
using HexMeshes: make_annulus_mesh
using HexSBPSAT: make_element, make_geometry

# Brute-force min adjacent-node spacing over the given reference axes,
# for the _min_node_spacing_2d regression test.
function _brute_spacing(c, N, Ne; axes)
    h = Inf
    for e in 1:Ne, j in 1:N, i in 1:N
        (1 in axes) && i > 1 &&
            (h = min(h, hypot(c[1,i,j,e]-c[1,i-1,j,e], c[2,i,j,e]-c[2,i-1,j,e])))
        (2 in axes) && j > 1 &&
            (h = min(h, hypot(c[1,i,j,e]-c[1,i,j-1,e], c[2,i,j,e]-c[2,i,j-1,e])))
    end
    return h
end

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

    _progress("evolve2d: constant-shift exact-solution convergence")
    @testset "periodic constant-shift: exact convergence (dispersion sign)" begin
        # Regression for the ω sign: with a nonzero shift the exact
        # solution must actually solve the PDE, so the L² error converges.
        # The previous +β·k dispersion made Φe a non-solution (O(1) error).
        errs = Float64[]
        for M in (8, 16)
            r = evolve2d(; N = 4, M, background = :constant_shift,
                         shift = (0.3, 0.2), ic = :exact, bc = :periodic,
                         t1 = 0.3, Nt = 4)
            push!(errs, maximum(r.l2_err))
        end
        @test errs[end] < errs[1]
        @test errs[1] / errs[end] > 4          # ~3rd-order over a 2× refine
        @test errs[end] < 1e-2
    end

    _progress("evolve2d: node spacing over all reference axes")
    @testset "_min_node_spacing_2d uses every axis (curved dt)" begin
        geom = make_geometry(make_annulus_mesh(Float64, 0.5, 2.0, 4),
                             make_element(Float64, 4))
        c = geom.coords; N = 4; Ne = geom.Ne
        all_axes = _brute_spacing(c, N, Ne; axes = (1, 2))
        axis1    = _brute_spacing(c, N, Ne; axes = (1,))
        @test all_axes < axis1     # angular (η) spacing is the smaller one
        @test WaveToySecondOrder._min_node_spacing_2d(c) ≈ all_axes
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

    _progress("evolve2d: cubed-square Dirichlet convergence")
    @testset "cubed-square + Dirichlet: converges to analytic" begin
        # Curved Dirichlet boundary injecting the exact plane-wave data;
        # the driver tracks the analytic solution and converges.
        errs = Float64[]
        for M in (2, 4, 8)
            r = evolve2d(; N = 4, M, mesh_kind = :cubed_square, R = 0.3,
                         background = :minkowski, ic = :exact,
                         bc = :dirichlet, t1 = 0.4, Nt = 5)
            push!(errs, maximum(r.l2_err))
        end
        @test all(isfinite, errs)
        @test (errs[1] / errs[end])^(1/2) > 2.5
        # Dirichlet requires exact data.
        @test_throws ArgumentError evolve2d(; N = 4, M = 2,
            mesh_kind = :cubed_square, bc = :dirichlet, ic = :noise,
            t1 = 0.1, Nt = 2)
    end

    _progress("evolve2d: inflated-square mesh")
    @testset "inflated-square: gaussian absorbed + Dirichlet converges" begin
        r = evolve2d(; N = 4, M = 2, mesh_kind = :inflated_square,
                     L = 0.2, R1 = 0.5, R2 = 1.0, ic = :gaussian,
                     ic_width = 0.2, ε_KO = 0.1, t1 = 0.5, Nt = 5)
        @test all(isfinite, r.Φ_final)
        @test r.mesh_kind === :inflated_square
        @test r.energy[end] ≤ r.energy[1]
        errs = Float64[]
        for M in (2, 4)
            rr = evolve2d(; N = 4, M, mesh_kind = :inflated_square,
                          background = :minkowski, ic = :exact,
                          bc = :dirichlet, t1 = 0.4, Nt = 5)
            push!(errs, maximum(rr.l2_err))
        end
        @test (errs[1] / errs[end]) > 3      # converges under refinement
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
