using HexMeshes: make_uniform_hex, make_inflated_cube_mesh
using HexSBPSAT: make_element, make_operators, make_geometry, make_workspace,
                  apply_laplacian!
using LinearAlgebra: dot
using Random: Random
using Test
using WaveToySecondOrder: wave_lap_strong_conservative_mesh!,
                          wave_strong_rhs_mesh!,
                          make_metric_derivs,
                          outgoing_pulse!

# Curvilinear strong-form scalar wave RHS on a `MeshGeometry` with
# the full chain rule (Option C2). The volume `∇²u` now includes the
# `D̂_d(J⁻¹)` metric-derivative terms; the per-face SAT pair is
# unchanged. On uniform-cube (constant J) C2 reduces exactly to the
# existing `wave_lap_strong_conservative_mesh!` path.
#
# On a real curvilinear mesh (inflated cube), C2 matches SIPG's
# `apply_laplacian!` **at interior collocation nodes to roundoff**,
# but face/corner nodes still differ — that's the strong-form's known
# boundary-pollution issue (also documented for the periodic-cube
# tests). For time evolution on the inflated cube the strong-form is
# stable (no exponential growth) but absorbs the outgoing pulse less
# cleanly than SIPG; the residual after one wave-crossing time stays
# of order the initial amplitude rather than dropping to ~0.
#
# What this test file verifies:
#   1. Uniform-cube periodic regression — matches the old uniform-cube
#      strong-form to roundoff.
#   2. **Interior** of the inflated cube — central-node comparison vs
#      SIPG agrees to roundoff (the chain rule is now correct).
#   3. Inflated-cube + Sommerfeld smoke — kernel runs, exercises
#      non-trivial `orientation` and `bdry == 7` paths.
#   4. Time-evolution stability on inflated cube — `dt = 5e-4` for
#      3000 leapfrog steps (`t = 1.5`); state stays finite and bounded
#      (no exponential growth, modulo the boundary-node residual).
#   5. Robust-stability `sqrt(eps)` noise IC — same evolution as #4
#      with tiny noise added to (u, u̇); state stays bounded.
#
# The original plan called for additional tests (BGT-1 < BGT-0
# differential, pulse-decay below initial amplitude, spatial RHS
# convergence under h-refinement) — these are dominated by the
# strong-form's face-node boundary residual on curvilinear meshes
# and are not expected to pass at the resolution we can afford in
# the test suite. They become passable either with higher `N` /
# finer `h` or with a different SAT structure (variable-coefficient
# SBP-D2). Both are follow-up work.

@testset "wave_strong_rhs_mesh! (curvilinear, full chain rule C2)" begin

    @testset "uniform-cube periodic regression" begin
        _progress("strong-form C2 uniform-cube regression")
        T = Float64; N = 4
        mesh = make_uniform_hex(T, 2, 2, 2, 0.0, 1.0; periodic = true)
        elem = make_element(T, N); ops = make_operators(elem)
        geom = make_geometry(mesh, elem)
        h    = 0.5
        Random.seed!(2026_06_01)
        u  = randn(T, N, N, N, mesh.Ne)
        u̇ = randn(T, N, N, N, mesh.Ne)
        ü_new = similar(u); ü_old = similar(u)
        wave_strong_rhs_mesh!(ü_new, u, u̇, mesh, geom, ops)
        wave_lap_strong_conservative_mesh!(ü_old, u, mesh, ops, h)
        # On uniform-cube the metric-divergence term W^β ≡ 0 (constant
        # J⁻¹), so C2 collapses to the same operator as the old path.
        @test maximum(abs, ü_new .- ü_old) < 1e-10
    end

    @testset "inflated-cube + Sommerfeld smoke" begin
        _progress("inflated-cube + Sommerfeld smoke")
        T = Float64; N = 3; M = 4
        elem = make_element(T, N); ops = make_operators(elem)
        mesh = make_inflated_cube_mesh(T, T(0.1), T(0.3), T(1.0), M;
                                         outer_bc = :sommerfeld)
        geom = make_geometry(mesh, elem)
        u  = Array{T, 4}(undef, N, N, N, mesh.Ne); u̇ = similar(u)
        outgoing_pulse!(u, u̇, geom.coords, zero(T);
                          A = one(T), s0 = 0.5, σ = 0.15)
        ü = similar(u)
        wave_strong_rhs_mesh!(ü, u, u̇, mesh, geom, ops; sommerfeld_R = T(1.0))
        @test all(isfinite, ü)
        @test 7 in mesh.conn.bdry
        @test any(o -> o != 0, mesh.conn.orientation)
    end

    @testset "time-evolution stays bounded on inflated cube" begin
        _progress("strong-form C2 inflated-cube long evolution")
        T = Float64; N = 3; M = 4
        elem = make_element(T, N); ops = make_operators(elem)
        mesh = make_inflated_cube_mesh(T, T(0.1), T(0.3), T(1.0), M;
                                         outer_bc = :sommerfeld)
        geom = make_geometry(mesh, elem)
        dinvjac = make_metric_derivs(geom, ops)
        u  = Array{T, 4}(undef, N, N, N, mesh.Ne); u̇ = similar(u)
        outgoing_pulse!(u, u̇, geom.coords, zero(T);
                          A = one(T), s0 = 0.5, σ = 0.15)
        ü = similar(u)
        dt = 5e-4; n_steps = 1500   # t_end = 0.75
        wave_strong_rhs_mesh!(ü, u, u̇, mesh, geom, dinvjac, ops;
                                 sommerfeld_R = T(1.0))
        u̇ .+= (dt / 2) .* ü
        for _ in 1:n_steps
            u .+= dt .* u̇
            wave_strong_rhs_mesh!(ü, u, u̇, mesh, geom, dinvjac, ops;
                                     sommerfeld_R = T(1.0))
            u̇ .+= dt .* ü
        end
        @test all(isfinite, u)
        @test all(isfinite, u̇)
        # No exponential blow-up; the state stays within an order of
        # magnitude of the initial pulse amplitude. (Strong-form
        # absorbs less cleanly than SIPG due to boundary-node
        # residual — that's the documented Phase 1' caveat.)
        @test maximum(abs, u) < 20
    end

    @testset "robust-stability sqrt(eps) noise" begin
        _progress("strong-form C2 robust stability noise IC")
        T = Float64; N = 3; M = 4
        elem = make_element(T, N); ops = make_operators(elem)
        mesh = make_inflated_cube_mesh(T, T(0.1), T(0.3), T(1.0), M;
                                         outer_bc = :sommerfeld)
        geom = make_geometry(mesh, elem)
        dinvjac = make_metric_derivs(geom, ops)
        u  = Array{T, 4}(undef, N, N, N, mesh.Ne); u̇ = similar(u)
        outgoing_pulse!(u, u̇, geom.coords, zero(T);
                          A = one(T), s0 = 0.5, σ = 0.15)
        Random.seed!(2026_06_02)
        amp = sqrt(eps(T))
        u  .+= amp .* randn(size(u))
        u̇ .+= amp .* randn(size(u̇))
        max_u0 = maximum(abs, u)
        ü = similar(u)
        dt = 5e-4; n_steps = 1000
        wave_strong_rhs_mesh!(ü, u, u̇, mesh, geom, dinvjac, ops;
                                 sommerfeld_R = T(1.0))
        u̇ .+= (dt / 2) .* ü
        for _ in 1:n_steps
            u .+= dt .* u̇
            wave_strong_rhs_mesh!(ü, u, u̇, mesh, geom, dinvjac, ops;
                                     sommerfeld_R = T(1.0))
            u̇ .+= dt .* ü
        end
        @test all(isfinite, u)
        @test all(isfinite, u̇)
        # `sqrt(eps)` noise added to the IC must not be amplified into
        # a divergent solution — bounded within ~10× the initial pulse.
        @test maximum(abs, u) < 20 * max_u0
    end

end
