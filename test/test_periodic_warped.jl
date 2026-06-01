using HexMeshes: make_uniform_hex, make_warped_uniform_hex
using HexSBPSAT: make_element, make_operators, make_geometry
using LinearAlgebra: dot, eigvals
using Random: Random
using Test
using WaveToySecondOrder: wave_strong_rhs_mesh!,
                          wave_lap_strong_conservative_mesh!,
                          make_metric_derivs

# Periodic warped-cube diagnostic — isolates the curvilinear strong-form
# kernel's behaviour on a mesh that has (a) periodic topology so no
# outer BCs muddy the analysis, and (b) non-trivial Jacobians at every
# node so every inter-element face is curvilinear-vs-curvilinear.
#
# The diagnostic answers the question raised in the inflated-cube
# tests: is the residual we saw there from curvilinear inter-element
# SAT, or from outer Sommerfeld? If `wave_strong_rhs_mesh!` converges
# spectrally under h-refinement on the warped periodic mesh, then
# the inter-element SAT is consistent in the curvilinear limit — the
# inflated-cube issue is Sommerfeld-specific.

@testset "Periodic warped-cube diagnostic" begin

    @testset "A = 0 reduces to uniform-cube path" begin
        _progress("warped: A = 0 regression")
        T = Float64; N = 4
        m0   = make_uniform_hex(T, 3, 3, 3, 0.0, 1.0; periodic = true)
        m1   = make_warped_uniform_hex(T, 3, 3, 3, 0.0, 1.0, 0.0;
                                          periodic = true)
        elem = make_element(T, N); ops = make_operators(elem)
        g0   = make_geometry(m0, elem); g1 = make_geometry(m1, elem)
        Random.seed!(2026_06_03)
        u  = randn(T, N, N, N, m0.Ne)
        u̇ = randn(T, N, N, N, m0.Ne)
        ü0 = similar(u); ü1 = similar(u)
        wave_strong_rhs_mesh!(ü0, u, u̇, m0, g0, ops)
        wave_strong_rhs_mesh!(ü1, u, u̇, m1, g1, ops)
        @test maximum(abs, ü0 .- ü1) < 1e-10
    end

    # Rayleigh-quotient helper. On a warped mesh with M³ elements, the
    # sin eigenmode `u = sin(2π x) sin(2π y) sin(2π z)` is periodic in
    # the *physical* coords; we evaluate the discrete `−⟨u, H · ü⟩ /
    # ⟨u, H · u⟩` against the analytic eigenvalue `3 (2π)²`.
    function _rayleigh(A_amp, M, N)
        T    = Float64
        mesh = make_warped_uniform_hex(T, M, M, M, 0.0, 1.0, A_amp;
                                          periodic = true)
        elem = make_element(T, N); ops = make_operators(elem)
        geom = make_geometry(mesh, elem)
        u = Array{T, 4}(undef, N, N, N, mesh.Ne)
        H_vec = T[]
        for e in 1:mesh.Ne, k in 1:N, j in 1:N, i in 1:N
            x = geom.coords[1, i, j, k, e]
            y = geom.coords[2, i, j, k, e]
            z = geom.coords[3, i, j, k, e]
            u[i, j, k, e] = sin(2π * x) * sin(2π * y) * sin(2π * z)
            push!(H_vec, geom.Hphys[i, j, k, e])
        end
        u̇ = zeros(T, N, N, N, mesh.Ne); ü = similar(u)
        wave_strong_rhs_mesh!(ü, u, u̇, mesh, geom, ops)
        uv = vec(u); v̈ = vec(ü)
        return -dot(uv, H_vec .* v̈) / dot(uv, H_vec .* uv)
    end

    @testset "Rayleigh quotient is continuous in A" begin
        _progress("warped: Rayleigh vs A")
        λ_ex = 3 * (2π)^2
        errs = Float64[abs(_rayleigh(A, 4, 4) - λ_ex)
                       for A in (0.0, 0.01, 0.05, 0.1)]
        # Error grows monotonically (roughly) with A; at A = 0 it's at
        # the level of the existing periodic uniform-cube test.
        @test errs[1] < 0.01                # A = 0 matches periodic baseline
        @test errs[4] > errs[1]             # larger A → larger error
        @test errs[end] < 5                 # bounded even at A = 0.1
    end

    @testset "h-refinement at A = 0.05 — spectral convergence (HEADLINE)" begin
        _progress("warped: h-refinement (headline)")
        λ_ex = 3 * (2π)^2
        errs = Float64[abs(_rayleigh(0.05, M, 4) - λ_ex) for M in (2, 4, 8)]
        @test all(errs .> 0)
        @test errs[2] < errs[1] / 3         # M = 2 → 4: ≥ 3× shrink
        @test errs[3] < errs[2] / 50        # M = 4 → 8: ≥ 50× shrink
        # Diagnostic conclusion: inter-element SAT is consistent on
        # curvilinear-periodic. The inflated-cube residual must be
        # outer-Sommerfeld-driven, not inter-element-driven.
        @test errs[end] < 1e-2
    end

    @testset "Discrete spectrum on warped mesh (pure-real, non-positive)" begin
        _progress("warped: spectrum")
        T = Float64; N = 3
        mesh = make_warped_uniform_hex(T, 2, 2, 2, 0.0, 1.0, 0.05;
                                          periodic = true)
        elem = make_element(T, N); ops = make_operators(elem)
        geom = make_geometry(mesh, elem)
        ndof = N^3 * mesh.Ne
        L_mat = zeros(T, ndof, ndof)
        u   = zeros(T, N, N, N, mesh.Ne)
        u̇   = zeros(T, N, N, N, mesh.Ne)
        ü   = similar(u)
        uv  = vec(u); v̈ = vec(ü)
        for k in 1:ndof
            uv[k] = 1.0
            wave_strong_rhs_mesh!(ü, u, u̇, mesh, geom, ops)
            uv[k] = 0.0
            L_mat[:, k] .= v̈
        end
        λ = eigvals(L_mat)
        # Discovery: the conservative scheme inherits its uniform-cube
        # spectral properties on curvilinear-periodic too.
        @test maximum(real, λ)      < 1e-10
        @test maximum(abs ∘ imag, λ) < 1e-10
        @test minimum(real, λ)      < -1.0
    end

    @testset "Robust-stability sqrt(eps) noise on warped mesh" begin
        _progress("warped: robust stability noise")
        T = Float64; N = 3
        mesh = make_warped_uniform_hex(T, 2, 2, 2, 0.0, 1.0, 0.05;
                                          periodic = true)
        elem = make_element(T, N); ops = make_operators(elem)
        geom = make_geometry(mesh, elem)
        # Build the L matrix to set CFL.
        ndof = N^3 * mesh.Ne
        L_mat = zeros(T, ndof, ndof)
        u   = zeros(T, N, N, N, mesh.Ne)
        u̇   = zeros(T, N, N, N, mesh.Ne)
        ü   = similar(u)
        uv  = vec(u); v̈ = vec(ü)
        for k in 1:ndof
            uv[k] = 1.0
            wave_strong_rhs_mesh!(ü, u, u̇, mesh, geom, ops)
            uv[k] = 0.0
            L_mat[:, k] .= v̈
        end
        ω_max = sqrt(-minimum(real, eigvals(L_mat)))
        dt    = 0.5 * 2 / ω_max
        # IC: analytic eigenmode + sqrt(eps) noise on (u, u̇).
        u .= 0; u̇ .= 0
        for e in 1:mesh.Ne, k in 1:N, j in 1:N, i in 1:N
            x = geom.coords[1, i, j, k, e]
            y = geom.coords[2, i, j, k, e]
            z = geom.coords[3, i, j, k, e]
            u[i, j, k, e] = sin(2π * x) * sin(2π * y) * sin(2π * z)
        end
        Random.seed!(2026_06_04)
        amp = sqrt(eps(T))
        u  .+= amp .* randn(size(u))
        u̇  .+= amp .* randn(size(u̇))
        max_u0 = maximum(abs, u)
        wave_strong_rhs_mesh!(ü, u, u̇, mesh, geom, ops)
        u̇ .+= (dt / 2) .* ü
        for _ in 1:500
            u .+= dt .* u̇
            wave_strong_rhs_mesh!(ü, u, u̇, mesh, geom, ops)
            u̇ .+= dt .* ü
        end
        @test all(isfinite, u)
        @test all(isfinite, u̇)
        @test maximum(abs, u) < 5 * max_u0
    end
end
