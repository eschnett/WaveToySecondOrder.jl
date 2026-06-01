using HexMeshes: make_uniform_hex, make_warped_uniform_hex
using HexSBPSAT: make_element, make_operators, make_geometry
using LinearAlgebra: dot, eigvals
using Random: Random
using Test
using WaveToySecondOrder: wave_strong_rhs_mesh!

# Phase 2: Dirichlet outer-BC tests on the non-periodic warped cube.
# Combines the C2 curvilinear chain rule (validated by the periodic-
# warped diagnostic) with the Dirichlet SAT branch (validated by the
# flat Dirichlet tests). No new source code — this test file just
# exercises the combination.

@testset "Dirichlet outer BC on warped non-periodic cube" begin

    @testset "A = 0 regression vs flat-Dirichlet path" begin
        _progress("warped Dirichlet: A = 0 regression")
        T = Float64; N = 4
        m0 = make_uniform_hex(T, 3, 3, 3, 0.0, 1.0)
        m1 = make_warped_uniform_hex(T, 3, 3, 3, 0.0, 1.0, 0.0;
                                       periodic = false)
        elem = make_element(T, N); ops = make_operators(elem)
        g0   = make_geometry(m0, elem); g1 = make_geometry(m1, elem)
        Random.seed!(2026_06_05)
        u  = randn(T, N, N, N, m0.Ne)
        u̇ = randn(T, N, N, N, m0.Ne)
        ü0 = similar(u); ü1 = similar(u)
        wave_strong_rhs_mesh!(ü0, u, u̇, m0, g0, ops)
        wave_strong_rhs_mesh!(ü1, u, u̇, m1, g1, ops)
        @test maximum(abs, ü0 .- ü1) < 1e-10
        # Mesh outer-tag layout must match too.
        @test sort(unique(m0.conn.bdry)) == sort(unique(m1.conn.bdry))
    end

    function _rayleigh(A_amp, M, N)
        T    = Float64
        mesh = make_warped_uniform_hex(T, M, M, M, 0.0, 1.0, A_amp;
                                          periodic = false)
        elem = make_element(T, N); ops = make_operators(elem)
        geom = make_geometry(mesh, elem)
        u = Array{T, 4}(undef, N, N, N, mesh.Ne)
        H_vec = T[]
        for e in 1:mesh.Ne, k in 1:N, j in 1:N, i in 1:N
            x = geom.coords[1, i, j, k, e]
            y = geom.coords[2, i, j, k, e]
            z = geom.coords[3, i, j, k, e]
            u[i, j, k, e] = sin(π * x) * sin(π * y) * sin(π * z)
            push!(H_vec, geom.Hphys[i, j, k, e])
        end
        u̇ = zeros(T, N, N, N, mesh.Ne); ü = similar(u)
        wave_strong_rhs_mesh!(ü, u, u̇, mesh, geom, ops)
        uv = vec(u); v̈ = vec(ü)
        return -dot(uv, H_vec .* v̈) / dot(uv, H_vec .* uv)
    end

    @testset "Rayleigh quotient ≈ 3π² at A = 0.05, M = 4, N = 4" begin
        _progress("warped Dirichlet: eigenmode Rayleigh")
        λ_ex = 3 * π^2
        λ    = _rayleigh(0.05, 4, 4)
        @test abs(λ - λ_ex) / λ_ex < 0.05
    end

    @testset "h-refinement convergence at A = 0.05" begin
        _progress("warped Dirichlet: h-refinement")
        T = Float64
        λ_ex = 3 * π^2
        errs = Float64[abs(_rayleigh(0.05, M, 4) - λ_ex) for M in (2, 4, 8)]
        @test all(errs .> 0)
        @test errs[end] < errs[1]
        # Geometric mean of the two halving ratios — robust to a
        # mid-refinement bobble (the convergence is asymptotic).
        gmean = (errs[1] / errs[end])^(1 / (length(errs) - 1))
        @test gmean > 4.0
        @test errs[end] < 1e-3
    end

    @testset "Spectrum is pure-real on negative axis (A = 0.05)" begin
        _progress("warped Dirichlet: spectrum")
        T = Float64; N = 3
        mesh = make_warped_uniform_hex(T, 2, 2, 2, 0.0, 1.0, 0.05;
                                          periodic = false)
        elem = make_element(T, N); ops = make_operators(elem)
        geom = make_geometry(mesh, elem)
        ndof  = N^3 * mesh.Ne
        L_mat = zeros(T, ndof, ndof)
        u  = zeros(T, N, N, N, mesh.Ne); u̇ = zeros(T, N, N, N, mesh.Ne); ü = similar(u)
        uv = vec(u); v̈ = vec(ü)
        for k in 1:ndof
            uv[k] = 1.0
            wave_strong_rhs_mesh!(ü, u, u̇, mesh, geom, ops)
            uv[k] = 0.0
            L_mat[:, k] .= v̈
        end
        λ = eigvals(L_mat)
        @test maximum(real, λ)       < 1e-10
        @test maximum(abs ∘ imag, λ)  < 1e-10
        @test minimum(real, λ)       < -1.0
    end

    @testset "Long-time leapfrog stays bounded" begin
        _progress("warped Dirichlet: long-time leapfrog")
        T = Float64; N = 4
        mesh = make_warped_uniform_hex(T, 2, 2, 2, 0.0, 1.0, 0.05;
                                          periodic = false)
        elem = make_element(T, N); ops = make_operators(elem)
        geom = make_geometry(mesh, elem)
        u = Array{T, 4}(undef, N, N, N, mesh.Ne)
        for e in 1:mesh.Ne, k in 1:N, j in 1:N, i in 1:N
            x = geom.coords[1, i, j, k, e]
            y = geom.coords[2, i, j, k, e]
            z = geom.coords[3, i, j, k, e]
            u[i, j, k, e] = sin(π * x) * sin(π * y) * sin(π * z)
        end
        u̇ = zeros(T, N, N, N, mesh.Ne); ü = similar(u)
        max_u0 = maximum(abs, u)
        dt = 0.005; n_steps = 1000
        wave_strong_rhs_mesh!(ü, u, u̇, mesh, geom, ops)
        u̇ .+= (dt / 2) .* ü
        bound_seen = max_u0
        for _ in 1:n_steps
            u .+= dt .* u̇
            wave_strong_rhs_mesh!(ü, u, u̇, mesh, geom, ops)
            u̇ .+= dt .* ü
            bound_seen = max(bound_seen, maximum(abs, u))
        end
        @test all(isfinite, u)
        @test all(isfinite, u̇)
        @test bound_seen < 5 * max_u0
    end

end
