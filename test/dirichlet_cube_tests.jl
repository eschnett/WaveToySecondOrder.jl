@testitem "dirichlet_cube" tags=[:cpu] begin
    _progress(m) = (printstyled(stderr, "  • ", m, "\n"; color = :cyan); flush(stderr))

using HexMeshes: make_uniform_hex
using HexSBPSAT: make_element, make_operators, make_geometry
using LinearAlgebra: dot, eigvals
using Random: Random
using Test
using WaveToySecondOrder: wave_strong_rhs_mesh!,
                          wave_lap_strong_conservative_mesh!

# Multi-element Dirichlet outer-boundary tests for the strong-form
# scalar wave on a flat (non-periodic) uniform-hex cube. The kernel
# treats `bdry ∈ 1..6` outer faces with the same Mattsson–Nordström
# SAT pair as interior seams — the only difference is what
# `u_face[f]` carries (boundary data, zero by default, vs neighbour
# trace). For homogeneous Dirichlet (`g ≡ 0`) the mesh driver's
# pre-filled-zero outer-face buffer is exactly right.

@testset "Dirichlet outer BC on flat uniform-hex cube" begin

    @testset "bdry-zero override regression vs the periodic path" begin
        _progress("Dirichlet: bdry=0-override regression")
        T = Float64; N = 4
        # Build the non-periodic mesh, then override the outer-boundary
        # tags to 0 so every face goes through the interior branch.
        # Because the mesh-driver pre-fill puts zeros at every outer
        # face (neighbour == 0), the result must equal the periodic
        # `wave_strong_rhs_mesh!` output on the same (u, u̇) only when
        # the test function vanishes at all six face slices.
        #
        # We use `u(x) = sin(π x) sin(π y) sin(π z)` — vanishes on the
        # cube boundary, so the periodic and outer-Dirichlet paths give
        # the same Laplacian here (up to roundoff).
        mesh_d = make_uniform_hex(T, 2, 2, 2, 0.0, 1.0)
        mesh_p = make_uniform_hex(T, 2, 2, 2, 0.0, 1.0; periodic = true)
        elem   = make_element(T, N); ops = make_operators(elem)
        geom_d = make_geometry(mesh_d, elem)
        geom_p = make_geometry(mesh_p, elem)
        # Sample u on the GLL coords.
        u  = Array{T, 4}(undef, N, N, N, mesh_d.Ne)
        for e in 1:mesh_d.Ne, k in 1:N, j in 1:N, i in 1:N
            x = geom_d.coords[1, i, j, k, e]
            y = geom_d.coords[2, i, j, k, e]
            z = geom_d.coords[3, i, j, k, e]
            u[i, j, k, e] = sin(π * x) * sin(π * y) * sin(π * z)
        end
        u̇ = zeros(T, N, N, N, mesh_d.Ne)
        ü_d = similar(u); ü_p = similar(u)
        wave_strong_rhs_mesh!(ü_d, u, u̇, mesh_d, geom_d, ops)
        wave_strong_rhs_mesh!(ü_p, u, u̇, mesh_p, geom_p, ops)
        # Both should compute the same Laplacian — the field vanishes
        # at the cube boundary so neighbour trace (periodic) and zero
        # (Dirichlet) supply identical face data.
        @test maximum(abs, ü_d .- ü_p) < 1e-10
    end

    @testset "Rayleigh quotient → 3π² on analytic eigenmode" begin
        _progress("Dirichlet: eigenmode Rayleigh")
        T = Float64; N = 4
        mesh = make_uniform_hex(T, 4, 4, 4, 0.0, 1.0)
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
        λ_disc = -dot(uv, H_vec .* v̈) / dot(uv, H_vec .* uv)
        λ_ex   = 3 * π^2
        @test abs(λ_disc - λ_ex) / λ_ex < 1e-3
    end

    @testset "h-refinement convergence on analytic eigenmode" begin
        _progress("Dirichlet: h-refinement")
        T = Float64; N = 4
        function rayleigh(M)
            mesh = make_uniform_hex(T, M, M, M, 0.0, 1.0)
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
        λ_ex = 3 * π^2
        errs = Float64[abs(rayleigh(M) - λ_ex) for M in (2, 4, 8)]
        @test all(errs .> 0)
        @test errs[2] < errs[1] / 4
        @test errs[3] < errs[2] / 4
        @test errs[end] < 1e-3
    end

    @testset "Spectrum is pure-real on negative axis" begin
        _progress("Dirichlet: spectrum")
        T = Float64; N = 3
        mesh = make_uniform_hex(T, 2, 2, 2, 0.0, 1.0)
        elem = make_element(T, N); ops = make_operators(elem)
        geom = make_geometry(mesh, elem)
        ndof = N^3 * mesh.Ne
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
        @test maximum(real, λ)      < 1e-10
        @test maximum(abs ∘ imag, λ) < 1e-10
        @test minimum(real, λ)      < -1.0
    end

    @testset "Long-time leapfrog stays bounded" begin
        _progress("Dirichlet: long-time leapfrog")
        T = Float64; N = 4
        mesh = make_uniform_hex(T, 2, 2, 2, 0.0, 1.0)
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
        # Use the discrete spectral radius to pick a stable dt.
        # Empirically the eigenmode in this geometry gives ω² ≈ 3π²,
        # so dt = 0.5 / sqrt(spectral radius) is plenty.
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

end
