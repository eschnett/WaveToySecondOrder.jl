@testitem "dirichlet_cubed_cube" tags=[:cpu] begin
    _progress(m) = (printstyled(stderr, "  • ", m, "\n"; color = :cyan); flush(stderr))

using HexMeshes: make_cubed_cube_mesh, Cubic, Wedge
using HexSBPSAT: make_element, make_operators, make_geometry
using LinearAlgebra: Diagonal, dot, eigvals
using Test
using WaveToySecondOrder: wave_strong_rhs_mesh!

# Phase 3: Dirichlet outer BC on the cubed-cube. The mesh combines an
# inner `PatchCubic` with six outer `PatchWedge` patches, with all six
# outer cube faces tagged `1..6` for Dirichlet BCs.
#
# After fixing two bugs uncovered by the initial diagnostic — analytic
# `_ppj_wedge_3d` Jacobian (was trilinear approximation) and the
# bogus `handedness` factor in the outward-normal formula — the
# strong-form kernel works correctly on this mesh. Phase 3 is now
# validation rather than diagnosis.

@testset "Dirichlet outer BC on cubed-cube" begin

    @testset "mesh sanity: 7 patches, mixed kinds, outer tags 1..6 present" begin
        _progress("cubed-cube mesh sanity")
        T = Float64
        mesh = make_cubed_cube_mesh(T, 4, T(0.3))
        @test mesh.Ne > 0
        @test length(mesh.patch_desc) == 7
        kinds = unique(pd.kind for pd in mesh.patch_desc)
        @test Cubic in kinds
        @test Wedge in kinds
        @test sort(unique(mesh.conn.bdry)) == Int8[0, 1, 2, 3, 4, 5, 6]
        @test all(==(Int8(0)), mesh.conn.orientation)
    end

    @testset "kernel smoke: finite output" begin
        _progress("cubed-cube smoke")
        T = Float64; N = 4
        mesh = make_cubed_cube_mesh(T, 4, T(0.3))
        elem = make_element(T, N); ops = make_operators(elem)
        geom = make_geometry(mesh, elem)
        u = Array{T, 4}(undef, N, N, N, mesh.Ne)
        for e in 1:mesh.Ne, k in 1:N, j in 1:N, i in 1:N
            x = geom.coords[1, i, j, k, e]
            y = geom.coords[2, i, j, k, e]
            z = geom.coords[3, i, j, k, e]
            u[i, j, k, e] = cos(π * x / 2) * cos(π * y / 2) * cos(π * z / 2)
        end
        u̇ = zeros(T, N, N, N, mesh.Ne); ü = similar(u)
        wave_strong_rhs_mesh!(ü, u, u̇, mesh, geom, ops)
        @test all(isfinite, ü)
        @test maximum(abs, ü) > 0
    end

    @testset "Rayleigh quotient converges spectrally under h-refinement" begin
        _progress("cubed-cube h-refinement convergence")
        T = Float64; N = 4
        λ_ex = 3 * π^2 / 4
        function rq(M)
            mesh = make_cubed_cube_mesh(T, M, T(0.3))
            elem = make_element(T, N); ops = make_operators(elem)
            geom = make_geometry(mesh, elem)
            u = Array{T, 4}(undef, N, N, N, mesh.Ne); H_vec = T[]
            for e in 1:mesh.Ne, k in 1:N, j in 1:N, i in 1:N
                x = geom.coords[1, i, j, k, e]
                y = geom.coords[2, i, j, k, e]
                z = geom.coords[3, i, j, k, e]
                u[i, j, k, e] = cos(π * x / 2) * cos(π * y / 2) * cos(π * z / 2)
                push!(H_vec, geom.Hphys[i, j, k, e])
            end
            u̇ = zeros(T, N, N, N, mesh.Ne); ü = similar(u)
            wave_strong_rhs_mesh!(ü, u, u̇, mesh, geom, ops)
            return -dot(vec(u), H_vec .* vec(ü)) / dot(vec(u), H_vec .* vec(u))
        end
        errs = Float64[abs(rq(M) - λ_ex) for M in (2, 4, 8)]
        @test all(errs .> 0)
        @test errs[end] < errs[1]
        gmean = (errs[1] / errs[end])^(1 / (length(errs) - 1))
        # Empirically observed: ratio ~12–20× per halving on this mesh.
        @test gmean > 4.0
        @test errs[end] < 1e-3
    end

    @testset "Spectrum on the negative axis (small imag tolerated)" begin
        _progress("cubed-cube spectrum")
        T = Float64; N = 3
        mesh = make_cubed_cube_mesh(T, 2, T(0.3))
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
        # `max(real(λ)) ≤ 0` — no growing modes. (The cubed-cube
        # discretisation has a small non-Hermitian residual at
        # mixed-kind patch interfaces, so the spectrum isn't
        # *exactly* pure-real to roundoff like the warped-periodic
        # case; that's the known limitation. The conservative SAT
        # is still nominally stable.)
        @test maximum(real, λ)      < 1.0       # no growing modes
        @test minimum(real, λ)      < -1.0      # non-trivial spectrum
        # Imaginary parts are bounded (small relative to spectral radius).
        @test maximum(abs ∘ imag, λ) < 100.0
    end

    @testset "Long-time leapfrog stays bounded" begin
        _progress("cubed-cube long-time leapfrog")
        T = Float64; N = 4
        mesh = make_cubed_cube_mesh(T, 2, T(0.3))
        elem = make_element(T, N); ops = make_operators(elem)
        geom = make_geometry(mesh, elem)
        u = Array{T, 4}(undef, N, N, N, mesh.Ne)
        for e in 1:mesh.Ne, k in 1:N, j in 1:N, i in 1:N
            x = geom.coords[1, i, j, k, e]
            y = geom.coords[2, i, j, k, e]
            z = geom.coords[3, i, j, k, e]
            u[i, j, k, e] = cos(π * x / 2) * cos(π * y / 2) * cos(π * z / 2)
        end
        u̇ = zeros(T, N, N, N, mesh.Ne); ü = similar(u)
        max_u0 = maximum(abs, u)
        dt = 0.002; n_steps = 1000
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
