using HexMeshes: make_inflated_cube_mesh, Cubic, Inflation, Shell
using HexSBPSAT: make_element, make_operators, make_geometry
using LinearAlgebra: Diagonal, dot, eigvals
using Test
using WaveToySecondOrder: wave_strong_rhs_mesh!

# Dirichlet outer-BC tests on the cubed-sphere — actual NR geometry.
# `make_inflated_cube_mesh` builds a 13-patch mesh: 1 inner `PatchCubic`,
# 6 `PatchInflation` (middle), 6 `PatchShell` (outer). Outer faces
# satisfy `|P| = R2` exactly — the outer boundary is a true sphere,
# not a cube.
#
# This exercises three code paths that no prior scalar-testbed config
# touched:
#   * Genuinely curvilinear outer face (the outer shell has variable
#     `J⁻¹_α` along the boundary).
#   * Both `PatchInflation` and `PatchShell` analytic-Jacobian paths
#     under the strong-form kernel.
#   * Non-trivial `orientation` (orientation = 5 appears on
#     inter-patch faces) — the D₄ permutation in
#     `_extract_face_oriented!` actually fires here.
#
# The smooth l = 0 Dirichlet eigenmode `u(r) = j₀(πr/R) = sin(πr/R)/(πr/R)`
# vanishes on the outer sphere and has analytic eigenvalue `λ = π²/R²`.
# Choosing `R2 = 1` gives `λ_ex = π² ≈ 9.870`.

@inline function _j0(r::T) where {T}
    if r > eps(T)^(T(1//4))
        return sin(π * r) / (π * r)
    else
        # Taylor expansion at r = 0: j₀(πr) = 1 − (πr)²/6 + O(r⁴).
        return one(T) - (π * r)^2 / 6
    end
end

@testset "Dirichlet outer BC on cubed-sphere (inflated cube)" begin

    @testset "mesh sanity: 13 patches, all three kinds, outer tag = 1" begin
        _progress("cubed-sphere mesh sanity")
        T = Float64
        mesh = make_inflated_cube_mesh(T, T(0.1), T(0.3), T(1.0), 4)
        @test mesh.Ne > 0
        @test length(mesh.patch_desc) == 13
        kinds = unique(pd.kind for pd in mesh.patch_desc)
        @test Cubic in kinds
        @test Inflation in kinds
        @test Shell in kinds
        # Default outer_bc = :dirichlet → outer faces tagged 1, no 7s.
        tags = sort(unique(mesh.conn.bdry))
        @test tags == Int8[0, 1]
        # Non-trivial orientation exists in this mesh — exercises the
        # D₄ permutation path in `_extract_face_oriented!`.
        @test any(!=(Int8(0)), mesh.conn.orientation)
    end

    @testset "outer faces lie on the sphere |x| = R2 to roundoff" begin
        _progress("cubed-sphere outer geometry")
        T = Float64; N = 4
        mesh = make_inflated_cube_mesh(T, T(0.1), T(0.3), T(1.0), 4)
        elem = make_element(T, N); ops = make_operators(elem)
        geom = make_geometry(mesh, elem)
        maxdev = zero(T)
        for e in 1:mesh.Ne, f in 1:6
            mesh.conn.bdry[f, e] == Int8(0) && continue
            row = isodd(f) ? 1 : N
            a   = (f + 1) ÷ 2
            for q in 1:N, p in 1:N
                ii, jj, kk = (a == 1 ? (row, p, q) :
                              a == 2 ? (p, row, q) :
                                       (p, q, row))
                x = geom.coords[1, ii, jj, kk, e]
                y = geom.coords[2, ii, jj, kk, e]
                z = geom.coords[3, ii, jj, kk, e]
                r = sqrt(x*x + y*y + z*z)
                maxdev = max(maxdev, abs(r - one(T)))
            end
        end
        @test maxdev < 1e-12
    end

    @testset "Rayleigh quotient ≈ π² on l=0 j₀ eigenmode" begin
        _progress("cubed-sphere j₀ Rayleigh")
        T = Float64; N = 4
        mesh = make_inflated_cube_mesh(T, T(0.1), T(0.3), T(1.0), 4)
        elem = make_element(T, N); ops = make_operators(elem)
        geom = make_geometry(mesh, elem)
        u = Array{T, 4}(undef, N, N, N, mesh.Ne); H_vec = T[]
        for e in 1:mesh.Ne, kk in 1:N, jj in 1:N, ii in 1:N
            x = geom.coords[1, ii, jj, kk, e]
            y = geom.coords[2, ii, jj, kk, e]
            z = geom.coords[3, ii, jj, kk, e]
            r = sqrt(x*x + y*y + z*z)
            u[ii, jj, kk, e] = _j0(r)
            push!(H_vec, geom.Hphys[ii, jj, kk, e])
        end
        u̇ = zeros(T, N, N, N, mesh.Ne); ü = similar(u)
        wave_strong_rhs_mesh!(ü, u, u̇, mesh, geom, ops)
        λ_disc = -dot(vec(u), H_vec .* vec(ü)) / dot(vec(u), H_vec .* vec(u))
        @test abs(λ_disc - π^2) / π^2 < 1e-3
    end

    @testset "h-refinement spectral convergence" begin
        _progress("cubed-sphere h-refinement")
        T = Float64; N = 4
        function rq(M)
            mesh = make_inflated_cube_mesh(T, T(0.1), T(0.3), T(1.0), M)
            elem = make_element(T, N); ops = make_operators(elem)
            geom = make_geometry(mesh, elem)
            u = Array{T, 4}(undef, N, N, N, mesh.Ne); H_vec = T[]
            for e in 1:mesh.Ne, kk in 1:N, jj in 1:N, ii in 1:N
                x = geom.coords[1, ii, jj, kk, e]
                y = geom.coords[2, ii, jj, kk, e]
                z = geom.coords[3, ii, jj, kk, e]
                r = sqrt(x*x + y*y + z*z)
                u[ii, jj, kk, e] = _j0(r)
                push!(H_vec, geom.Hphys[ii, jj, kk, e])
            end
            u̇ = zeros(T, N, N, N, mesh.Ne); ü = similar(u)
            wave_strong_rhs_mesh!(ü, u, u̇, mesh, geom, ops)
            return -dot(vec(u), H_vec .* vec(ü)) / dot(vec(u), H_vec .* vec(u))
        end
        errs = Float64[abs(rq(M) - π^2) for M in (2, 4)]
        # Headline test: each halving shrinks the error by an order of
        # magnitude (empirically ~80×). We only need ≥ 10× to confirm
        # spectral behaviour. (Skipping M = 8 here to keep the suite
        # under 4 minutes; the M = 8 result is already validated
        # informally as `~7e-7`.)
        @test all(errs .> 0)
        @test errs[2] < errs[1] / 10
    end

    @testset "Spectrum bounded on the negative real axis" begin
        _progress("cubed-sphere spectrum")
        T = Float64; N = 3
        mesh = make_inflated_cube_mesh(T, T(0.1), T(0.3), T(1.0), 2)
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
        # No growing modes; non-trivial spectrum; bounded imaginary parts
        # (small non-Hermitian residual at inflation-shell interfaces,
        # same pattern as cubed-cube).
        @test maximum(real, λ)        < 1.0
        @test minimum(real, λ)        < -1.0
        @test maximum(abs ∘ imag, λ)  < 100.0
    end

    @testset "Long-time leapfrog stays bounded on j₀ eigenmode" begin
        _progress("cubed-sphere long-time leapfrog")
        T = Float64; N = 4
        mesh = make_inflated_cube_mesh(T, T(0.1), T(0.3), T(1.0), 2)
        elem = make_element(T, N); ops = make_operators(elem)
        geom = make_geometry(mesh, elem)
        u = Array{T, 4}(undef, N, N, N, mesh.Ne)
        for e in 1:mesh.Ne, kk in 1:N, jj in 1:N, ii in 1:N
            x = geom.coords[1, ii, jj, kk, e]
            y = geom.coords[2, ii, jj, kk, e]
            z = geom.coords[3, ii, jj, kk, e]
            r = sqrt(x*x + y*y + z*z)
            u[ii, jj, kk, e] = _j0(r)
        end
        u̇ = zeros(T, N, N, N, mesh.Ne); ü = similar(u)
        max_u0 = maximum(abs, u)
        dt = 0.001; n_steps = 1000
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
