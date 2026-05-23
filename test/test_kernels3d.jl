# Tests for `src/kernels3d.jl`: curvilinear-aware `rhs3d!`. The full
# wave-evolution check against the analytic separable eigenmode is the
# end-to-end correctness gate.

using WaveToySecondOrder
using WaveToySecondOrder: make_element, make_operators,
    make_cubical_mesh, make_inflated_cube_mesh, make_geometry,
    initialize3d!, rhs3d!, recommended_dt
using OrdinaryDiffEqSymplecticRK
using Test

@testset "kernels3d" begin

    @testset "wave evolution matches analytic solution (M=4, N=5)" begin
        # Evolve u_tt = ∇²u with the separable analytic eigenmode
        #   u(x,y,z,t) = sin(2π x)·sin(2π y)·sin(2π z) · cos(ω·t)
        # on the unit cube with homogeneous Dirichlet BC on all six outer
        # faces. ω² = kx² + ky² + kz² = 3·(2π)².
        N    = 5
        M    = 4
        elem = make_element(Float64, N)
        ops  = make_operators(elem)
        mesh = make_cubical_mesh(Float64, M, 0.0, 1.0)
        geom = make_geometry(mesh, elem)
        coords = geom.coords

        dx = elem.h * (1 / M)        # node spacing within an element of width 1/M

        u  = Array{Float64,4}(undef, N, N, N, mesh.Ne)
        u̇  = similar(u)

        A  = 1.0
        kx = ky = kz = 2π
        ω  = sqrt(kx^2 + ky^2 + kz^2)
        initialize3d!(u, u̇, coords, 0.0; A, kx, ky, kz, ω)

        τ  = 3//2 * (N-1)^2
        dt = (1//2 * dx) / sqrt(3)
        t1 = 1.0   # ≈ 1.73 periods of the eigenmode

        bdry_values = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        f!(ü, u̇, u, p, t) = rhs3d!(ü, u, u̇, bdry_values; geom, ops, τ)
        prob = SecondOrderODEProblem(f!, u̇, u, (0.0, t1))
        sol  = solve(prob, KahanLi8(); dt)

        u_exact = similar(u);  u̇_exact = similar(u)
        initialize3d!(u_exact, u̇_exact, coords, t1; A, kx, ky, kz, ω)

        # SecondOrderODEProblem state layout: [du; u]
        n     = N^3 * mesh.Ne
        final = sol(t1)
        u_num = reshape(final[n+1 : 2n], N, N, N, mesh.Ne)

        # Empirical error at this resolution is ≈ 5e-4; allow ~10× margin.
        @test maximum(abs, u_num - u_exact) < 5e-3
    end

    @testset "inflated cube evolution stays bounded (M=2, N=4, R=0.3)" begin
        # Quick stability check on the curvilinear / multi-patch mesh: the
        # 7-patch inflated cube exercises non-axis-aligned face matchings,
        # left-handed outer-patch frames, and the curvilinear penalty
        # threshold. With `τ ≈ 4·(N−1)²` (the curvilinear rule of thumb)
        # the discrete operator is negative semi-definite, so a symplectic
        # integrator preserves bounded energy.
        N    = 4
        M    = 2
        R    = 0.3
        elem = make_element(Float64, N)
        ops  = make_operators(elem)
        mesh = make_inflated_cube_mesh(Float64, M, R)
        geom = make_geometry(mesh, elem)
        coords = geom.coords

        # Domain-normalised sine IC on [-1, +1]³ (vanishes on outer faces).
        x0, x1, L_ = -1.0, 1.0, 2.0
        kx = ky = kz = 3π
        u  = Array{Float64,4}(undef, N, N, N, mesh.Ne)
        u̇ = similar(u)
        for e in 1:mesh.Ne, k in 1:N, j in 1:N, i in 1:N
            X = (coords[1, i, j, k, e] - x0) / L_
            Y = (coords[2, i, j, k, e] - x0) / L_
            Z = (coords[3, i, j, k, e] - x0) / L_
            u[i, j, k, e]  = sin(kx*X) * sin(ky*Y) * sin(kz*Z)
            u̇[i, j, k, e] = 0.0
        end
        u0_max = maximum(abs, u)

        τ  = 8.0 * (N-1)^2                  # curvilinear rule of thumb
        dt = recommended_dt(geom, ops, τ)   # power-iteration estimate
        bdry_values = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        f!(ü, u̇, u, p, t) = rhs3d!(ü, u, u̇, bdry_values; geom, ops, τ)
        prob = SecondOrderODEProblem(f!, u̇, u, (0.0, 0.1))
        sol  = solve(prob, KahanLi8(); dt,
                     save_everystep = false, save_start = false,
                     dense = false, save_end = true)

        n     = N^3 * mesh.Ne
        final = sol.u[end]
        u_end = reshape(view(final, n+1 : 2n), N, N, N, mesh.Ne)

        @test all(isfinite, u_end)
        # IC has |u| ≤ 1; with a coercive operator and a symplectic
        # integrator the amplitude should stay near 1 (some swell is OK
        # but a 2× margin is well below any instability signature).
        @test maximum(abs, u_end) < 2 * u0_max
    end

end
