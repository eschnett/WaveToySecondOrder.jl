# Tests for `src/kernels3d.jl`: per-element `apply_laplacian3d!`, `FaceData`,
# and the global `rhs3d!`. Includes a full 3D wave-evolution check against
# the analytic separable eigenmode.

using WaveToySecondOrder
using WaveToySecondOrder: make_element, make_operators,
    make_cubical_mesh, element_coords,
    apply_laplacian3d!, initialize3d!, rhs3d!, FaceData
using OrdinaryDiffEqSymplecticRK
using StaticArrays
using Test

@testset "kernels3d" begin

    @testset "apply_laplacian3d! recovers ∇²(x²+y²+z²) = 6" begin
        # Single element on the unit cube. With exact Dirichlet data and
        # exact boundary gradients on every outer face (α = 1 = full
        # Nitsche), every SAT contribution vanishes for this quadratic, so
        # the result is `L · u + L · u + L · u` along the three axes = 6.
        elem = make_element(Float64, 5)
        ops  = make_operators(elem)
        xs   = elem.xs

        u = [xs[i]^2 + xs[j]^2 + xs[k]^2 for i in 1:5, j in 1:5, k in 1:5]
        Lu = similar(u)

        # Face value slices: evaluate the polynomial at the two boundary
        # planes orthogonal to each axis.
        ux_m = SMatrix{5,5,Float64}([0.0     + xs[j]^2 + xs[k]^2 for j in 1:5, k in 1:5])
        ux_p = SMatrix{5,5,Float64}([1.0     + xs[j]^2 + xs[k]^2 for j in 1:5, k in 1:5])
        uy_m = SMatrix{5,5,Float64}([xs[i]^2 + 0.0     + xs[k]^2 for i in 1:5, k in 1:5])
        uy_p = SMatrix{5,5,Float64}([xs[i]^2 + 1.0     + xs[k]^2 for i in 1:5, k in 1:5])
        uz_m = SMatrix{5,5,Float64}([xs[i]^2 + xs[j]^2 + 0.0     for i in 1:5, j in 1:5])
        uz_p = SMatrix{5,5,Float64}([xs[i]^2 + xs[j]^2 + 1.0     for i in 1:5, j in 1:5])

        # Exact boundary gradients: ∂u/∂xᵢ = 2xᵢ, so 0 at the −face and 2
        # at the +face for every axis.
        Gminus = zero(SMatrix{5,5,Float64})
        Gplus  = SMatrix{5,5,Float64}(fill(2.0, 5, 5))

        facex = FaceData(ux_m, ux_p, Gminus, Gplus, 1.0, 1.0)
        facey = FaceData(uy_m, uy_p, Gminus, Gplus, 1.0, 1.0)
        facez = FaceData(uz_m, uz_p, Gminus, Gplus, 1.0, 1.0)

        apply_laplacian3d!(Lu, u, facex, facey, facez; ops, τ=64.0)

        @test maximum(abs, Lu .- 6.0) < 1e-10
    end

    @testset "FaceData has expected layout" begin
        z  = zero(SMatrix{5,5,Float64})
        fd = FaceData(z, z, z, z, 0.5, 0.5)
        @test fd isa FaceData{5, Float64, 25}
        @test isbitstype(typeof(fd))    # GPU-friendly value-type struct
    end

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
        coords = element_coords(mesh, elem)

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
        f!(ü, u̇, u, p, t) = rhs3d!(ü, u, u̇, bdry_values; mesh, ops, τ)
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

end
