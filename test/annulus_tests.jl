@testitem "annulus" tags=[:cpu] begin
    _progress(m) = (printstyled(stderr, "  • ", m, "\n"; color = :cyan); flush(stderr))

# 2D annulus (pure shell ring) with a radial shift that is SUPERLUMINAL
# at the inner circle and SUBLUMINAL (<0.1) at the outer circle — the 2D
# BH-excision setup. The inner circle is excised (no SAT = pure outflow,
# mesh tag 8) and the outer circle is absorbing (Sommerfeld). In this
# solver's convention a face's incoming-mode speed is a_n + β^n with
# β^n = n̂·β; the OUTWARD radial shift makes β^n < −1 at the inner circle
# (outward normal −r̂) ⇒ a superluminal-outflow face, correctly handled
# by excision. Verified: the RHS spectrum is stable and a noisy
# evolution stays bounded with non-increasing energy.

using HexMeshes: make_annulus_mesh
using HexSBPSAT: make_element, make_operators, make_geometry, make_metric_terms2d
using LinearAlgebra
using Test
using WaveToySecondOrder: make_coef2d, sample_background2d!, make_wave2d_workspace,
                          wave2d_curved_rhs!, make_bc2d, evolve2d

_radial_bg(T, V) = WaveToySecondOrder._background2d(:radial_shift, T;
                        A = V, d = 1, shift = (zero(T), zero(T)),
                        R1 = T(1)/2, R2 = T(2))[1]

@testset "annulus: inner excision + outer Sommerfeld, radial superluminal shift" begin
    T = Float64; N = 4

    _progress("annulus: radial shift crosses the light cone across the shell")
    @testset "radial shift superluminal inner, subluminal outer" begin
        mesh = make_annulus_mesh(T, T(1)/2, T(2), 3)
        elem = make_element(T, N); ops = make_operators(elem)
        geom = make_geometry(mesh, elem); coef = make_coef2d(geom)
        xg = geom.coords[1,:,:,:]; yg = geom.coords[2,:,:,:]
        sample_background2d!(coef, _radial_bg(T, 1.5), zero(T), xg, yg)
        r  = sqrt.(xg.^2 .+ yg.^2)
        bn = sqrt.(coef.b1.^2 .+ coef.b2.^2)        # |β| (a_n = 1 here)
        @test minimum(bn[r .< 0.6]) > 1             # superluminal at inner
        @test maximum(bn[r .> 1.9]) < T(1)/10       # subluminal (<0.1) at outer
        @test Int8(8) in geom.conn.bdry             # inner excision tag present
        @test Int8(7) in geom.conn.bdry             # outer Sommerfeld tag present
    end

    _progress("annulus: RHS spectrum stable with inner excision")
    @testset "spectrum: max Re(λ) ≤ round-off" begin
        mesh = make_annulus_mesh(T, T(1)/2, T(2), 2)
        elem = make_element(T, N); ops = make_operators(elem)
        geom = make_geometry(mesh, elem); metric = make_metric_terms2d(geom, ops)
        ws = make_wave2d_workspace(geom, ops); coef = make_coef2d(geom); Ne = geom.Ne
        sample_background2d!(coef, _radial_bg(T, 1.2), zero(T),
                             geom.coords[1,:,:,:], geom.coords[2,:,:,:])
        # Inner faces (tag 8) excised; outer (tag 7) Sommerfeld.
        bc = make_bc2d((:sommerfeld,:sommerfeld,:sommerfeld,:sommerfeld); excision_tag = 8)
        nn = N*N*Ne; n = 2nn; A = zeros(T, n, n)
        Φ = zeros(T,N,N,Ne); Π = similar(Φ); Φ̇ = similar(Φ); Π̇ = similar(Φ)
        for j in 1:n
            fill!(Φ,0); fill!(Π,0); j ≤ nn ? (Φ[j]=1) : (Π[j-nn]=1)
            wave2d_curved_rhs!(Φ̇, Π̇, Φ, Π, coef; geom, ops, ws, ε_KO=0.1,
                               bc2d=bc, metric)
            A[1:nn,j]=vec(Φ̇); A[nn+1:end,j]=vec(Π̇)
        end
        λ = eigvals(A)
        @test maximum(real, λ) ≤ 1e-4 * maximum(abs, λ)
    end

    _progress("annulus: noisy evolution bounded, energy non-increasing")
    @testset "noise evolution stable (excision + Sommerfeld)" begin
        res = evolve2d(; mesh_kind = :annulus, R1 = 0.5, R2 = 2.0, M = 4, N = 4,
                       background = :radial_shift, A = 1.5, ic = :noise,
                       ε_KO = 0.1, t1 = 2.0, Nt = 8, cfl = 0.1)
        @test all(isfinite, res.Φ_final)
        @test maximum(res.energy) ≤ res.energy[1] * (1 + 1e-6)   # no growth
        @test res.energy[end] < res.energy[1]                    # net decay
    end
end

end
