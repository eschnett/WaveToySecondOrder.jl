@testitem "inflated_square" tags=[:cpu] begin
    _progress(m) = (printstyled(stderr, "  • ", m, "\n"; color = :cyan); flush(stderr))

# Curvilinear 2D scalar wave on the inflated-square mesh — the GOALS
# stepping stone after cubed-square. Unlike cubed-square (all
# connectivity orientation 0), inflated-square mixes orientations
# {0,1} between the inner square and the inflation patches, so this
# exercises the SAT's `_neigh_p` orientation transform end-to-end:
# free-stream/spectrum stability + analytic convergence with the
# free-stream-preserving curvilinear operator and the physical-normal
# Sommerfeld / Dirichlet boundary.

using HexMeshes: make_inflated_square_mesh
using HexSBPSAT: make_element, make_operators, make_geometry, make_metric_terms2d
using LinearAlgebra
using Test
using WaveToySecondOrder: AnalyticBackground2D, make_coef2d,
                          sample_background2d!, make_wave2d_workspace,
                          wave2d_curved_rhs!, make_bc2d

@testset "inflated-square curvilinear wave" begin
    T = Float64; N = 4
    flat = AnalyticBackground2D((t,x,y)->one(T), (t,x,y)->(zero(T),zero(T)),
                                (t,x,y)->(one(T),zero(T),one(T)))

    _progress("inflated-square: spectrum stable with Sommerfeld")
    @testset "spectrum: max Re(λ) ≤ round-off (Sommerfeld)" begin
        mesh = make_inflated_square_mesh(T, T(0.2), T(0.5), T(1.0), 2)
        elem = make_element(T, N); ops = make_operators(elem)
        geom = make_geometry(mesh, elem); metric = make_metric_terms2d(geom, ops)
        ws = make_wave2d_workspace(geom, ops); coef = make_coef2d(geom)
        Ne = geom.Ne
        sample_background2d!(coef, flat, zero(T),
                             geom.coords[1,:,:,:], geom.coords[2,:,:,:])
        bc = make_bc2d((:sommerfeld,:sommerfeld,:sommerfeld,:sommerfeld))
        nn = N*N*Ne; n = 2nn; A = zeros(T, n, n)
        Φ = zeros(T,N,N,Ne); Π = similar(Φ); Φ̇ = similar(Φ); Π̇ = similar(Φ)
        for j in 1:n
            fill!(Φ,0); fill!(Π,0); j ≤ nn ? (Φ[j]=1) : (Π[j-nn]=1)
            wave2d_curved_rhs!(Φ̇, Π̇, Φ, Π, coef; geom, ops, ws, ε_KO=0.1,
                               bc2d=bc, metric)
            A[1:nn,j]=vec(Φ̇); A[nn+1:end,j]=vec(Π̇)
        end
        λ = eigvals(A)
        @test maximum(real, λ) ≤ 1e-5 * maximum(abs, λ)
    end

    _progress("inflated-square: plane-wave convergence (Dirichlet)")
    @testset "plane-wave convergence (curved Dirichlet)" begin
        κ = T(2); ω = κ*sqrt(T(2))
        Φe(t,x,y) =  sin(κ*(x+y) - ω*t)
        Πe(t,x,y) = -ω*cos(κ*(x+y) - ω*t)
        De(t,x,y) =  κ*cos(κ*(x+y) - ω*t)
        errs = T[]
        for M in (2, 4)
            mesh = make_inflated_square_mesh(T, T(0.2), T(0.5), T(1.0), M)
            elem = make_element(T, N); ops = make_operators(elem)
            geom = make_geometry(mesh, elem); metric = make_metric_terms2d(geom, ops)
            ws = make_wave2d_workspace(geom, ops); coef = make_coef2d(geom)
            xg = geom.coords[1,:,:,:]; yg = geom.coords[2,:,:,:]
            Φ = Φe.(zero(T), xg, yg); Π = Πe.(zero(T), xg, yg)
            k = [similar(Φ) for _ in 1:8]; Φs = similar(Φ); Πs = similar(Π)
            h = 2*T(1.0)/M; dt = T(0.1)*minimum(diff(elem.xs))*h
            rhs(a,b,c,d,t) = begin
                sample_background2d!(coef, flat, t, xg, yg)
                bc = make_bc2d((:dirichlet,:dirichlet,:dirichlet,:dirichlet);
                               gΠ=Πe.(t,xg,yg), gDx=De.(t,xg,yg), gDy=De.(t,xg,yg))
                wave2d_curved_rhs!(a,b,c,d,coef; geom, ops, ws, ε_KO=0.0,
                                   bc2d=bc, metric)
            end
            t = zero(T)
            for _ in 1:ceil(Int, 0.4/dt)
                rhs(k[1],k[2],Φ,Π,t); @. Φs=Φ+dt/2*k[1]; @. Πs=Π+dt/2*k[2]
                rhs(k[3],k[4],Φs,Πs,t+dt/2); @. Φs=Φ+dt/2*k[3]; @. Πs=Π+dt/2*k[4]
                rhs(k[5],k[6],Φs,Πs,t+dt/2); @. Φs=Φ+dt*k[5]; @. Πs=Π+dt*k[6]
                rhs(k[7],k[8],Φs,Πs,t+dt)
                @. Φ+=dt/6*(k[1]+2k[3]+2k[5]+k[7]); @. Π+=dt/6*(k[2]+2k[4]+2k[6]+k[8])
                t += dt
            end
            push!(errs, sqrt(sum(@. (Φ - Φe(t,xg,yg))^2 * metric.Hd)))
        end
        @test all(isfinite, errs)
        @test errs[2] < errs[1]
        @test log2(errs[1]/errs[2]) > 1.8
    end
end

end
