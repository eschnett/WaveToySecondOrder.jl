@testitem "curvilinear_robustness" tags=[:cpu] begin
    _progress(m) = (printstyled(stderr, "  • ", m, "\n"; color = :cyan); flush(stderr))

# Curvilinear-path parity with the affine 2D tests (GOALS test
# categories): robust stability under noise, energy behaviour under the
# absorbing boundary, and a VARIABLE background (gaugewave, varying
# lapse) on a curved mesh. All on cubed-square with the
# free-stream-preserving conservative operator + physical-normal BCs.

using HexMeshes: make_cubed_square_mesh
using HexSBPSAT: make_element, make_operators, make_geometry,
                 make_metric_terms2d, make_workspace
using Random
using Test
using WaveToySecondOrder: AnalyticBackground2D, make_coef2d,
                          sample_background2d!, make_wave2d_workspace,
                          wave2d_curved_rhs!, wave2d_energy, make_bc2d

# One curved-mesh setup at angular resolution M.
function _curv_setup(::Type{T}, N, M) where {T}
    mesh = make_cubed_square_mesh(T, M, T(0.3))
    elem = make_element(T, N); ops = make_operators(elem)
    geom = make_geometry(mesh, elem); metric = make_metric_terms2d(geom, ops)
    ws = make_wave2d_workspace(geom, ops); coef = make_coef2d(geom)
    xg = geom.coords[1, :, :, :]; yg = geom.coords[2, :, :, :]
    dxmin = minimum(diff(elem.xs)) * (2 * T(1) / M)   # rough phys node spacing
    return (; mesh, elem, ops, geom, metric, ws, coef, xg, yg, dxmin)
end

# In-place RK4 step for the curvilinear RHS with a fixed bc bundle.
function _rk4_curv!(Φ, Π, t, dt, bg, s, bc; ε_KO, k, Φs, Πs)
    rhs(a, b, c, d, tt) = begin
        sample_background2d!(s.coef, bg, tt, s.xg, s.yg)
        wave2d_curved_rhs!(a, b, c, d, s.coef; s.geom, s.ops, s.ws,
                           ε_KO, bc2d = bc, s.metric)
    end
    rhs(k[1],k[2],Φ,Π,t);            @. Φs=Φ+dt/2*k[1]; @. Πs=Π+dt/2*k[2]
    rhs(k[3],k[4],Φs,Πs,t+dt/2);     @. Φs=Φ+dt/2*k[3]; @. Πs=Π+dt/2*k[4]
    rhs(k[5],k[6],Φs,Πs,t+dt/2);     @. Φs=Φ+dt*k[5];   @. Πs=Π+dt*k[6]
    rhs(k[7],k[8],Φs,Πs,t+dt)
    @. Φ += dt/6*(k[1]+2k[3]+2k[5]+k[7]); @. Π += dt/6*(k[2]+2k[4]+2k[6]+k[8])
    return nothing
end

@testset "curvilinear robustness / energy / variable background" begin
    T = Float64; N = 4
    flat = AnalyticBackground2D((t,x,y)->one(T), (t,x,y)->(zero(T),zero(T)),
                                (t,x,y)->(one(T),zero(T),one(T)))

    # ---- #3 robust stability: √eps noise stays bounded (Sommerfeld) ----
    _progress("curvilinear: noise bounded (Sommerfeld, ~8 crossings)")
    @testset "noise robustness (cubed-square, Sommerfeld + KO)" begin
        s = _curv_setup(T, N, 2)
        Random.seed!(20260604)
        amp = sqrt(eps(T))
        Φ = amp .* randn(T, N, N, s.geom.Ne); Π = amp .* randn(T, N, N, s.geom.Ne)
        bc = make_bc2d((:sommerfeld,:sommerfeld,:sommerfeld,:sommerfeld))
        k = [similar(Φ) for _ in 1:8]; Φs = similar(Φ); Πs = similar(Π)
        dt = T(0.1) * s.dxmin
        nst = ceil(Int, 16 / dt)        # domain ≈ 2, speed ≈ 1 ⇒ ~8 crossings
        t = zero(T)
        for _ in 1:nst
            _rk4_curv!(Φ, Π, t, dt, flat, s, bc; ε_KO=0.1, k, Φs, Πs); t += dt
        end
        @test all(isfinite, Φ) && all(isfinite, Π)
        @test maximum(abs, Φ) < 1000 * amp
    end

    # ---- #4 energy: monotone non-increasing under the absorbing BC ----
    _progress("curvilinear: energy non-increasing (Sommerfeld absorbs)")
    @testset "energy decays (cubed-square, Sommerfeld)" begin
        s = _curv_setup(T, N, 3)
        # Smooth centred pulse that radiates out through the boundary.
        Φ = exp.(-(s.xg.^2 .+ s.yg.^2) ./ (2 * T(0.15)^2)); Π = zeros(T, N, N, s.geom.Ne)
        bc = make_bc2d((:sommerfeld,:sommerfeld,:sommerfeld,:sommerfeld))
        sample_background2d!(s.coef, flat, zero(T), s.xg, s.yg)
        E0 = wave2d_energy(Φ, Π, s.coef; s.geom, s.ops, s.ws, s.metric)
        k = [similar(Φ) for _ in 1:8]; Φs = similar(Φ); Πs = similar(Π)
        dt = T(0.1) * s.dxmin; nst = ceil(Int, 6 / dt)
        t = zero(T); Emax = E0
        for _ in 1:nst
            _rk4_curv!(Φ, Π, t, dt, flat, s, bc; ε_KO=0.05, k, Φs, Πs); t += dt
            E = wave2d_energy(Φ, Π, s.coef; s.geom, s.ops, s.ws, s.metric)
            Emax = max(Emax, E)
        end
        Ef = wave2d_energy(Φ, Π, s.coef; s.geom, s.ops, s.ws, s.metric)
        @test isfinite(Ef)
        @test Emax ≤ E0 * (1 + 1e-6)     # no energy growth (stable + absorbing)
        @test Ef < E0                    # net absorption/dissipation
    end

    # ---- #5 variable background: gaugewave (varying α) convergence ----
    _progress("curvilinear: gaugewave (variable α) convergence")
    @testset "gaugewave convergence (cubed-square, curved Dirichlet)" begin
        bg, Φe, Πe, Dxe, Dye, _ =
            WaveToySecondOrder._background2d(:gaugewave, T; A=T(0.1), d=T(1),
                                             shift=(zero(T),zero(T)))
        errs = T[]
        for M in (2, 4)
            s = _curv_setup(T, N, M)
            Φ = Φe.(zero(T), s.xg, s.yg); Π = Πe.(zero(T), s.xg, s.yg)
            k = [similar(Φ) for _ in 1:8]; Φs = similar(Φ); Πs = similar(Π)
            dt = T(0.1) * s.dxmin
            bc(t) = make_bc2d((:dirichlet,:dirichlet,:dirichlet,:dirichlet);
                              gΠ = Πe.(t, s.xg, s.yg),
                              gDx = Dxe.(t, s.xg, s.yg), gDy = Dye.(t, s.xg, s.yg))
            rhs(a,b,c,d,t) = begin
                sample_background2d!(s.coef, bg, t, s.xg, s.yg)
                wave2d_curved_rhs!(a,b,c,d, s.coef; s.geom, s.ops, s.ws,
                                   ε_KO=0.0, bc2d=bc(t), s.metric)
            end
            t = zero(T)
            for _ in 1:ceil(Int, 0.3 / dt)
                rhs(k[1],k[2],Φ,Π,t);        @. Φs=Φ+dt/2*k[1]; @. Πs=Π+dt/2*k[2]
                rhs(k[3],k[4],Φs,Πs,t+dt/2); @. Φs=Φ+dt/2*k[3]; @. Πs=Π+dt/2*k[4]
                rhs(k[5],k[6],Φs,Πs,t+dt/2); @. Φs=Φ+dt*k[5];   @. Πs=Π+dt*k[6]
                rhs(k[7],k[8],Φs,Πs,t+dt)
                @. Φ += dt/6*(k[1]+2k[3]+2k[5]+k[7]); @. Π += dt/6*(k[2]+2k[4]+2k[6]+k[8])
                t += dt
            end
            push!(errs, sqrt(sum(@. (Φ - Φe(t, s.xg, s.yg))^2 * s.metric.Hd)))
        end
        @test all(isfinite, errs)
        @test errs[2] < errs[1]
        @test log2(errs[1] / errs[2]) > 1.5
    end
end

end
