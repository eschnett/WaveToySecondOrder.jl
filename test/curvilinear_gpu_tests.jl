@testitem "curvilinear_gpu" tags=[:gpu] begin
    _progress(m) = (printstyled(stderr, "  • ", m, "\n"; color = :cyan); flush(stderr))

# GPU smoke test for the CURVILINEAR 2D path on Metal (Float32) vs CPU
# Float32: a short curved-Sommerfeld RK4 evolution on a cubed-square
# mesh, run entirely on-device. This exercises the on-device
# free-stream-preserving gradient/divergence (HexSBPSAT
# kernels2d_curv.jl) AND the on-device curvilinear boundary pass
# (_apply_bc2d_curv! GPU kernel). Auto-skips without Metal. The
# operator-only CPU↔GPU agreement is gated separately in
# HexSBPSAT/test/test_curvilinear2d.jl.

using HexMeshes: make_cubed_square_mesh
using HexSBPSAT: make_element, make_operators, make_geometry, to_device,
                 make_metric_terms2d
using KernelAbstractions
using Test
using WaveToySecondOrder: AnalyticBackground2D, make_coef2d,
                          sample_background2d!, make_wave2d_workspace,
                          wave2d_curved_rhs!, make_bc2d

if !@isdefined(HAS_METAL)
    const HAS_METAL = try
        @eval using Metal
        Metal.functional()
    catch
        false
    end
end

if HAS_METAL
    @testset "wave2d curvilinear on Metal (Float32)" begin
        _progress("curvilinear Metal smoke test (Float32)")
        T = Float32; N = 4; M = 4
        bg = AnalyticBackground2D((t,x,y) -> 1.0f0,
                                  (t,x,y) -> (0.0f0, 0.0f0),
                                  (t,x,y) -> (1.0f0, 0.0f0, 1.0f0))
        mesh = make_cubed_square_mesh(T, M, T(0.3))
        elem = make_element(T, N); ops = make_operators(elem)
        geom_h = make_geometry(mesh, elem)
        metric_h = make_metric_terms2d(geom_h, ops); Ne = geom_h.Ne
        xg_h = geom_h.coords[1, :, :, :]; yg_h = geom_h.coords[2, :, :, :]
        Φ0 = exp.(-(xg_h.^2 .+ yg_h.^2) ./ (2 * 0.15f0^2)); Π0 = zeros(T, N, N, Ne)
        dt = T(1) / 512; nst = 30

        mk(b, a) = (d = KernelAbstractions.allocate(b, T, size(a)); copyto!(d, a); d)

        run_on = function (backend, geom, metric, xg, yg)
            ws = make_wave2d_workspace(geom, ops); coef = make_coef2d(geom)
            Φ = mk(backend, Φ0); Π = mk(backend, Π0)
            k = [similar(Φ) for _ in 1:8]; Φs = similar(Φ); Πs = similar(Π)
            t = zero(T)
            rhs(a1, a2, p1, p2, tt) = begin
                sample_background2d!(coef, bg, tt, xg, yg)
                bc = make_bc2d((:sommerfeld, :sommerfeld, :sommerfeld, :sommerfeld))
                wave2d_curved_rhs!(a1, a2, p1, p2, coef; geom, ops, ws,
                                   ε_KO = T(0.05), bc2d = bc, metric)
            end
            for _ in 1:nst
                rhs(k[1],k[2],Φ,Π,t);        @. Φs=Φ+dt/2*k[1]; @. Πs=Π+dt/2*k[2]
                rhs(k[3],k[4],Φs,Πs,t+dt/2); @. Φs=Φ+dt/2*k[3]; @. Πs=Π+dt/2*k[4]
                rhs(k[5],k[6],Φs,Πs,t+dt/2); @. Φs=Φ+dt*k[5];   @. Πs=Π+dt*k[6]
                rhs(k[7],k[8],Φs,Πs,t+dt)
                @. Φ += dt/6*(k[1]+2k[3]+2k[5]+k[7]); @. Π += dt/6*(k[2]+2k[4]+2k[6]+k[8])
                t += dt
            end
            return Array(Φ)
        end

        Φc = run_on(KernelAbstractions.CPU(), geom_h, metric_h, xg_h, yg_h)
        backend = MetalBackend()
        geom_d = to_device(geom_h, backend)
        metric_d = (ax1 = mk(backend, metric_h.ax1), ax2 = mk(backend, metric_h.ax2),
                    ay1 = mk(backend, metric_h.ay1), ay2 = mk(backend, metric_h.ay2),
                    invdetJ = mk(backend, metric_h.invdetJ),
                    Hd = mk(backend, metric_h.Hd))
        xg_d = mk(backend, xg_h); yg_d = mk(backend, yg_h)
        Φg = run_on(backend, geom_d, metric_d, xg_d, yg_d)
        @test all(isfinite, Φg)
        @test maximum(abs, Φg .- Φc) ≤ 1e-3 * max(1, maximum(abs, Φc))
    end
end

end
