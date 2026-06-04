@testitem "wave2d" tags=[:gpu] begin
    _progress(m) = (printstyled(stderr, "  • ", m, "\n"; color = :cyan); flush(stderr))

# Conservative-form 2D scalar wave on a 2+1 ADM background
# (`wave2d_curved_rhs!` in src/wave2d_curved.jl), on axis-aligned
# affine meshes (make_uniform_quad). Mirrors test_wave1d.jl.
#
# Testsets: spectrum (max Re(λ) ≤ round-off for flat + constant
# shift), plane-wave convergence, energy conservation, noise
# robustness, gauge-wave (varying lapse via SpacetimeMetrics).

using HexMeshes: make_uniform_quad, make_cubed_square_mesh
using HexSBPSAT: make_element, make_operators, make_geometry, to_device,
                 make_metric_terms2d
using KernelAbstractions
using LinearAlgebra
using MultiFloats
using Random
using SpacetimeMetrics: GaugeWave
using Test
using WaveToySecondOrder: AnalyticBackground2D, MetricBackground2D,
                          make_coef2d, sample_background2d!,
                          make_wave2d_workspace, wave2d_curved_rhs!,
                          wave2d_energy, make_bc2d


function _setup2d(::Type{T}, N, M; periodic = true) where {T}
    mesh = make_uniform_quad(T, M, M, zero(T), one(T); periodic)
    elem = make_element(T, N); ops = make_operators(elem)
    geom = make_geometry(mesh, elem)
    ws   = make_wave2d_workspace(geom, ops)
    coef = make_coef2d(geom)
    xg = geom.coords[1, :, :, :]; yg = geom.coords[2, :, :, :]
    return (; mesh, elem, ops, geom, ws, coef, xg, yg)
end

# RK4 step with stage-time background sampling.
function _rk4_2d!(Φ, Π, t, dt, bg, s; ε_KO, k, Φs, Πs)
    (; geom, ops, ws, coef, xg, yg) = s
    sample_background2d!(coef, bg, t, xg, yg)
    wave2d_curved_rhs!(k[1], k[2], Φ, Π, coef; geom, ops, ws, ε_KO)
    sample_background2d!(coef, bg, t + dt/2, xg, yg)
    @. Φs = Φ + dt/2*k[1]; @. Πs = Π + dt/2*k[2]
    wave2d_curved_rhs!(k[3], k[4], Φs, Πs, coef; geom, ops, ws, ε_KO)
    @. Φs = Φ + dt/2*k[3]; @. Πs = Π + dt/2*k[4]
    wave2d_curved_rhs!(k[5], k[6], Φs, Πs, coef; geom, ops, ws, ε_KO)
    sample_background2d!(coef, bg, t + dt, xg, yg)
    @. Φs = Φ + dt*k[5]; @. Πs = Π + dt*k[6]
    wave2d_curved_rhs!(k[7], k[8], Φs, Πs, coef; geom, ops, ws, ε_KO)
    @. Φ += dt/6*(k[1] + 2k[3] + 2k[5] + k[7])
    @. Π += dt/6*(k[2] + 2k[4] + 2k[6] + k[8])
    return nothing
end

_flat2d(::Type{T}) where {T} =
    AnalyticBackground2D((t,x,y) -> one(T), (t,x,y) -> (zero(T), zero(T)),
                         (t,x,y) -> (one(T), zero(T), one(T)))

@testset "2D ADM scalar-wave kernel (wave2d_curved_rhs!)" begin
    T = Float64; N = 4

    _progress("wave2d: spectrum max Re(λ)")
    @testset "spectrum: max Re(λ) ≤ round-off" begin
        M = 3
        s = _setup2d(T, N, M)
        nn = N*N*s.geom.Ne; n = 2nn
        Φ = zeros(T,N,N,s.geom.Ne); Π = similar(Φ); Φ̇ = similar(Φ); Π̇ = similar(Φ)
        for (label, bg) in (
                ("flat",          _flat2d(T)),
                ("shift (0.3,0.2)", AnalyticBackground2D((t,x,y)->one(T),
                     (t,x,y)->(T(0.3),T(0.2)), (t,x,y)->(one(T),zero(T),one(T)))),
                ("aniso metric",  AnalyticBackground2D((t,x,y)->one(T),
                     (t,x,y)->(zero(T),zero(T)),
                     (t,x,y)->(one(T)+T(0.3)*sinpi(2x), zero(T), one(T)+T(0.2)*sinpi(2y)))))
            sample_background2d!(s.coef, bg, zero(T), s.xg, s.yg)
            A = zeros(T, n, n)
            for j in 1:n
                fill!(Φ,0); fill!(Π,0); j ≤ nn ? (Φ[j]=1) : (Π[j-nn]=1)
                wave2d_curved_rhs!(Φ̇, Π̇, Φ, Π, s.coef; s.geom, s.ops, s.ws, ε_KO=0.0)
                A[1:nn,j] = vec(Φ̇); A[nn+1:end,j] = vec(Π̇)
            end
            λ = eigvals(A)
            @test maximum(real, λ) ≤ 1e-5 * maximum(abs, λ)
        end
    end

    _progress("wave2d: plane-wave convergence")
    @testset "diagonal plane-wave convergence (periodic)" begin
        ω = 2π*sqrt(2)
        Φe(t,x,y) = sin(2π*(x+y) - ω*t)
        Πe(t,x,y) = -ω*cos(2π*(x+y) - ω*t)
        bg = _flat2d(T)
        errs = T[]
        for M in (4, 8, 16)
            s = _setup2d(T, N, M)
            Φ = Φe.(zero(T), s.xg, s.yg); Π = Πe.(zero(T), s.xg, s.yg)
            k = [similar(Φ) for _ in 1:8]; Φs = similar(Φ); Πs = similar(Π)
            h = one(T)/M; dxmin = minimum(diff(s.elem.xs))*h; dt = T(0.1)*dxmin/sqrt(2)
            t = zero(T)
            for _ in 1:ceil(Int, 0.5/dt)
                _rk4_2d!(Φ, Π, t, dt, bg, s; ε_KO=0.0, k, Φs, Πs); t += dt
            end
            push!(errs, maximum(abs.(Φ .- Φe.(t, s.xg, s.yg))))
        end
        @test all(isfinite, errs)
        @test (errs[1]/errs[end])^(1/(length(errs)-1)) > 2.5
    end

    _progress("wave2d: energy conservation (flat)")
    @testset "energy drift < 1e-3 (flat periodic)" begin
        M = 12
        s = _setup2d(T, N, M)
        ω = 2π*sqrt(2)
        Φ = sin.(2π .* (s.xg .+ s.yg)); Π = -ω .* cos.(2π .* (s.xg .+ s.yg))
        bg = _flat2d(T)
        sample_background2d!(s.coef, bg, zero(T), s.xg, s.yg)
        E0 = wave2d_energy(Φ, Π, s.coef; s.geom, s.ops, s.ws)
        k = [similar(Φ) for _ in 1:8]; Φs = similar(Φ); Πs = similar(Π)
        h = one(T)/M; dt = T(0.1)*minimum(diff(s.elem.xs))*h/sqrt(2)
        t = zero(T)
        for _ in 1:ceil(Int, 1.0/dt)
            _rk4_2d!(Φ, Π, t, dt, bg, s; ε_KO=0.0, k, Φs, Πs); t += dt
        end
        E1 = wave2d_energy(Φ, Π, s.coef; s.geom, s.ops, s.ws)
        @test abs(E1/E0 - 1) < 1e-3
    end

    _progress("wave2d: noise robustness")
    @testset "noise bounded (flat + shift, 20 crossings)" begin
        for (label, bg, maxβ) in (("flat", _flat2d(T), 0.0),
                ("shift", AnalyticBackground2D((t,x,y)->one(T),
                     (t,x,y)->(T(0.4),T(0.3)), (t,x,y)->(one(T),zero(T),one(T))), 0.7))
            M = 8
            s = _setup2d(T, N, M)
            Random.seed!(20260604)
            amp = sqrt(eps(T))
            Φ = amp .* randn(T,N,N,s.geom.Ne); Π = amp .* randn(T,N,N,s.geom.Ne)
            k = [similar(Φ) for _ in 1:8]; Φs = similar(Φ); Πs = similar(Π)
            h = one(T)/M; dxmin = minimum(diff(s.elem.xs))*h
            dt = T(0.1)*dxmin/(1+maxβ); nst = ceil(Int, 20/(1+maxβ)/dt)
            t = zero(T)
            for _ in 1:nst
                _rk4_2d!(Φ, Π, t, dt, bg, s; ε_KO=0.1, k, Φs, Πs); t += dt
            end
            @test all(isfinite, Φ) && all(isfinite, Π)
            @test maximum(abs, Φ) < 1000*amp
        end
    end

    _progress("wave2d: gauge wave (varying lapse via SpacetimeMetrics)")
    @testset "gauge-wave background convergence (MetricBackground2D)" begin
        # AwA gauge wave propagating in x: α=√H, β=0, γ=diag(H,1),
        # H = 1 − A sin(2π(x−t)). Exact Φ = sin(k₀(x̂−t̂)),
        # x̂−t̂ = x − t + 2C cos(2π(x−t)), C = A/(4π) (d=1).
        A = T(0.1); k = 2T(π); k₀ = 2T(π); C = A/(4T(π))
        bg = MetricBackground2D(GaugeWave(A, one(T)))
        ψ(t,x) = x - t + 2C*cos(k*(x-t))
        Φe(t,x,y) = sin(k₀*ψ(t,x))
        Πe(t,x,y) = -k₀*(1 - A*sin(k*(x-t)))*cos(k₀*ψ(t,x))
        errs = T[]
        for M in (4, 8, 16)
            s = _setup2d(T, N, M)
            Φ = Φe.(zero(T), s.xg, s.yg); Π = Πe.(zero(T), s.xg, s.yg)
            k8 = [similar(Φ) for _ in 1:8]; Φs = similar(Φ); Πs = similar(Π)
            h = one(T)/M; dt = T(0.1)*minimum(diff(s.elem.xs))*h
            t = zero(T)
            for _ in 1:ceil(Int, 0.5/dt)
                _rk4_2d!(Φ, Π, t, dt, bg, s; ε_KO=0.0, k=k8, Φs, Πs); t += dt
            end
            push!(errs, maximum(abs.(Φ .- Φe.(t, s.xg, s.yg))))
        end
        @test all(isfinite, errs)
        @test (errs[1]/errs[end])^(1/(length(errs)-1)) > 2.5
    end

    # Curvilinear mesh (cubed-square): free-stream-preserving
    # conservative operator + physical-normal Sommerfeld outer BC.
    _progress("wave2d: curvilinear cubed-square (spectrum, free-stream, energy)")
    @testset "cubed-square: stability, free-stream, energy decay" begin
        flatbg(::Type{S}) where {S} =
            AnalyticBackground2D((t,x,y)->one(S), (t,x,y)->(zero(S),zero(S)),
                                 (t,x,y)->(one(S),zero(S),one(S)))
        sommerfeld4 = make_bc2d((:sommerfeld,:sommerfeld,:sommerfeld,:sommerfeld))

        # Spectrum: stable (max Re(λ) ≤ round-off) with the outer
        # Sommerfeld BC — the closed domain is unstable without it.
        # M=2 only (Ne≈12, n=2·N²·Ne≈384) keeps the dense eigensolve
        # fast; free-stream/energy below run at M=4.
        let M = 2
            mesh = make_cubed_square_mesh(T, M, T(0.3))
            elem = make_element(T, N); ops = make_operators(elem)
            geom = make_geometry(mesh, elem); metric = make_metric_terms2d(geom, ops)
            ws = make_wave2d_workspace(geom, ops); coef = make_coef2d(geom)
            Ne = geom.Ne
            sample_background2d!(coef, flatbg(T), zero(T),
                                 geom.coords[1,:,:,:], geom.coords[2,:,:,:])
            nn = N*N*Ne; n = 2nn
            Φ = zeros(T,N,N,Ne); Π = similar(Φ); Φ̇ = similar(Φ); Π̇ = similar(Φ)
            for ε in (0.0, 0.1)
                A = zeros(T, n, n)
                for j in 1:n
                    fill!(Φ,0); fill!(Π,0); j ≤ nn ? (Φ[j]=1) : (Π[j-nn]=1)
                    wave2d_curved_rhs!(Φ̇, Π̇, Φ, Π, coef; geom, ops, ws,
                                       ε_KO=ε, bc2d=sommerfeld4, metric)
                    A[1:nn,j]=vec(Φ̇); A[nn+1:end,j]=vec(Π̇)
                end
                λ = eigvals(A)
                @test maximum(real, λ) ≤ 1e-5 * maximum(abs, λ)
            end
        end

        # Full-RHS free-stream: a constant (Φ, Π=0) state is a fixed
        # point (∂_tΦ = ∂_tΠ = 0) on the curved mesh.
        mesh = make_cubed_square_mesh(T, 4, T(0.3))
        elem = make_element(T, N); ops = make_operators(elem)
        geom = make_geometry(mesh, elem); metric = make_metric_terms2d(geom, ops)
        ws = make_wave2d_workspace(geom, ops); coef = make_coef2d(geom)
        Ne = geom.Ne
        sample_background2d!(coef, flatbg(T), zero(T),
                             geom.coords[1,:,:,:], geom.coords[2,:,:,:])
        Φ̇ = zeros(T,N,N,Ne); Π̇ = similar(Φ̇)
        wave2d_curved_rhs!(Φ̇, Π̇, fill(T(3), N,N,Ne), zeros(T,N,N,Ne), coef;
                           geom, ops, ws, ε_KO=0.1, bc2d=sommerfeld4, metric)
        @test maximum(abs, Φ̇) ≤ 1e-10
        @test maximum(abs, Π̇) ≤ 1e-9

        # Noise + Sommerfeld: bounded, energy non-increasing (absorbed).
        Random.seed!(20260604)
        Φ = randn(T,N,N,Ne); Π = randn(T,N,N,Ne)
        Eof(Φ,Π) = wave2d_energy(Φ,Π,coef; geom, ops, ws, metric)
        E0 = Eof(Φ,Π)
        k = [similar(Φ) for _ in 1:8]; Φs = similar(Φ); Πs = similar(Π)
        h = 2*T(0.3)/4; dt = T(0.05)*minimum(diff(elem.xs))*h
        rhs(a,b,c,d) = wave2d_curved_rhs!(a,b,c,d,coef; geom, ops, ws,
                                          ε_KO=0.1, bc2d=sommerfeld4, metric)
        for _ in 1:200
            rhs(k[1],k[2],Φ,Π); @. Φs=Φ+dt/2*k[1]; @. Πs=Π+dt/2*k[2]
            rhs(k[3],k[4],Φs,Πs); @. Φs=Φ+dt/2*k[3]; @. Πs=Π+dt/2*k[4]
            rhs(k[5],k[6],Φs,Πs); @. Φs=Φ+dt*k[5]; @. Πs=Π+dt*k[6]
            rhs(k[7],k[8],Φs,Πs)
            @. Φ+=dt/6*(k[1]+2k[3]+2k[5]+k[7]); @. Π+=dt/6*(k[2]+2k[4]+2k[6]+k[8])
        end
        @test all(isfinite, Φ) && all(isfinite, Π)
        @test Eof(Φ,Π) ≤ E0
    end

    # Curvilinear convergence vs an analytic solution: a flat-space
    # plane wave on the cubed-square with a curved Dirichlet boundary
    # that injects the exact field-radiation data (Π, ∇Φ). At β=0 the
    # field-radiation Dirichlet is exact, so the interior tracks the
    # analytic solution; the rate is ~2 (capped by the one-sided
    # boundary). This is the GOALS-required analytic convergence test
    # for the curvilinear path.
    _progress("wave2d: curvilinear convergence vs analytic (Dirichlet)")
    @testset "cubed-square plane-wave convergence (curved Dirichlet)" begin
        κ = T(2); ω = κ * sqrt(T(2))
        Φe(t,x,y) =  sin(κ*(x+y) - ω*t)
        Πe(t,x,y) = -ω*cos(κ*(x+y) - ω*t)
        De(t,x,y) =  κ*cos(κ*(x+y) - ω*t)        # ∂_xΦ = ∂_yΦ
        bg = AnalyticBackground2D((t,x,y)->one(T), (t,x,y)->(zero(T),zero(T)),
                                  (t,x,y)->(one(T),zero(T),one(T)))
        errs = T[]
        for M in (2, 4, 8)
            mesh = make_cubed_square_mesh(T, M, T(0.3))
            elem = make_element(T, N); ops = make_operators(elem)
            geom = make_geometry(mesh, elem); metric = make_metric_terms2d(geom, ops)
            ws = make_wave2d_workspace(geom, ops); coef = make_coef2d(geom)
            xg = geom.coords[1,:,:,:]; yg = geom.coords[2,:,:,:]
            Φ = Φe.(zero(T), xg, yg); Π = Πe.(zero(T), xg, yg)
            k = [similar(Φ) for _ in 1:8]; Φs = similar(Φ); Πs = similar(Π)
            h = 2*T(0.3)/M; dt = T(0.1)*minimum(diff(elem.xs))*h
            rhs(a,b,c,d,t) = begin
                sample_background2d!(coef, bg, t, xg, yg)
                bc = make_bc2d((:dirichlet,:dirichlet,:dirichlet,:dirichlet);
                               gΠ = Πe.(t,xg,yg), gDx = De.(t,xg,yg), gDy = De.(t,xg,yg))
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
        @test errs[end] < errs[1]
        @test (errs[1]/errs[end])^(1/(length(errs)-1)) > 1.8
    end

    _progress("wave2d: Float64x2 (MultiFloats) CPU")
    @testset "Float64x2 agrees with Float64 (plane wave)" begin
        M = 4
        results = Dict{DataType, Array{Float64,3}}()
        for T2 in (Float64, Float64x2)
            s = _setup2d(T2, N, M)
            xg64 = Float64.(s.xg)
            Φ = T2.(sin.(2 .* Float64(π) .* xg64))
            Π = T2.(zeros(Float64, size(xg64)))
            bg = _flat2d(T2)
            k = [similar(Φ) for _ in 1:8]; Φs = similar(Φ); Πs = similar(Π)
            dt = T2(1) / 256
            t = zero(T2)
            for _ in 1:48
                _rk4_2d!(Φ, Π, t, dt, bg, s; ε_KO = T2(0.1), k, Φs, Πs)
                t += dt
            end
            @test all(isfinite, Φ)
            results[T2] = Float64.(Φ)
        end
        @test maximum(abs, results[Float64] .- results[Float64x2]) < 1e-11
    end
end

# GPU smoke test: full 2D RHS + background sampling on Metal (Float32)
# vs CPU Float32. Auto-skips without Metal.
if !@isdefined(HAS_METAL)
    const HAS_METAL = try
        @eval using Metal
        Metal.functional()
    catch
        false
    end
end

if HAS_METAL
    @testset "wave2d on Metal (Float32)" begin
        _progress("wave2d Metal smoke test (Float32)")
        T = Float32; N = 4; M = 8
        # isbits closures (no Type capture) so the bg passes to the kernel.
        bg = AnalyticBackground2D((t,x,y) -> 1.0f0,
                                  (t,x,y) -> (0.3f0, 0.2f0),
                                  (t,x,y) -> (1.0f0, 0.0f0, 1.0f0))
        mesh = make_uniform_quad(T, M, M, 0.0f0, 1.0f0; periodic = true)
        elem = make_element(T, N); ops = make_operators(elem)
        geom_h = make_geometry(mesh, elem)
        xg_h = geom_h.coords[1, :, :, :]; yg_h = geom_h.coords[2, :, :, :]
        Φ0 = sinpi.(2 .* xg_h); Π0 = -2 .* T(π) .* cospi.(2 .* xg_h)
        dt = T(5.0f-4); nst = 40

        run_on = function (backend, geom, xg, yg)
            ws = make_wave2d_workspace(geom, ops)
            coef = make_coef2d(geom)
            Φ = KernelAbstractions.allocate(backend, T, N, N, geom.Ne)
            Π = similar(Φ); copyto!(Φ, Φ0); copyto!(Π, Π0)
            k = [similar(Φ) for _ in 1:8]; Φs = similar(Φ); Πs = similar(Π)
            t = zero(T)
            for _ in 1:nst
                sample_background2d!(coef, bg, t, xg, yg)
                wave2d_curved_rhs!(k[1], k[2], Φ, Π, coef; geom, ops, ws, ε_KO=1f-4)
                @. Φs = Φ + dt/2*k[1]; @. Πs = Π + dt/2*k[2]
                sample_background2d!(coef, bg, t+dt/2, xg, yg)
                wave2d_curved_rhs!(k[3], k[4], Φs, Πs, coef; geom, ops, ws, ε_KO=1f-4)
                @. Φs = Φ + dt/2*k[3]; @. Πs = Π + dt/2*k[4]
                wave2d_curved_rhs!(k[5], k[6], Φs, Πs, coef; geom, ops, ws, ε_KO=1f-4)
                @. Φs = Φ + dt*k[5]; @. Πs = Π + dt*k[6]
                sample_background2d!(coef, bg, t+dt, xg, yg)
                wave2d_curved_rhs!(k[7], k[8], Φs, Πs, coef; geom, ops, ws, ε_KO=1f-4)
                @. Φ += dt/6*(k[1]+2k[3]+2k[5]+k[7]); @. Π += dt/6*(k[2]+2k[4]+2k[6]+k[8])
                t += dt
            end
            return Array(Φ)
        end

        Φc = run_on(KernelAbstractions.CPU(), geom_h, xg_h, yg_h)
        backend = MetalBackend()
        geom_d = to_device(geom_h, backend)
        xg_d = KernelAbstractions.allocate(backend, T, N, N, geom_h.Ne)
        yg_d = similar(xg_d); copyto!(xg_d, xg_h); copyto!(yg_d, yg_h)
        Φg = run_on(backend, geom_d, xg_d, yg_d)
        @test all(isfinite, Φg)
        @test maximum(abs, Φg .- Φc) ≤ 1e-3 * max(1, maximum(abs, Φc))
    end
end

end
