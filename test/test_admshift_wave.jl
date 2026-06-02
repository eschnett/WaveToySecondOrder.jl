using HexSBPSAT: make_element, make_operators
using Random
using Test
using WaveToySecondOrder: make_periodic_domain_1d, wave1d_admshift_rhs!

# ADM-form scalar wave with constant shift β, periodic 1D mesh,
# consistent-D SBP-SAT + Kreiss-Oliger dissipation. The kernel's
# spatial operator is exactly skew in the H-norm (verified
# empirically: `H·D + (H·D)^T = 0` to roundoff), giving discrete
# H-norm energy conservation for any β. RK4 on a purely-imaginary
# spectrum is marginally stable; the observed growth on noise is
# roughly linear in time at ~0.25/crossing.
#
# State variables use CAPITAL Greek `Φ, Π` rather than the
# lowercase `φ, π` because Julia's `π = 3.14…` constant clashes
# with a matrix-valued local `π` and produces confusing errors.

# Inline RK4 step for the (Φ, Π) system.
function _rk4_step_adm!(Φ::AbstractMatrix{T}, Π::AbstractMatrix{T},
                          dt::T; dom, ops, β, ε_KO::Real = 0.0) where {T}
    Φ̇ = similar(Φ); Π̇ = similar(Π)

    wave1d_admshift_rhs!(Φ̇, Π̇, Φ, Π; dom, ops, β, ε_KO)
    k1Φ = copy(Φ̇); k1Π = copy(Π̇)

    Φ_t = Φ .+ (dt / 2) .* k1Φ
    Π_t = Π .+ (dt / 2) .* k1Π
    wave1d_admshift_rhs!(Φ̇, Π̇, Φ_t, Π_t; dom, ops, β, ε_KO)
    k2Φ = copy(Φ̇); k2Π = copy(Π̇)

    Φ_t = Φ .+ (dt / 2) .* k2Φ
    Π_t = Π .+ (dt / 2) .* k2Π
    wave1d_admshift_rhs!(Φ̇, Π̇, Φ_t, Π_t; dom, ops, β, ε_KO)
    k3Φ = copy(Φ̇); k3Π = copy(Π̇)

    Φ_t = Φ .+ dt .* k3Φ
    Π_t = Π .+ dt .* k3Π
    wave1d_admshift_rhs!(Φ̇, Π̇, Φ_t, Π_t; dom, ops, β, ε_KO)
    k4Φ = copy(Φ̇); k4Π = copy(Π̇)

    Φ .+= (dt / 6) .* (k1Φ .+ 2 .* k2Φ .+ 2 .* k3Φ .+ k4Φ)
    Π .+= (dt / 6) .* (k1Π .+ 2 .* k2Π .+ 2 .* k3Π .+ k4Π)
    return nothing
end

function _crossings_and_dt(dom, elem, β; n_xing, cfl = 0.1)
    T = typeof(dom.h)
    ξs = elem.xs
    dx_min = minimum(ξs[i+1] - ξs[i] for i in 1:(length(ξs) - 1)) * dom.h
    L = dom.x1 - dom.x0
    dt = T(cfl) * dx_min / (one(T) + abs(T(β)))
    t1 = T(n_xing) * L / (one(T) + abs(T(β)))
    n_steps = ceil(Int, t1 / dt)
    return dt, n_steps, t1
end

# Build a per-node β-field by sampling `f(x)` at every GLL node.
function _β_field(f, dom, elem, ::Type{T}) where {T}
    N = length(elem.xs); M = dom.M
    β = Matrix{T}(undef, N, M)
    for m in 1:M, i in 1:N
        x = T(dom.xs[m] + dom.h * elem.xs[i])
        β[i, m] = f(x)
    end
    return β
end

@testset "1D ADM-shift scalar wave (consistent-D + KO, periodic)" begin
    T = Float64; N = 4; M = 16
    elem = make_element(T, N); ops = make_operators(elem)
    dom  = make_periodic_domain_1d(T, M, 0.0, 1.0)

    # (a) β = 0 sanity — bounded.
    _progress("ADM-shift consistent-D: β = 0 sanity (50 crossings)")
    @testset "β = 0 sanity: bounded" begin
        Random.seed!(20260602)
        amp = sqrt(eps(T))
        Φ = amp .* randn(T, N, M)
        Π = amp .* randn(T, N, M)
        dt, n_steps, _ = _crossings_and_dt(dom, elem, 0.0; n_xing = 50)
        for _ in 1:n_steps
            _rk4_step_adm!(Φ, Π, dt; dom, ops, β = 0.0)
        end
        @test all(isfinite, Φ) && all(isfinite, Π)
        # `Π` naturally has larger magnitude than `Φ` on random-noise
        # IC because it is essentially a discrete time derivative:
        # noise on the GLL grid has characteristic length ~ dx_min,
        # so `Π = ∂_t Φ` scales by ~1/dx_min ≈ 60× over the IC amp
        # on this mesh.
        @test maximum(abs, Φ) < 100 * amp
        @test maximum(abs, Π) < 1000 * amp
    end

    # (b), (c) Robust stability for both β regimes.
    for (β, label) in ((0.5, "subluminal β = 0.5"),
                         (2.0, "superluminal β = 2.0"))
        _progress("ADM-shift consistent-D robust stability: $label, 50 crossings")
        @testset "$label: robust stability (50 crossings)" begin
            Random.seed!(20260602 + round(Int, 1000 * β))
            amp = sqrt(eps(T))
            Φ = amp .* randn(T, N, M)
            Π = amp .* randn(T, N, M)
            dt, n_steps, _ = _crossings_and_dt(dom, elem, β; n_xing = 50)
            for _ in 1:n_steps
                _rk4_step_adm!(Φ, Π, dt; dom, ops, β)
            end
            @test all(isfinite, Φ) && all(isfinite, Π)
            @test maximum(abs, Φ) < 100 * amp
            @test maximum(abs, Π) < 1000 * amp
        end
    end

    # (d) Extended-horizon catch-slow-modes test.
    _progress("ADM-shift consistent-D extended horizon: β = 2.0, 500 crossings")
    @testset "β = 2.0: extended horizon (500 crossings)" begin
        Random.seed!(20260602 + 2000)
        amp = sqrt(eps(T))
        Φ = amp .* randn(T, N, M)
        Π = amp .* randn(T, N, M)
        dt, n_steps, _ = _crossings_and_dt(dom, elem, 2.0; n_xing = 500)
        for _ in 1:n_steps
            _rk4_step_adm!(Φ, Π, dt; dom, ops, β = 2.0)
        end
        @test all(isfinite, Φ) && all(isfinite, Π)
        @test maximum(abs, Φ) < 1000 * amp
        @test maximum(abs, Π) < 10000 * amp
    end

    # ---- Variable β(x) tests ----
    β_sub  = _β_field(x -> T(0.3) + T(0.2) * sin(2π * x), dom, elem, T)
    β_sup  = _β_field(x -> T(2.0) + T(0.5) * sin(2π * x), dom, elem, T)
    β_sonic = _β_field(x -> T(0.5) + sin(2π * x),         dom, elem, T)

    # With variable β(x) the discrete energy identity picks up a
    # source `−½ ∫ β'(x) Π² dx`. For β > 1 everywhere this source
    # accumulates and the scheme exhibits slow growth that even
    # Kreiss-Oliger dissipation can only partially tame at M=16.
    # Subluminal variable β stays bounded (~50× over 50 crossings);
    # superluminal and sonic-horizon-crossing variable β are
    # LOOSELY bounded (~10⁷× at the same horizon) — flagged as a
    # known limitation pending the spectral analysis in
    # `bin/admshift_spectrum.jl`.

    _progress("Variable β subluminal: bounded growth (50 crossings)")
    @testset "Variable β = 0.3 + 0.2 sin(2π x): bounded" begin
        Random.seed!(20260603)
        amp = sqrt(eps(T))
        Φ = amp .* randn(T, N, M)
        Π = amp .* randn(T, N, M)
        max_β = T(0.5)
        dt, n_steps, _ = _crossings_and_dt(dom, elem, max_β; n_xing = 50)
        for _ in 1:n_steps
            _rk4_step_adm!(Φ, Π, dt; dom, ops, β = β_sub)
        end
        @test all(isfinite, Φ) && all(isfinite, Π)
        @test maximum(abs, Φ) < 200 * amp
    end

    _progress("Variable β superluminal: bounded growth, KO required")
    @testset "Variable β = 2.0 + 0.5 sin(2π x): bounded with KO" begin
        Random.seed!(20260603 + 1)
        amp = sqrt(eps(T))
        Φ = amp .* randn(T, N, M)
        Π = amp .* randn(T, N, M)
        max_β = T(2.5)
        dt, n_steps, _ = _crossings_and_dt(dom, elem, max_β; n_xing = 50,
                                             cfl = T(0.05))
        for _ in 1:n_steps
            _rk4_step_adm!(Φ, Π, dt; dom, ops, β = β_sup, ε_KO = T(1e-5))
        end
        @test all(isfinite, Φ) && all(isfinite, Π)
        @test maximum(abs, Φ) < T(1e8) * amp
    end

    _progress("Variable β sonic-horizon: bounded growth, KO required")
    @testset "Variable β = 0.5 + sin(2π x), crosses 1: bounded with KO" begin
        Random.seed!(20260603 + 2)
        amp = sqrt(eps(T))
        Φ = amp .* randn(T, N, M)
        Π = amp .* randn(T, N, M)
        max_β = T(1.5)
        dt, n_steps, _ = _crossings_and_dt(dom, elem, max_β; n_xing = 50,
                                             cfl = T(0.05))
        for _ in 1:n_steps
            _rk4_step_adm!(Φ, Π, dt; dom, ops, β = β_sonic, ε_KO = T(1e-5))
        end
        @test all(isfinite, Φ) && all(isfinite, Π)
        @test maximum(abs, Φ) < T(1e9) * amp
    end

    # (e) Plane-wave convergence — smooth IC, one transit period,
    # max-error vs analytic decays under M-refinement.
    _progress("ADM-shift consistent-D plane-wave convergence (β = 0.5)")
    @testset "Plane-wave convergence under M-refinement (β = 0.5)" begin
        β = 0.5
        # Right-mover Φ = sin(k(x - c+ t)) with c+ = 1 - β = 0.5
        # (sign per ADM-shift convention ∂_t Φ = Π + β ∂_x Φ).
        # Then Π = ∂_t Φ - β ∂_x Φ = -c+ · k · cos - β · k · cos =
        #          -(c+ + β) k cos = -k cos.
        k_w = T(2π)
        c_plus = one(T) - T(β)         # ≈ 0.5
        t_final = one(T) / c_plus       # one transit period
        errs = T[]
        for M_test in (8, 16, 32)
            dom_test = make_periodic_domain_1d(T, M_test, 0.0, 1.0)
            ξs = elem.xs
            xgrid = T[T(dom_test.xs[m] + dom_test.h * ξs[i])
                       for i in 1:N, m in 1:M_test]
            Φ = T[sin(k_w * x) for x in xgrid]
            Π = T[-k_w * cos(k_w * x) for x in xgrid]
            dt, n_steps, _ = _crossings_and_dt(dom_test, elem, β;
                                                 n_xing = 1)
            n_steps = ceil(Int, t_final / dt)
            for _ in 1:n_steps
                _rk4_step_adm!(Φ, Π, dt; dom = dom_test, ops, β)
            end
            t_actual = n_steps * dt
            Φ_exact = T[sin(k_w * (x - c_plus * t_actual)) for x in xgrid]
            push!(errs, maximum(abs, Φ .- Φ_exact))
        end
        @test all(isfinite, errs)
        @test all(errs .> 0)
        @test errs[end] < errs[1]
        gmean = (errs[1] / errs[end])^(1 / (length(errs) - 1))
        @test gmean > 2.5
    end
end
