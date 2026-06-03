using HexSBPSAT: make_element, make_operators
using LinearAlgebra
using Random
using Test
using WaveToySecondOrder: make_periodic_domain_1d,
                           wave1d_curved1d_rhs!

# Conservative-form 1D scalar-wave kernel `wave1d_curved1d_rhs!`.
# Evolves densitised Π := √γ_xx (∂_t Φ − β ∂_x Φ) and supports
# space- and time-varying β(t,x), γ_xx(t,x) (with α = 1). The flux-
# divergence form of ∂_t Π includes both `β ∂_x Π` advection and the
# `(∂_x β) Π` source automatically.
#
# Two testsets:
#   (a) Noise robust stability: constant β ∈ {0, 0.5, 2.0} and
#       variable β (subluminal / superluminal / sonic), 50 light-
#       crossings, √eps noise IC.
#   (b) Plane-wave convergence (β=0.5, γ=1) + gauge-wave convergence
#       (β, γ varying in t and x) with inertial-energy drift check.

function _rk4_step_curved!(Φ::AbstractMatrix{T}, Π::AbstractMatrix{T},
                            t::T, dt::T;
                            dom, ops, bg!, ε_KO::Real = 0.1) where {T}
    N, M = size(Φ)
    Φ̇ = similar(Φ); Π̇ = similar(Π)
    β     = Matrix{T}(undef, N, M)
    sγ    = Matrix{T}(undef, N, M)
    isγ   = Matrix{T}(undef, N, M)

    bg!(β, sγ, isγ, t)
    wave1d_curved1d_rhs!(Φ̇, Π̇, Φ, Π;
                          dom, ops, β, sqrtγ = sγ, inv_sqrtγ = isγ, ε_KO)
    k1Φ = copy(Φ̇); k1Π = copy(Π̇)

    bg!(β, sγ, isγ, t + dt/2)
    wave1d_curved1d_rhs!(Φ̇, Π̇, Φ .+ (dt/2) .* k1Φ, Π .+ (dt/2) .* k1Π;
                          dom, ops, β, sqrtγ = sγ, inv_sqrtγ = isγ, ε_KO)
    k2Φ = copy(Φ̇); k2Π = copy(Π̇)

    wave1d_curved1d_rhs!(Φ̇, Π̇, Φ .+ (dt/2) .* k2Φ, Π .+ (dt/2) .* k2Π;
                          dom, ops, β, sqrtγ = sγ, inv_sqrtγ = isγ, ε_KO)
    k3Φ = copy(Φ̇); k3Π = copy(Π̇)

    bg!(β, sγ, isγ, t + dt)
    wave1d_curved1d_rhs!(Φ̇, Π̇, Φ .+ dt .* k3Φ, Π .+ dt .* k3Π;
                          dom, ops, β, sqrtγ = sγ, inv_sqrtγ = isγ, ε_KO)
    k4Φ = copy(Φ̇); k4Π = copy(Π̇)

    Φ .+= (dt/6) .* (k1Φ .+ 2 .* k2Φ .+ 2 .* k3Φ .+ k4Φ)
    Π .+= (dt/6) .* (k1Π .+ 2 .* k2Π .+ 2 .* k3Π .+ k4Π)
    return nothing
end

# Sample f(t, x) at every GLL node into the supplied (N, M) buffer.
function _fill_field!(buf::AbstractMatrix{T}, f, t::T, dom, elem) where {T}
    N = length(elem.xs); M = dom.M
    @inbounds for m in 1:M, i in 1:N
        x = T(dom.xs[m] + dom.h * elem.xs[i])
        buf[i, m] = f(t, x)
    end
    return buf
end

# Build a closure capturing dom, elem that fills the three background
# matrices at time `t` using analytic functions β_fn(t,x),
# sqrtγ_fn(t,x).
function _make_bg(β_fn, sqrtγ_fn, dom, elem)
    return (β_buf, sγ_buf, isγ_buf, t) -> begin
        _fill_field!(β_buf,  β_fn,     t, dom, elem)
        _fill_field!(sγ_buf, sqrtγ_fn, t, dom, elem)
        @inbounds for idx in eachindex(sγ_buf)
            isγ_buf[idx] = one(eltype(sγ_buf)) / sγ_buf[idx]
        end
        return nothing
    end
end

function _crossings_and_dt_curved(dom, elem, max_β; n_xing, cfl = 0.1)
    T = typeof(dom.h)
    ξs = elem.xs
    dx_min = minimum(ξs[i+1] - ξs[i] for i in 1:(length(ξs) - 1)) * dom.h
    L = dom.x1 - dom.x0
    dt = T(cfl) * dx_min / (one(T) + abs(T(max_β)))
    t1 = T(n_xing) * L / (one(T) + abs(T(max_β)))
    n_steps = ceil(Int, t1 / dt)
    return dt, n_steps
end

@testset "1D conservative scalar-wave kernel (wave1d_curved1d_rhs!)" begin
    T = Float64; N = 4
    elem = make_element(T, N); ops = make_operators(elem)

    # (a) Noise robust stability — six setups.
    # KO is off for constant / subluminal-variable β because the KO
    # D⁶ term has its own (tighter) CFL; the wave operator alone is
    # already stable in these regimes. The variable superluminal and
    # sonic-horizon cases need a small ε_KO to tame the nonlinear
    # source from variable β.
    _progress("curved1d noise: constant β, 50 crossings")
    for (β_val, label) in ((0.0, "β = 0"),
                             (0.5, "subluminal β = 0.5"),
                             (2.0, "superluminal β = 2.0"))
        @testset "Constant $label: bounded (50 crossings)" begin
            M = 16
            dom = make_periodic_domain_1d(T, M, 0.0, 1.0)
            bg! = _make_bg((t,x) -> T(β_val), (t,x) -> one(T), dom, elem)
            Random.seed!(20260603 + round(Int, 1000*β_val))
            amp = sqrt(eps(T))
            Φ = amp .* randn(T, N, M)
            Π = amp .* randn(T, N, M)
            dt, n_steps = _crossings_and_dt_curved(dom, elem, β_val;
                                                    n_xing = 50)
            t = zero(T)
            for _ in 1:n_steps
                _rk4_step_curved!(Φ, Π, t, dt;
                                   dom, ops, bg!, ε_KO = 0.0)
                t += dt
            end
            @test all(isfinite, Φ) && all(isfinite, Π)
            @test maximum(abs, Φ) < 100 * amp
            @test maximum(abs, Π) < 1000 * amp
        end
    end

    _progress("curved1d noise: variable β subluminal")
    @testset "Variable β = 0.3 + 0.2 sin(2π x): bounded" begin
        M = 16
        dom = make_periodic_domain_1d(T, M, 0.0, 1.0)
        bg! = _make_bg((t,x) -> T(0.3) + T(0.2)*sin(2π*x),
                        (t,x) -> one(T), dom, elem)
        Random.seed!(20260604)
        amp = sqrt(eps(T))
        Φ = amp .* randn(T, N, M); Π = amp .* randn(T, N, M)
        max_β = T(0.5)
        dt, n_steps = _crossings_and_dt_curved(dom, elem, max_β;
                                                n_xing = 50)
        t = zero(T)
        for _ in 1:n_steps
            _rk4_step_curved!(Φ, Π, t, dt; dom, ops, bg!, ε_KO = 0.0)
            t += dt
        end
        @test all(isfinite, Φ) && all(isfinite, Π)
        @test maximum(abs, Φ) < 200 * amp
        @test maximum(abs, Π) < 2000 * amp
    end

    _progress("curved1d noise: variable β superluminal")
    @testset "Variable β = 2.0 + 0.5 sin(2π x): bounded" begin
        M = 16
        dom = make_periodic_domain_1d(T, M, 0.0, 1.0)
        bg! = _make_bg((t,x) -> T(2.0) + T(0.5)*sin(2π*x),
                        (t,x) -> one(T), dom, elem)
        Random.seed!(20260605)
        amp = sqrt(eps(T))
        Φ = amp .* randn(T, N, M); Π = amp .* randn(T, N, M)
        max_β = T(2.5)
        dt, n_steps = _crossings_and_dt_curved(dom, elem, max_β;
                                                n_xing = 50)
        t = zero(T)
        for _ in 1:n_steps
            _rk4_step_curved!(Φ, Π, t, dt; dom, ops, bg!, ε_KO = 1e-4)
            t += dt
        end
        @test all(isfinite, Φ) && all(isfinite, Π)
        @test maximum(abs, Φ) < 200 * amp
        @test maximum(abs, Π) < 2000 * amp
    end

    _progress("curved1d noise: sonic-horizon variable β")
    @testset "Variable β = 0.5 + sin(2π x), crosses 1: bounded" begin
        M = 16
        dom = make_periodic_domain_1d(T, M, 0.0, 1.0)
        bg! = _make_bg((t,x) -> T(0.5) + sin(2π*x),
                        (t,x) -> one(T), dom, elem)
        Random.seed!(20260606)
        amp = sqrt(eps(T))
        Φ = amp .* randn(T, N, M); Π = amp .* randn(T, N, M)
        max_β = T(1.5)
        dt, n_steps = _crossings_and_dt_curved(dom, elem, max_β;
                                                n_xing = 50)
        t = zero(T)
        for _ in 1:n_steps
            _rk4_step_curved!(Φ, Π, t, dt; dom, ops, bg!, ε_KO = 1e-4)
            t += dt
        end
        @test all(isfinite, Φ) && all(isfinite, Π)
        @test maximum(abs, Φ) < 500 * amp
        @test maximum(abs, Π) < 100 * amp
    end

    # (b.1) Plane-wave convergence on flat background.
    _progress("curved1d plane-wave convergence (β=0.5, γ=1)")
    @testset "Plane-wave convergence under M-refinement (β=0.5)" begin
        β_val = 0.5; k_w = T(2π); c_plus = one(T) - T(β_val)
        t_final = one(T) / c_plus
        errs = T[]
        for M_test in (8, 16, 32)
            dom = make_periodic_domain_1d(T, M_test, 0.0, 1.0)
            ξs = elem.xs
            xgrid = T[T(dom.xs[m] + dom.h * ξs[i])
                      for i in 1:N, m in 1:M_test]
            Φ = T[sin(k_w * x) for x in xgrid]
            # Π_densitised = √γ · Π_nat. γ=1 → Π = ∂_t Φ − β ∂_x Φ
            # for right-mover sin(k(x − c+ t)): Π = −k cos(k x).
            Π = T[-k_w * cos(k_w * x) for x in xgrid]
            bg! = _make_bg((t,x) -> T(β_val), (t,x) -> one(T), dom, elem)
            dt, _ = _crossings_and_dt_curved(dom, elem, β_val; n_xing = 1)
            n_steps = ceil(Int, t_final / dt)
            t = zero(T)
            for _ in 1:n_steps
                _rk4_step_curved!(Φ, Π, t, dt; dom, ops, bg!, ε_KO = 0.0)
                t += dt
            end
            Φ_exact = T[sin(k_w * (x - c_plus * t)) for x in xgrid]
            push!(errs, maximum(abs, Φ .- Φ_exact))
        end
        @test all(isfinite, errs)
        @test errs[end] < errs[1]
        gmean = (errs[1] / errs[end])^(1 / (length(errs) - 1))
        @test gmean > 2.5
    end

    # (b.2) Gauge-wave convergence + inertial-energy drift.
    _progress("curved1d gauge-wave convergence + energy")
    @testset "Gauge-wave convergence (β,γ vary in t and x)" begin
        ε = T(0.05); d = T(1); k = T(2π) / d; k₀ = T(2π)
        β_fn(t, x) = begin
            c = cos(k * (x - t))
            -ε * k * c / (1 + ε * k * c)
        end
        sγ_fn(t, x) = 1 + ε * k * cos(k * (x - t))
        ψ_fn(t, x)  = x + ε * sin(k * (x - t)) - t
        Φ_exact_fn(t, x) = sin(k₀ * ψ_fn(t, x))
        # Π_exact = √γ · (∂_t Φ − β ∂_x Φ). Verified analytically:
        # the inner bracket simplifies to −k₀ cos(k₀ ψ), giving
        # Π_exact = −k₀ √γ cos(k₀ ψ).
        Π_exact_fn(t, x) = -k₀ * sγ_fn(t, x) * cos(k₀ * ψ_fn(t, x))

        t_final = T(1.0)
        errs = T[]
        energies = NTuple{2, T}[]   # (E_initial, E_final) for finest M
        Ms = (8, 16, 32, 64)
        for M_test in Ms
            dom = make_periodic_domain_1d(T, M_test, 0.0, 1.0)
            ξs = elem.xs
            xgrid = T[T(dom.xs[m] + dom.h * ξs[i])
                      for i in 1:N, m in 1:M_test]
            Φ = T[Φ_exact_fn(zero(T), x) for x in xgrid]
            Π = T[Π_exact_fn(zero(T), x) for x in xgrid]
            bg! = _make_bg(β_fn, sγ_fn, dom, elem)
            # Maximum |β| over the run. For ε=0.05, k=2π:
            # max |β| ≈ εk / (1 − εk) ≈ 0.31/0.69 ≈ 0.46.
            max_β = ε * k / (1 - ε * k)
            dt, _ = _crossings_and_dt_curved(dom, elem, max_β; n_xing = 1)
            n_steps = ceil(Int, t_final / dt)

            # Inertial energy diagnostic at t=0 and t=t_final.
            # E_inertial = ½ Σ (Π² + (DΦ)²) / √γ · H_diag · h.
            function _E_inertial(Φ_state, Π_state, t_eval)
                DΦ = similar(Φ_state)
                # Use bulk-G (no SAT) for diagnostic — same caveat as
                # bin/curved1d_spectrum.jl.
                inv_h = 1 / dom.h
                @inbounds for m in 1:M_test, i in 1:N
                    s = zero(T)
                    for p in 1:N
                        s += T(ops.G[i, p]) * Φ_state[p, m]
                    end
                    DΦ[i, m] = s * inv_h
                end
                E = zero(T)
                @inbounds for m in 1:M_test, i in 1:N
                    x = T(dom.xs[m] + dom.h * ξs[i])
                    sγ = sγ_fn(t_eval, x)
                    w = T(ops.H[i, i]) * dom.h
                    E += T(0.5) * (Π_state[i,m]^2 + DΦ[i,m]^2) / sγ * w
                end
                return E
            end

            E_initial = _E_inertial(Φ, Π, zero(T))
            t = zero(T)
            for _ in 1:n_steps
                _rk4_step_curved!(Φ, Π, t, dt; dom, ops, bg!, ε_KO = 0.0)
                t += dt
            end
            t_actual = t
            Φ_exact = T[Φ_exact_fn(t_actual, x) for x in xgrid]
            err = maximum(abs, Φ .- Φ_exact)
            push!(errs, err)
            if M_test == last(Ms)
                E_final = _E_inertial(Φ, Π, t_actual)
                push!(energies, (E_initial, E_final))
            end
        end
        @test all(isfinite, errs)
        @test errs[end] < errs[1]
        rate = log2(errs[1] / errs[end]) / (length(errs) - 1)
        @test rate > 2   # Looser than plane-wave's ~5 — variable
        # coefficients reduce the asymptotic rate, but it must be at
        # least 2nd-order to indicate consistency.
        gmean = (errs[1] / errs[end])^(1 / (length(errs) - 1))
        @test gmean > 4
        # Energy drift at finest M. With ε_KO=0 the only dissipation
        # is RK4 phase error.
        E_initial, E_final = energies[1]
        rel_drift = abs(E_final / E_initial - 1)
        @test rel_drift < 1e-3
    end
end
