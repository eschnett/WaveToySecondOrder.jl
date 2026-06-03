# Conservative-form 1D scalar wave on a 1+1 ADM background
# (`wave1d_curved_rhs!` in src/wave1d.jl), discretised on
# HexMeshes.Mesh{1} + HexSBPSAT.apply_D!. Evolves the densitised
# momentum Π := (√γ_xx/α)(∂_t Φ − β ∂_x Φ) with arbitrary α(t,x),
# β(t,x), γ_xx(t,x).
#
# Testsets:
#   (1) Spectrum: max Re(λ) of the column-probed RHS operator is zero
#       up to eigensolver round-off for stable configurations; the
#       sonic-horizon β needs KO and is then strictly dissipative.
#   (2) Noise robust stability: constant β ∈ {0, 0.5, 2.0} and
#       variable β (subluminal / superluminal / sonic), 50 light-
#       crossings, √eps noise IC.
#   (3) Plane-wave convergence (β = 0.5, α = γ = 1).

using HexMeshes: make_uniform_line
using HexSBPSAT: make_element, make_operators, make_geometry, to_device
using KernelAbstractions
using LinearAlgebra
using MultiFloats
using Random
using SpacetimeMetrics: GaugeWave, SineShift, adm_decompose
using StaticArrays: SVector
using Test
using WaveToySecondOrder: AnalyticBackground1D, MetricBackground1D,
                          make_wave1d_workspace, sample_background!,
                          wave1d_curved_rhs!, wave1d_energy

@isdefined(_progress) ||
    (_progress(msg) = (printstyled(stderr, "  • ", msg, "\n"; color = :cyan);
                       flush(stderr)))

# Bundle mesh + operators + scratch for one (N, M) configuration.
function _make_setup1d(::Type{T}, N, M; x0 = zero(T), x1 = one(T)) where {T}
    mesh = make_uniform_line(T, M, x0, x1; periodic = true)
    elem = make_element(T, N)
    ops  = make_operators(elem)
    geom = make_geometry(mesh, elem)
    ws   = make_wave1d_workspace(geom)
    xgrid = reshape(geom.coords[1, :, :], N, M)
    return (; mesh, elem, ops, geom, ws, xgrid)
end

# Background sampler closure over the package's `sample_background!`.
_bg_closure(bg, xgrid) =
    (a, β, sγ, t) -> sample_background!(a, β, sγ, bg, t, xgrid)

# Convenience: analytic closures α_fn(t,x), β_fn(t,x), γ_fn(t,x)
# (γ_fn returns γ_xx).
_make_bg1d(α_fn, β_fn, γ_fn, xgrid) =
    _bg_closure(AnalyticBackground1D(α_fn, β_fn, γ_fn), xgrid)

# RK4 with time-dependent background sampling at stage times.
function _rk4_wave1d!(Φ::AbstractMatrix{T}, Π::AbstractMatrix{T},
                      t::T, dt::T;
                      setup, bg!, ε_KO::Real, stages) where {T}
    (; geom, ops, ws) = setup
    (; a, β, sγ, k1Φ, k1Π, k2Φ, k2Π, k3Φ, k3Π, k4Φ, k4Π, Φs, Πs) = stages

    bg!(a, β, sγ, t)
    wave1d_curved_rhs!(k1Φ, k1Π, Φ, Π, a, β; geom, ops, ws, ε_KO)

    bg!(a, β, sγ, t + dt/2)
    @. Φs = Φ + (dt/2) * k1Φ; @. Πs = Π + (dt/2) * k1Π
    wave1d_curved_rhs!(k2Φ, k2Π, Φs, Πs, a, β; geom, ops, ws, ε_KO)

    @. Φs = Φ + (dt/2) * k2Φ; @. Πs = Π + (dt/2) * k2Π
    wave1d_curved_rhs!(k3Φ, k3Π, Φs, Πs, a, β; geom, ops, ws, ε_KO)

    bg!(a, β, sγ, t + dt)
    @. Φs = Φ + dt * k3Φ; @. Πs = Π + dt * k3Π
    wave1d_curved_rhs!(k4Φ, k4Π, Φs, Πs, a, β; geom, ops, ws, ε_KO)

    @. Φ += (dt/6) * (k1Φ + 2k2Φ + 2k3Φ + k4Φ)
    @. Π += (dt/6) * (k1Π + 2k2Π + 2k3Π + k4Π)
    return nothing
end

function _make_stages1d(::Type{T}, N, M;
                        backend = KernelAbstractions.CPU()) where {T}
    bufs = ntuple(_ -> KernelAbstractions.allocate(backend, T, N, M), 13)
    return (; a = bufs[1], β = bufs[2], sγ = bufs[3],
            k1Φ = bufs[4], k1Π = bufs[5], k2Φ = bufs[6], k2Π = bufs[7],
            k3Φ = bufs[8], k3Π = bufs[9], k4Φ = bufs[10], k4Π = bufs[11],
            Φs = bufs[12], Πs = bufs[13])
end

function _cfl_dt_1d(setup, max_speed; n_xing, cfl = 0.1)
    elem = setup.elem
    T = eltype(setup.xgrid)
    h = T(setup.geom.jac[1, 1, 1, 1])
    ξs = elem.xs
    dx_min = minimum(ξs[i+1] - ξs[i] for i in 1:(length(ξs)-1)) * h
    L = h * setup.geom.Ne
    dt = T(cfl) * dx_min / (one(T) + abs(T(max_speed)))
    t1 = T(n_xing) * L / (one(T) + abs(T(max_speed)))
    n_steps = ceil(Int, t1 / dt)
    return dt, n_steps
end

# Column-probe the linear RHS operator at a frozen background.
function _build_rhs_operator1d(setup, a, β; ε_KO)
    (; geom, ops, ws) = setup
    T = eltype(a)
    N, M = size(a)
    n = 2 * N * M
    A = zeros(T, n, n)
    Φ = zeros(T, N, M); Π = zeros(T, N, M)
    Φ̇ = similar(Φ); Π̇ = similar(Π)
    for j in 1:n
        fill!(Φ, 0); fill!(Π, 0)
        j ≤ N*M ? (Φ[j] = 1) : (Π[j - N*M] = 1)
        wave1d_curved_rhs!(Φ̇, Π̇, Φ, Π, a, β; geom, ops, ws, ε_KO)
        A[1:N*M, j]     = vec(Φ̇)
        A[N*M+1:end, j] = vec(Π̇)
    end
    return A
end

@testset "1D ADM scalar-wave kernel (wave1d_curved_rhs!)" begin
    T = Float64; N = 4

    # (1) Spectrum of the frozen-background RHS operator.
    _progress("wave1d: spectrum max Re(λ)")
    @testset "spectrum: max Re(λ) ≤ round-off (stable configs)" begin
        M = 8
        setup = _make_setup1d(T, N, M)
        xg = setup.xgrid
        configs = (
            ("β = 0",            x -> zero(T),                    0.0),
            ("β = 0.5",          x -> T(0.5),                     0.0),
            ("β = 2 (superlum)", x -> T(2.0),                     0.0),
            ("β var sublum",     x -> T(0.3) + T(0.2)*sinpi(2x),  0.0),
            ("β var sonic + KO", x -> T(0.5) + sinpi(2x),         1e-4),
            ("β var superlum + KO", x -> T(2.0) + T(0.5)*sinpi(2x), 1e-4),
        )
        for (label, β_fn, ε_KO) in configs
            a = ones(T, N, M)          # α = γ = 1
            β = β_fn.(xg)
            A = _build_rhs_operator1d(setup, a, β; ε_KO)
            λ = eigvals(A)
            max_re = maximum(real, λ)
            scale  = maximum(abs, λ)
            # Stable configurations: spectrum on (or left of) the
            # imaginary axis up to dense-eigensolver round-off on a
            # non-normal matrix (~√eps relative).
            @test max_re ≤ 1e-5 * scale
        end
        # Control: sonic-horizon β with ε_KO = 0 is genuinely unstable —
        # this is what KO is for.
        a = ones(T, N, M)
        β = (x -> T(0.5) + sinpi(2x)).(xg)
        λ = eigvals(_build_rhs_operator1d(setup, a, β; ε_KO = 0.0))
        @test maximum(real, λ) > 0.1
    end

    # (2) Noise robust stability — six setups, 50 light-crossings.
    # KO off where the wave operator alone is stable (constant /
    # subluminal-variable β); the superluminal- and sonic-variable
    # cases need a small ε_KO.
    _progress("wave1d noise: six setups, 50 crossings")
    noise_configs = (
        ("constant β = 0",     (t,x) -> zero(T),                 0.0,  0.0,  100, 1000),
        ("constant β = 0.5",   (t,x) -> T(0.5),                  0.5,  0.0,  100, 1000),
        ("constant β = 2.0",   (t,x) -> T(2.0),                  2.0,  0.0,  100, 1000),
        ("variable β sublum",  (t,x) -> T(0.3) + T(0.2)*sinpi(2x), 0.5, 0.0, 200, 2000),
        ("variable β superlum",(t,x) -> T(2.0) + T(0.5)*sinpi(2x), 2.5, 1e-4, 200, 2000),
        ("variable β sonic",   (t,x) -> T(0.5) + sinpi(2x),        1.5, 1e-4, 500, 2000),
    )
    for (i, (label, β_fn, max_β, ε_KO, boundΦ, boundΠ)) in enumerate(noise_configs)
        @testset "noise: $label bounded (50 crossings)" begin
            M = 16
            setup = _make_setup1d(T, N, M)
            bg! = _make_bg1d((t,x) -> one(T), β_fn, (t,x) -> one(T),
                             setup.xgrid)
            stages = _make_stages1d(T, N, M)
            Random.seed!(20260603 + i)
            amp = sqrt(eps(T))
            Φ = amp .* randn(T, N, M)
            Π = amp .* randn(T, N, M)
            dt, n_steps = _cfl_dt_1d(setup, max_β; n_xing = 50)
            t = zero(T)
            for _ in 1:n_steps
                _rk4_wave1d!(Φ, Π, t, dt; setup, bg!, ε_KO, stages)
                t += dt
            end
            @test all(isfinite, Φ) && all(isfinite, Π)
            @test maximum(abs, Φ) < boundΦ * amp
            @test maximum(abs, Π) < boundΠ * amp
        end
    end

    # (3) Plane-wave convergence on flat background, constant shift.
    _progress("wave1d plane-wave convergence (β = 0.5)")
    @testset "plane-wave convergence under M-refinement (β = 0.5)" begin
        β_val = T(0.5); k_w = T(2π); c_plus = one(T) - β_val
        t_final = one(T) / c_plus
        errs = T[]
        for M_test in (8, 16, 32)
            setup = _make_setup1d(T, N, M_test)
            xg = setup.xgrid
            # Right-mover Φ = sin(k(x − c₊t)); α = γ = 1 ⇒
            # Π = ∂_t Φ − β ∂_x Φ = −k cos(kx) at t = 0.
            Φ = sin.(k_w .* xg)
            Π = -k_w .* cos.(k_w .* xg)
            bg! = _make_bg1d((t,x) -> one(T), (t,x) -> β_val,
                             (t,x) -> one(T), xg)
            stages = _make_stages1d(T, N, M_test)
            dt, _ = _cfl_dt_1d(setup, β_val; n_xing = 1)
            n_steps = ceil(Int, t_final / dt)
            t = zero(T)
            for _ in 1:n_steps
                _rk4_wave1d!(Φ, Π, t, dt; setup, bg!, ε_KO = 0.0, stages)
                t += dt
            end
            Φ_exact = sin.(k_w .* (xg .- c_plus .* t))
            push!(errs, maximum(abs, Φ .- Φ_exact))
        end
        @test all(isfinite, errs)
        @test errs[end] < errs[1]
        gmean = (errs[1] / errs[end])^(1 / (length(errs) - 1))
        @test gmean > 2.5
    end

    # (4) Gauge wave via SpacetimeMetrics: varying lapse α = √H,
    # β = 0, γ_xx = H. Exercises MetricBackground1D / adm_decompose
    # and the α ≠ 1 momentum definition Π = (√γ/α)(∂_t Φ − β ∂_x Φ).
    # (In 1+1 this chart is conformally flat, so the kernel
    # coefficients are a = 1, β = 0 — the variable-coefficient paths
    # are exercised by the sine-shift test below.)
    _progress("wave1d gauge-wave (varying lapse) convergence")
    @testset "gauge-wave convergence (MetricBackground1D, α = √H)" begin
        A = T(0.1); d = T(1); k = T(2π) / d; k₀ = T(2π)
        C = A * d / (4 * T(π))
        bg = MetricBackground1D(GaugeWave(A, d))
        # Exact: Φ = sin(k₀(x̂ − t̂)) with x̂ − t̂ = x − t + 2C cos(k(x−t)).
        ψ_fn(t, x) = x - t + 2C * cos(k * (x - t))
        Φ_exact_fn(t, x) = sin(k₀ * ψ_fn(t, x))
        # Π = (√γ/α)(∂_t Φ − β ∂_x Φ) = ∂_t Φ = −k₀ H cos(k₀ ψ),
        # H = 1 − A sin(k(x − t)).
        Π_exact_fn(t, x) =
            -k₀ * (1 - A * sin(k * (x - t))) * cos(k₀ * ψ_fn(t, x))

        t_final = T(1)
        errs = T[]
        for M_test in (8, 16, 32)
            setup = _make_setup1d(T, N, M_test)
            xg = setup.xgrid
            Φ = Φ_exact_fn.(zero(T), xg)
            Π = Π_exact_fn.(zero(T), xg)
            bg! = _bg_closure(bg, xg)
            stages = _make_stages1d(T, N, M_test)
            dt, _ = _cfl_dt_1d(setup, 0.0; n_xing = 1)
            n_steps = ceil(Int, t_final / dt)
            t = zero(T)
            for _ in 1:n_steps
                _rk4_wave1d!(Φ, Π, t, dt; setup, bg!, ε_KO = 0.0, stages)
                t += dt
            end
            push!(errs, maximum(abs, Φ .- Φ_exact_fn.(t, xg)))
        end
        @test all(isfinite, errs)
        @test errs[end] < errs[1]
        gmean = (errs[1] / errs[end])^(1 / (length(errs) - 1))
        @test gmean > 2.5
    end

    # (5) Sine shift via SpacetimeMetrics: flat spacetime in a chart
    # with α = 1 and genuinely space- and time-varying β(t,x) and
    # γ_xx(t,x) — the variable-coefficient stress test (the successor
    # of the legacy "gauge-wave" testset), plus the energy monitor.
    _progress("wave1d sine-shift convergence + energy")
    @testset "sine-shift convergence + energy (β, γ vary in t, x)" begin
        A = T(0.3); d = T(1); k = T(2π) / d; k₀ = T(2π)
        C = A * d / (2 * T(π))
        bg = MetricBackground1D(SineShift(A, d))
        # Exact: Φ = sin(k₀(x̂ − t̂)), x̂ = x + C sin(k(x − t)), t̂ = t.
        ψ_fn(t, x) = x + C * sin(k * (x - t)) - t
        Φ_exact_fn(t, x) = sin(k₀ * ψ_fn(t, x))
        # Π = √γ (∂_t Φ − β ∂_x Φ) = −k₀ √γ cos(k₀ ψ),
        # √γ = 1 + A cos(k(x − t)).
        Π_exact_fn(t, x) =
            -k₀ * (1 + A * cos(k * (x - t))) * cos(k₀ * ψ_fn(t, x))

        max_β = A / (1 - A)
        t_final = T(1)
        errs = T[]
        drift = T(NaN)
        Ms = (8, 16, 32, 64)
        for M_test in Ms
            setup = _make_setup1d(T, N, M_test)
            xg = setup.xgrid
            Φ = Φ_exact_fn.(zero(T), xg)
            Π = Π_exact_fn.(zero(T), xg)
            bg! = _bg_closure(bg, xg)
            stages = _make_stages1d(T, N, M_test)
            dt, _ = _cfl_dt_1d(setup, max_β; n_xing = 1)
            n_steps = ceil(Int, t_final / dt)

            sample_background!(stages.a, stages.β, stages.sγ, bg,
                               zero(T), xg)
            E0 = wave1d_energy(Φ, Π, stages.sγ;
                               setup.geom, setup.ops, setup.ws)
            t = zero(T)
            for _ in 1:n_steps
                _rk4_wave1d!(Φ, Π, t, dt; setup, bg!, ε_KO = 0.0, stages)
                t += dt
            end
            push!(errs, maximum(abs, Φ .- Φ_exact_fn.(t, xg)))
            if M_test == last(Ms)
                # The background is time-periodic with period d, so at
                # t ≈ d the energy must return to its initial value up
                # to discretisation error.
                sample_background!(stages.a, stages.β, stages.sγ, bg,
                                   t, xg)
                E1 = wave1d_energy(Φ, Π, stages.sγ;
                                   setup.geom, setup.ops, setup.ws)
                drift = abs(E1 / E0 - 1)
            end
        end
        @test all(isfinite, errs)
        @test errs[end] < errs[1]
        rate = log2(errs[1] / errs[end]) / (length(errs) - 1)
        @test rate > 2
        gmean = (errs[1] / errs[end])^(1 / (length(errs) - 1))
        @test gmean > 4
        @test drift < 1e-3
    end

    # (6) Type-genericity: Float64x2 (MultiFloats) on the CPU. Trig of
    # MultiFloat is not reliable, so the IC is built in Float64 and
    # converted, and the background is the (constant-coefficient)
    # analytic one. The RK4 trajectory must agree with Float64 far
    # below Float64 round-off accumulation.
    _progress("wave1d Float64x2 (MultiFloats) CPU")
    @testset "Float64x2 agrees with Float64 (plane wave, β = 0.5)" begin
        M = 8
        results = Dict{DataType, Matrix{Float64}}()
        for T2 in (Float64, Float64x2)
            setup = _make_setup1d(T2, N, M)
            xg64 = Float64.(Float64(1) .* setup.xgrid)  # exact grid in Float64
            Φ = T2.(sin.(2π .* xg64))
            Π = T2.(-2π .* cos.(2π .* xg64))
            bg! = _make_bg1d((t,x) -> one(T2), (t,x) -> T2(1)/2,
                             (t,x) -> one(T2), setup.xgrid)
            stages = _make_stages1d(T2, N, M)
            # dt within both the wave CFL and the (tighter) KO-term
            # stability limit at this resolution.
            dt = T2(1) / 1024
            t = zero(T2)
            for _ in 1:64
                _rk4_wave1d!(Φ, Π, t, dt; setup, bg!, ε_KO = 1e-4, stages)
                t += dt
            end
            @test all(isfinite, Φ) && all(isfinite, Π)
            results[T2] = Float64.(Φ)
        end
        @test maximum(abs, results[Float64] .- results[Float64x2]) < 1e-12
    end
end

# (7) GPU smoke test: full RHS + background sampling on Metal
# (Float32), compared against the identical CPU Float32 run.
# Auto-skips when Metal isn't installed or functional.
if !@isdefined(HAS_METAL)
    const HAS_METAL = try
        @eval using Metal
        Metal.functional()
    catch
        false
    end
end

if HAS_METAL
    @testset "wave1d on Metal (Float32)" begin
        _progress("wave1d Metal smoke test (Float32)")
        T = Float32; N = 4; M = 16
        setup = _make_setup1d(T, N, M)
        # Closures must capture no `Type` objects (non-isbits) so the
        # background struct can be passed into a GPU kernel.
        bg = AnalyticBackground1D((t, x) -> 1.0f0,
                                  (t, x) -> 0.5f0 + 0.3f0 * sinpi(2 * x),
                                  (t, x) -> 1.0f0 + 0.2f0 * sinpi(2 * x)^2)
        # Smooth IC: the KO D⁶ chain on rough data amplifies Float32
        # round-off (different GPU summation order) beyond any useful
        # comparison tolerance; on smooth data the h⁵-scaled KO term is
        # tame and CPU/GPU agree to ordinary Float32 round-off.
        xg0 = setup.xgrid
        Φ0 = sinpi.(2 .* xg0)
        Π0 = -2 .* T(π) .* cospi.(2 .* xg0)
        # dt within both the wave CFL and the (stiffer) KO-term limit.
        dt = T(5.0f-4)
        n_steps = 50

        run_on = function (backend, geom, xgrid)
            ws = make_wave1d_workspace(geom)
            stages = _make_stages1d(T, N, M; backend)
            Φ = KernelAbstractions.allocate(backend, T, N, M)
            Π = KernelAbstractions.allocate(backend, T, N, M)
            copyto!(Φ, Φ0); copyto!(Π, Π0)
            bg! = (a, β, sγ, t) -> sample_background!(a, β, sγ, bg, t, xgrid)
            local setup2 = (; setup.elem, setup.ops, geom, ws,
                            xgrid = setup.xgrid)
            t = zero(T)
            for _ in 1:n_steps
                _rk4_wave1d!(Φ, Π, t, dt; setup = setup2, bg!,
                             ε_KO = 1.0f-4, stages)
                t += dt
            end
            sγ = KernelAbstractions.allocate(backend, T, N, M)
            a = similar(sγ); β = similar(sγ)
            sample_background!(a, β, sγ, bg, t, xgrid)
            E = wave1d_energy(Φ, Π, sγ; geom, setup.ops, ws)
            return Array(Φ), Array(Π), Float64(E)
        end

        Φc, Πc, Ec = run_on(KernelAbstractions.CPU(), setup.geom,
                            setup.xgrid)

        backend = MetalBackend()
        geom_dev = to_device(setup.geom, backend)
        xg_dev = KernelAbstractions.allocate(backend, T, N, M)
        copyto!(xg_dev, setup.xgrid)
        Φg, Πg, Eg = run_on(backend, geom_dev, xg_dev)

        @test all(isfinite, Φg) && all(isfinite, Πg)
        @test maximum(abs, Φg .- Φc) ≤ 1e-3 * max(1, maximum(abs, Φc))
        @test maximum(abs, Πg .- Πc) ≤ 1e-3 * max(1, maximum(abs, Πc))
        @test abs(Eg - Ec) ≤ 1e-3 * abs(Ec)
    end
end
