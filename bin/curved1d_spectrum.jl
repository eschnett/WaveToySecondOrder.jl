using BenchmarkTools
using CairoMakie
using HexSBPSAT: make_element, make_operators
using LinearAlgebra
using Printf
using WaveToySecondOrder: make_periodic_domain_1d, wave1d_curved1d_rhs!

# Spectral analysis + benchmark + energy-conservation diagnostic for
# the conservative-form 1D kernel `wave1d_curved1d_rhs!`. The kernel
# evolves Φ, Π (densitised) on a 1+1 metric with α=1, varying β(t,x)
# and γ_xx(t,x). For the spatial-operator-only analysis here we
# freeze (β, γ_xx) at one snapshot (t=0 for the gauge-wave setup).

const N_GLL = 4
const T     = Float64

function _β_field(f, dom, elem)
    N = length(elem.xs); M = dom.M
    out = Matrix{T}(undef, N, M)
    @inbounds for m in 1:M, i in 1:N
        x = T(dom.xs[m] + dom.h * elem.xs[i])
        out[i, m] = f(x)
    end
    return out
end

# Build dense (2NM)×(2NM) operator by column-probing.
function build_curved_operator(dom, ops, β, sqrtγ, inv_sqrtγ; ε_KO = 0.0)
    N = length(ops.H.diag); M = dom.M
    NM = N * M
    M_op = zeros(T, 2NM, 2NM)
    Φ̇ = zeros(T, N, M); Π̇ = zeros(T, N, M)
    for j in 1:(2NM)
        v   = zeros(T, 2NM); v[j] = one(T)
        Φ   = reshape(view(v, 1:NM),     (N, M)) |> collect
        Π   = reshape(view(v, NM+1:2NM), (N, M)) |> collect
        fill!(Φ̇, zero(T)); fill!(Π̇, zero(T))
        wave1d_curved1d_rhs!(Φ̇, Π̇, Φ, Π; dom, ops,
                              β, sqrtγ, inv_sqrtγ, ε_KO)
        M_op[1:NM,     j] .= vec(Φ̇)
        M_op[NM+1:2NM, j] .= vec(Π̇)
    end
    return M_op
end

function spectrum_diagnostics(M_op::AbstractMatrix{T}) where {T}
    eig = eigen(M_op)
    λs  = eig.values
    max_re = maximum(real, λs)
    min_re = minimum(real, λs)
    max_im = maximum(abs ∘ imag, λs)
    ev_cond = cond(eig.vectors)
    return (; λs, max_re, min_re, max_im, ev_cond)
end

# Quick energy diagnostic — uses bulk SBP-G derivative only (no SAT),
# so it is NOT the kernel's conserved energy and dE/dt will look
# non-zero even for skew-stable setups. Useful only for order-of-
# magnitude monitoring; the true energy-drift check lives in the
# convergence test where it uses smooth states and the kernel D.
function inertial_energy(Φ, Π, β, sqrtγ, inv_sqrtγ, ops, dom)
    N, M = size(Φ)
    h = dom.h
    Hw = ops.H
    E = zero(T)
    # ∂_T φ = (Π/√γ) − β·(∂_xΦ/√γ) ... but β = -εk cos u/√γ, so we
    # use the physical relations directly:
    #   ∂_T φ = (Π/√γ) − (g_t/(1+g_x)) · (∂_xΦ/√γ),
    # which for our gauge-wave parametrisation reduces to
    #   ∂_T φ = Π/√γ + β·∂_xΦ/(sqrtγ)·(−1) but we don't need the
    # general formula here. The energy diagnostic instead uses the
    # ADM-norm formula
    #   E = ½ ∫ ((Π/√γ)² + (∂_xΦ)²/γ) √γ dx,
    # which equals the inertial energy for flat-γ_xx static metrics
    # and is the natural conserved quantity for the conservative
    # form when ∂_t γ = 0. For the gauge-wave (t-varying γ) this
    # diagnostic is not conserved exactly but is a sensible monitor.
    DΦ = Matrix{T}(undef, N, M)
    for m in 1:M, i in 1:N
        s = zero(T)
        for p in 1:N
            s += ops.G[i, p] * Φ[p, m]
        end
        DΦ[i, m] = s / h
    end
    @inbounds for m in 1:M, i in 1:N
        w = T(Hw[i, i]) * h
        pi_over_sqrtγ = Π[i, m] * inv_sqrtγ[i, m]
        dxΦ_over_sqrtγ = DΦ[i, m] * inv_sqrtγ[i, m]
        E += T(0.5) * (pi_over_sqrtγ^2 + dxΦ_over_sqrtγ^2) * sqrtγ[i, m] * w
    end
    return E
end

###############################################################################
# Setup ladder.

function run_all_setups()
    elem = make_element(T, N_GLL); ops = make_operators(elem)

    # Each entry: (label, M_test, β_fn(x), γ_fn(x) or nothing).
    # γ_fn returns √γ_xx (so for γ=1, pass identity-1 -> just 1).
    setups = Tuple{String, Int, Function, Union{Nothing, Function}}[]
    # Constant β, γ=1
    for M_test in (1, 4), β_val in (0.0, 0.5, 2.0)
        push!(setups,
              ("M=$M_test, β=$β_val, γ=1", M_test,
               (x -> T(β_val)), nothing))
    end
    # Variable β, γ=1
    push!(setups, ("M=1, β(x)=0.5+0.3 sin(2πx), γ=1", 1,
                   x -> T(0.5) + T(0.3)*sin(2π*x), nothing))
    push!(setups, ("M=4, β(x)=0.5+sin(2πx)   (sonic), γ=1", 4,
                   x -> T(0.5) + sin(2π*x), nothing))
    # Gauge-wave setup: β(t=0,x), √γ_xx(t=0,x). Choose ε so that
    # ε·k < 1/2 → max |β| < 1 everywhere (no sonic horizon).
    let ε = T(0.05), d = T(1)
        k = 2π / d
        push!(setups,
              ("M=4, gauge wave (ε=$ε, d=$d) at t=0", 4,
               x -> begin
                   c = cos(2π*x/d)
                   -ε*k*c / (1 + ε*k*c)
               end,
               x -> begin
                   c = cos(2π*x/d)
                   1 + ε*k*c
               end))
    end

    results = []
    for (label, M_test, β_fn, sqrtγ_fn) in setups
        dom = make_periodic_domain_1d(T, M_test, 0.0, 1.0)
        β = _β_field(β_fn, dom, elem)
        sqrtγ = sqrtγ_fn === nothing ? ones(T, N_GLL, M_test) :
                                        _β_field(sqrtγ_fn, dom, elem)
        inv_sqrtγ = one(T) ./ sqrtγ
        M_op = build_curved_operator(dom, ops, β, sqrtγ, inv_sqrtγ; ε_KO = 0.0)
        diag = spectrum_diagnostics(M_op)
        # Energy-drift diagnostic: dE/dt at a Gaussian-noise state.
        Φ_noise = randn(T, N_GLL, M_test)
        Π_noise = randn(T, N_GLL, M_test)
        Φ̇ = similar(Φ_noise); Π̇ = similar(Π_noise)
        wave1d_curved1d_rhs!(Φ̇, Π̇, Φ_noise, Π_noise; dom, ops,
                              β, sqrtγ, inv_sqrtγ, ε_KO = 0.0)
        # numerical dE/dt via 1-sided difference: E(state + h·ḋstate) − E(state)
        E0 = inertial_energy(Φ_noise, Π_noise, β, sqrtγ, inv_sqrtγ, ops, dom)
        h_pert = T(1e-6)
        E1 = inertial_energy(Φ_noise .+ h_pert.*Φ̇,
                              Π_noise .+ h_pert.*Π̇,
                              β, sqrtγ, inv_sqrtγ, ops, dom)
        dE_dt = (E1 - E0) / h_pert
        norm_state = sqrt(E0)
        push!(results,
              (; label, M = M_test, diag..., dE_dt, norm_state))
    end
    return results
end

function print_table(results)
    println()
    @printf("  %-2s  %-44s  %12s  %12s  %12s  %12s  %12s\n",
            "#", "setup", "max re(λ)", "min re(λ)", "max |im(λ)|",
            "cond(V)", "dE/dt")
    println("  ", "-"^124)
    for (i, r) in enumerate(results)
        @printf("  %-2d  %-44s  %+12.3e  %+12.3e  %12.3e  %12.3e  %+12.3e\n",
                i, r.label, r.max_re, r.min_re, r.max_im, r.ev_cond,
                r.dE_dt)
    end
    println()
end

function plot_spectra(results)
    n = length(results)
    cols = 2
    rows = ceil(Int, n / cols)
    fig = Figure(; size = (800, 240 * rows))
    for (i, r) in enumerate(results)
        row = (i - 1) ÷ cols + 1
        col = (i - 1) % cols + 1
        ax  = Axis(fig[row, col]; aspect = AxisAspect(1),
                    xlabel = "Re(λ)", ylabel = "Im(λ)",
                    title  = r.label)
        scatter!(ax, real.(r.λs), imag.(r.λs);
                 markersize = 6, color = :black, strokewidth = 0)
        vlines!(ax, [0.0]; color = (:gray, 0.4), linewidth = 0.5)
        hlines!(ax, [0.0]; color = (:gray, 0.4), linewidth = 0.5)
    end
    output = joinpath(@__DIR__, "curved1d_spectrum.png")
    save(output, fig)
    println("Wrote ", output)
    return fig
end

###############################################################################
# Top-of-output benchmark.

function bench_kernel()
    elem = make_element(T, N_GLL); ops = make_operators(elem)
    println("Kernel micro-benchmark (M=64, N=$N_GLL):")
    for εKO in (T(0.0), T(1e-4), T(0.1))
        dom = make_periodic_domain_1d(T, 64, 0.0, 1.0)
        Φ = randn(T, N_GLL, 64); Π = randn(T, N_GLL, 64)
        β = T(0.3) .* randn(T, N_GLL, 64)
        sqrtγ = ones(T, N_GLL, 64) .+ T(0.1) .* randn(T, N_GLL, 64)
        inv_sqrtγ = one(T) ./ sqrtγ
        Φ̇ = similar(Φ); Π̇ = similar(Π)
        b = @benchmark wave1d_curved1d_rhs!($Φ̇, $Π̇, $Φ, $Π;
                                              dom=$dom, ops=$ops,
                                              β=$β, sqrtγ=$sqrtγ,
                                              inv_sqrtγ=$inv_sqrtγ,
                                              ε_KO=$εKO) samples = 200 seconds = 1
        @printf("  ε_KO=%.0e  median = %8.3f μs  allocs = %d\n",
                εKO, median(b.times)/1e3, b.allocs)
    end
    println()
end

function main()
    bench_kernel()
    println("Building wave1d_curved1d spatial operators and analysing spectra…")
    results = run_all_setups()
    print_table(results)
    plot_spectra(results)
    return results
end

if abspath(PROGRAM_FILE) == (@__FILE__)
    main()
end
