using CairoMakie
using HexSBPSAT: make_element, make_operators
using LinearAlgebra
using Printf
using WaveToySecondOrder: make_periodic_domain_1d, wave1d_admshift_rhs!

# Spectral analysis of the 1D ADM-shift discrete spatial operator.
# No time evolution: build the linearised operator `M` as a dense
# matrix, compute `eigvals(M)`, and read off
#   - max real(λ)     — asymptotic growth rate
#   - max |imag(λ)|   — discrete wave-speed range
#   - cond(eigvecs M) — defectiveness (Jordan blocks on imag axis
#                       give polynomial growth even when real(λ)=0)
#
# Setup ladder (progressively more complex):
#   1.  M=1, β = 0
#   2.  M=1, β = 0.5
#   3.  M=1, β = 2.0        — single-element superluminal
#   4.  M=4, β = 0           — multi-element seam coupling
#   5.  M=4, β = 0.5
#   6.  M=4, β = 2.0        — multi-element superluminal
#   7.  M=1, β(x) = 0.5 + 0.3 sin(2πx)
#   8.  M=1, β(x) = 2.0 + 0.5 sin(2πx)    — single-element + variable + super
#   9.  M=4, β(x) = 2.0 + 0.5 sin(2πx)    — multi-element + variable + super
#   10. M=4, β(x) = 0.5 + sin(2πx)        — sonic horizon inside the domain
#
# Output: console table + `admshift_spectrum.png` with the
# eigenvalue scatter for every setup.

const N_GLL = 4               # GLL nodes per element
const T     = Float64

# Build the dense `(2NM) × (2NM)` operator matrix that maps
# vec((Φ, Π)) to vec((Φ̇, Π̇)) for `wave1d_admshift_rhs!` with the
# given (dom, ops, β, ε_KO). State layout: first `NM` entries are
# Φ, next `NM` are Π, both in column-major order matching `(N, M)`
# arrays.
function build_admshift_operator(dom, ops; β, ε_KO::Real = 0.0)
    N = length(ops.H.diag); M = dom.M
    NM   = N * M
    M_op = zeros(T, 2NM, 2NM)
    Φ̇ = zeros(T, N, M); Π̇ = zeros(T, N, M)
    for j in 1:(2NM)
        v   = zeros(T, 2NM); v[j] = one(T)
        Φ   = reshape(view(v, 1:NM),       (N, M)) |> collect
        Π   = reshape(view(v, NM+1:2NM),   (N, M)) |> collect
        fill!(Φ̇, zero(T)); fill!(Π̇, zero(T))
        wave1d_admshift_rhs!(Φ̇, Π̇, Φ, Π; dom, ops, β, ε_KO)
        M_op[1:NM,       j] .= vec(Φ̇)
        M_op[NM+1:2NM,   j] .= vec(Π̇)
    end
    return M_op
end

# Spectrum diagnostics: returns a NamedTuple summarising the
# spectrum of `M_op`.
function spectrum_diagnostics(M_op::AbstractMatrix{T}) where {T}
    eig = eigen(M_op)
    λs  = eig.values
    V   = eig.vectors
    max_re = maximum(real, λs)
    min_re = minimum(real, λs)
    max_im = maximum(abs ∘ imag, λs)
    ev_cond = cond(V)
    return (; λs, max_re, min_re, max_im, ev_cond)
end

# Sample β(x) at every GLL node into an `(N, M)` matrix.
function β_field(f, dom, ops)
    N = length(ops.H.diag); M = dom.M
    elem_xs = [zero(T); zero(T)]
    # `dom.xs[m]` is the left edge of element m; `ops` doesn't carry
    # the GLL nodes directly here, so recompute from the operator's
    # H weights. The cleanest path is to capture the GLL ξ values
    # from `make_element` at the call site and pass them in; for
    # this analysis we'll accept the `elem` argument as part of the
    # interface.
    error("β_field needs `elem` — use the version below")
end

function β_field(f, dom, elem)
    N = length(elem.xs); M = dom.M
    β = Matrix{T}(undef, N, M)
    for m in 1:M, i in 1:N
        x = T(dom.xs[m] + dom.h * elem.xs[i])
        β[i, m] = f(x)
    end
    return β
end

# Run all setups, return a Vector of NamedTuples — one per setup,
# carrying its label, the spectrum, and the (M, β-profile) metadata
# needed for plotting.
function run_all_setups()
    elem = make_element(T, N_GLL); ops = make_operators(elem)

    setups = Tuple{Int, String, Any}[]
    # Constant β, single + multi element
    for M_test in (1, 4), β in (0.0, 0.5, 2.0)
        β_label = β == 0.0 ? "β=0" : (β == 0.5 ? "β=0.5" : "β=2.0")
        push!(setups, (M_test, "M=$M_test, $β_label", β))
    end
    # Variable β (uses closures evaluated later)
    push!(setups, (1, "M=1, β(x)=0.5+0.3 sin(2πx)",
                     x -> T(0.5) + T(0.3) * sin(2π * x)))
    push!(setups, (1, "M=1, β(x)=2.0+0.5 sin(2πx)",
                     x -> T(2.0) + T(0.5) * sin(2π * x)))
    push!(setups, (4, "M=4, β(x)=2.0+0.5 sin(2πx)",
                     x -> T(2.0) + T(0.5) * sin(2π * x)))
    push!(setups, (4, "M=4, β(x)=0.5+sin(2πx)  (sonic horizon)",
                     x -> T(0.5) + sin(2π * x)))

    results = []
    for (M_test, label, β_spec) in setups
        dom = make_periodic_domain_1d(T, M_test, 0.0, 1.0)
        β_arg = β_spec isa Real ? β_spec : β_field(β_spec, dom, elem)
        M_op  = build_admshift_operator(dom, ops; β = β_arg, ε_KO = 0.0)
        diag  = spectrum_diagnostics(M_op)
        push!(results, (; M = M_test, label, diag...))
    end
    return results
end

function print_table(results)
    println()
    @printf("  %-2s  %-44s  %12s  %12s  %12s  %12s\n",
            "#", "setup", "max real(λ)", "min real(λ)",
            "max |im(λ)|", "cond(V)")
    println("  ", "-"^110)
    for (i, r) in enumerate(results)
        @printf("  %-2d  %-44s  %+12.3e  %+12.3e  %12.3e  %12.3e\n",
                i, r.label, r.max_re, r.min_re, r.max_im, r.ev_cond)
        if r.max_re > 1e-10
            τ = 1 / r.max_re
            @printf("       e-folding time τ = 1/max_re = %.2e\n", τ)
        end
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
        # Reference line: imag axis.
        vlines!(ax, [0.0]; color = (:gray, 0.4), linewidth = 0.5)
        hlines!(ax, [0.0]; color = (:gray, 0.4), linewidth = 0.5)
    end
    output = joinpath(@__DIR__, "admshift_spectrum.png")
    save(output, fig)
    println("Wrote ", output)
    return fig
end

function main()
    println("Building ADM-shift spatial operators and analysing spectra…")
    results = run_all_setups()
    print_table(results)
    plot_spectra(results)
    return results
end

if abspath(PROGRAM_FILE) == (@__FILE__)
    main()
end
