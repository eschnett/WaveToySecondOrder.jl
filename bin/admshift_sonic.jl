using CairoMakie
using HexSBPSAT: make_element, make_operators
using LinearAlgebra
using Printf
using WaveToySecondOrder: make_periodic_domain_1d, wave1d_admshift_rhs!

# Detailed analysis of the sonic-horizon-like growth in the
# consistent-D ADM-shift discretisation. The earlier spectral
# analysis showed that `β(x) = 0.5 + sin(2π x)` produces an
# eigenvalue with positive real part ~ 1.57 at M=4. This script
# digs in:
#
#   1. Convergence in M for several A in β(x) = 0.5 + A sin(2π x).
#   2. Comparison of constant β vs variable β at the same max β.
#   3. Spatial localisation of the most-unstable eigenmode.
#
# Output: console summary + `admshift_sonic.png`.

const N_GLL = 4
const T_TYPE     = Float64

include(joinpath(@__DIR__, "admshift_spectrum.jl"))

# (Defines build_admshift_operator, β_field, spectrum_diagnostics.)

function run_convergence()
    elem = make_element(T_TYPE, N_GLL); ops = make_operators(elem)
    As   = (0.0, 0.2, 0.4, 0.49, 0.5, 0.6, 0.8, 1.0)
    Ms   = (4, 8, 16, 32, 64)
    results = Dict{Float64, Vector{Float64}}()
    println("Convergence of max real(λ) under M-refinement for β(x) = 0.5 + A sin(2π x)")
    println()
    @printf("  %-6s ", "A")
    for M in Ms; @printf("%10s ", "M=$M"); end
    println()
    println("  ", "-"^(8 + 11 * length(Ms)))
    for A in As
        row = Float64[]
        @printf("  %-6.2f ", A)
        for M in Ms
            dom = make_periodic_domain_1d(T_TYPE, M, 0.0, 1.0)
            β = A == 0.0 ? 0.5 :
                β_field(x -> T_TYPE(0.5) + T_TYPE(A) * sin(2π * x),
                         dom, elem)
            M_op = build_admshift_operator(dom, ops; β, ε_KO = 0.0)
            max_re = maximum(real, eigvals(M_op))
            push!(row, max_re)
            @printf("%10.3e ", max_re)
        end
        println()
        results[A] = row
    end
    return results, Ms, As
end

function plot_convergence(results, Ms, As)
    fig = Figure(; size = (800, 480))
    ax = Axis(fig[1, 1];
              xlabel = "M (elements; N = 4 GLL nodes each)",
              ylabel = "max real(λ)",
              xscale = log2, yscale = log10,
              title  = "Sonic-horizon: max real(λ) vs M for β(x) = 0.5 + A sin(2π x)")
    for (A, row) in sort(collect(results))
        rowclamped = max.(row, 1e-12)   # floor for log plot
        crosses1   = (0.5 + A) > 1
        ls = crosses1 ? :solid : :dash
        scatterlines!(ax, collect(Ms), rowclamped;
                       label = "A = $A" * (crosses1 ? " (crosses 1)" : ""),
                       linestyle = ls)
    end
    hlines!(ax, [1e-6]; color = (:gray, 0.5), linestyle = :dot,
             label = "noise floor (≈ cond(V)·eps·‖M‖)")
    axislegend(ax; position = :rt, fontsize = 9)
    out = joinpath(@__DIR__, "admshift_sonic.png")
    save(out, fig)
    println("Wrote ", out)
end

function analyse_eigenmode()
    elem = make_element(T_TYPE, N_GLL); ops = make_operators(elem)
    println()
    println("Spatial localisation of the most-unstable mode at M = 32, A = 1.0:")
    M_test = 32
    dom = make_periodic_domain_1d(T_TYPE, M_test, 0.0, 1.0)
    β_arr = β_field(x -> T_TYPE(0.5) + sin(2π * x), dom, elem)
    M_op = build_admshift_operator(dom, ops; β = β_arr, ε_KO = 0.0)
    eig = eigen(M_op)
    imax = argmax(real.(eig.values))
    λ    = eig.values[imax]
    v    = eig.vectors[:, imax]
    NM   = N_GLL * M_test
    Φv = real.(v[1:NM]); Πv = real.(v[NM+1:2NM])
    @printf("  λ = %.4f + %.4f i\n", real(λ), imag(λ))
    @printf("  ‖Φ‖_∞ = %.3e   ‖Π‖_∞ = %.3e\n",
            maximum(abs, Φv), maximum(abs, Πv))
    # Spatial map of |Π| (the dominant component).
    Π_map  = reshape(abs.(Πv), N_GLL, M_test)
    β_map  = β_arr
    nodewise_β   = vec(β_arr)
    nodewise_Π = vec(abs.(Πv))
    perm = sortperm(nodewise_Π; rev = true)
    println("  Top 8 nodes by |Π|:")
    for k in 1:8
        j = perm[k]
        m = (j-1) ÷ N_GLL + 1; i = ((j-1) % N_GLL) + 1
        x = dom.xs[m] + dom.h * elem.xs[i]
        @printf("    x = %5.3f   β(x) = %5.3f   |Π| = %.3e\n",
                x, β_arr[i, m], nodewise_Π[j])
    end
    return (; λ, Φv, Πv, β_arr, dom, elem)
end

function main()
    results, Ms, As = run_convergence()
    plot_convergence(results, Ms, As)
    analyse_eigenmode()
end

if abspath(PROGRAM_FILE) == (@__FILE__)
    main()
end
