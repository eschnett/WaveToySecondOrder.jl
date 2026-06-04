# 1D scalar-wave app: evolve the conservative (Φ, Π) system on a 1+1
# ADM background (`W.evolve1d`) and render a four-panel figure — grid
# structure, initial vs final solution, total energy vs time, and L²
# error vs time. PNG output plus inline Sixel display when the
# terminal supports it.
#
# Run as a script with command-line flags:
#
#     julia --project bin/wave1d.jl --N 4 --M 32 --background sineshift \
#           --A 0.3 --t1 2.0 --out wave1d.png
#     julia --project bin/wave1d.jl --background constant_shift --shift 2.0 \
#           --ic noise --eps-ko 1e-4
#     julia --project bin/wave1d.jl --type Float32 --backend metal
#
# Flags (defaults in brackets):
#     --N <int>            GLL nodes per element [4]
#     --M <int>            number of elements [32]
#     --x0, --x1 <float>   domain bounds [0, 1]
#     --background <name>  minkowski | constant_shift | gaugewave |
#                          sineshift [sineshift]
#     --A <float>          background amplitude (gaugewave/sineshift) [0.3]
#     --d <float>          background period [1.0]
#     --shift <float>      shift β for constant_shift [0.5]
#     --ic <name>          exact | noise [exact]
#     --wavenumber <float> IC wavenumber k₀ [2π]
#     --bc <name>          periodic | auto | dirichlet | sommerfeld
#                          [periodic]; `auto` classifies each face from
#                          the background (subluminal → dirichlet for
#                          --ic exact / sommerfeld for --ic noise;
#                          superluminal → excision at outflow +
#                          full_dirichlet at inflow)
#     --bc-left <name>     per-face override (dirichlet | sommerfeld |
#     --bc-right <name>    excision | full_dirichlet)
#     --eps-ko <float>     Kreiss-Oliger coefficient [0.0]
#     --t1 <float>         final time [1.0]
#     --Nt <int>           number of samples [200]
#     --type <name>        Float64 | Float32 | Float64x2 [Float64]
#     --backend <name>     cpu | metal | cuda [cpu]
#     --out <path>         output PNG [bin/wave1d.png]
#
# Or call `main1d(; kwargs...)` from the REPL with `evolve1d` keywords.

using CairoMakie
using KernelAbstractions
using SixelTerm
using WaveToySecondOrder

const W = WaveToySecondOrder

function plot1d(res; out = joinpath(@__DIR__, "wave1d.png"))
    fig = Figure(size = (800, 700))

    # Panel 1: grid structure — element boundaries + GLL nodes.
    ax1 = Axis(fig[1, 1];
               title = "grid: M=$(res.mesh.Ne) elements, " *
                       "N=$(length(res.elem.xs)) GLL nodes",
               xlabel = "x", yticksvisible = false,
               yticklabelsvisible = false)
    vlines!(ax1, vec(res.mesh.vertex_coords);
            color = (:gray, 0.7), linewidth = 1)
    scatter!(ax1, res.xs_line, zero(res.xs_line);
             markersize = 5, color = :black)
    ylims!(ax1, -1, 1)

    # Panel 2: initial vs final solution.
    ax2 = Axis(fig[1, 2];
               title = "Φ: initial and final (t = " *
                       "$(round(res.ts_actual[end], sigdigits=4)))",
               xlabel = "x", ylabel = "Φ")
    lines!(ax2, res.xs_line, res.Φs[:, 1];   label = "t = $(res.ts[1])")
    lines!(ax2, res.xs_line, res.Φs[:, end]; label = "final")
    axislegend(ax2; position = :rb, framevisible = false)

    # Panel 3: total energy vs time.
    ax3 = Axis(fig[2, 1];
               title = "total energy (drift " *
                       "$(round(res.energy[end]/res.energy[1] - 1, sigdigits=3)))",
               xlabel = "t", ylabel = "E")
    lines!(ax3, res.ts_actual, res.energy)

    # Panel 4: L² error vs time (log scale; against the exact solution
    # for `ic = :exact`, against zero for `ic = :noise`).
    ax4 = Axis(fig[2, 2];
               title = res.ic === :exact ? "L² error vs exact" :
                                           "L² norm (noise)",
               xlabel = "t", ylabel = "‖Φ − Φ_exact‖", yscale = log10)
    err_floor = max(eps(Float64), 1e-300)
    lines!(ax4, res.ts_actual, max.(res.l2_err, err_floor))

    save(out, fig)
    println("wrote $out")
    isinteractive() || get(ENV, "TERM", "") != "" && try
        display(fig)
    catch
    end
    return fig
end

function main1d(; out = joinpath(@__DIR__, "wave1d.png"), kwargs...)
    res = W.evolve1d(; kwargs...)
    println("integrator = $(res.integrator_name)   ",
            "background = $(res.background)   ",
            "dt = $(round(res.dt, sigdigits=4))   ",
            "dx_min = $(round(res.dx, sigdigits=4))   ",
            "max L² err = $(round(maximum(res.l2_err), sigdigits=4))   ",
            "energy drift = $(round(res.energy[end]/res.energy[1] - 1,
                                    sigdigits=3))")
    plot1d(res; out)
    return res
end

# ----------------------------------------------------------------------
# Command-line interface.

function _parse_args(args)
    opts = Dict{String, String}()
    i = 1
    while i ≤ length(args)
        a = args[i]
        startswith(a, "--") ||
            error("wave1d.jl: expected --flag, got $a")
        key = a[3:end]
        i + 1 ≤ length(args) ||
            error("wave1d.jl: missing value for --$key")
        opts[key] = args[i + 1]
        i += 2
    end
    return opts
end

function _pick_type(name)
    name == "Float64" && return Float64
    name == "Float32" && return Float32
    if name == "Float64x2" || name == "Float32x2"
        try
            @eval using MultiFloats
            return name == "Float64x2" ? Base.invokelatest(() -> MultiFloats.Float64x2) :
                                         Base.invokelatest(() -> MultiFloats.Float32x2)
        catch
            error("wave1d.jl: --type $name requires the MultiFloats " *
                  "package in the active environment")
        end
    end
    error("wave1d.jl: unknown --type $name")
end

function _pick_backend(name, T)
    name == "cpu" && return KernelAbstractions.CPU()
    if name == "metal"
        T === Float32 ||
            error("wave1d.jl: Metal supports Float32 only; pass --type Float32")
        @eval using Metal
        return Base.invokelatest(() -> Metal.MetalBackend())
    end
    if name == "cuda"
        @eval using CUDA
        return Base.invokelatest(() -> CUDA.CUDABackend())
    end
    error("wave1d.jl: unknown --backend $name (cpu | metal | cuda)")
end

# Resolve the --bc / --bc-left / --bc-right flags into the `bc` kwarg
# of `evolve1d`: a bare Symbol (:periodic / :auto) or a per-face
# NamedTuple. Per-face overrides force the NamedTuple form (defaulting
# the other side from --bc when it names a concrete condition, else
# requiring both overrides).
function _pick_bc(o)
    base = get(o, "bc", "periodic")
    left = get(o, "bc-left", nothing)
    right = get(o, "bc-right", nothing)
    if left === nothing && right === nothing
        base in ("periodic", "auto") && return Symbol(base)
        return (left = Symbol(base), right = Symbol(base))
    end
    base in ("periodic", "auto") && (left === nothing || right === nothing) &&
        error("wave1d.jl: with --bc $base, give both --bc-left and --bc-right")
    return (left  = Symbol(something(left, base)),
            right = Symbol(something(right, base)))
end

function main1d_cli(args)
    o = _parse_args(args)
    T = _pick_type(get(o, "type", "Float64"))
    backend = _pick_backend(get(o, "backend", "cpu"), T)
    return main1d(;
        bc = _pick_bc(o),
        T, backend,
        N  = parse(Int, get(o, "N", "4")),
        M  = parse(Int, get(o, "M", "32")),
        x0 = parse(Float64, get(o, "x0", "0")),
        x1 = parse(Float64, get(o, "x1", "1")),
        background = Symbol(get(o, "background", "sineshift")),
        A  = parse(Float64, get(o, "A", "0.3")),
        d  = parse(Float64, get(o, "d", "1.0")),
        shift = parse(Float64, get(o, "shift", "0.5")),
        ic = Symbol(get(o, "ic", "exact")),
        ic_wavenumber = parse(Float64, get(o, "wavenumber", string(2π))),
        ε_KO = parse(Float64, get(o, "eps-ko", "0")),
        t1 = parse(Float64, get(o, "t1", "1.0")),
        Nt = parse(Int, get(o, "Nt", "200")),
        out = get(o, "out", joinpath(@__DIR__, "wave1d.png")))
end

if abspath(PROGRAM_FILE) == @__FILE__
    main1d_cli(ARGS)
end
