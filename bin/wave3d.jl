# 3D conservative scalar-wave app: evolve the first-order (Φ,Π) system on
# a 3+1 ADM background (`W.evolve3d(; formulation = :conservative)`) on the
# axis-aligned uniform_hex mesh (Milestone 1), and render an x-slice
# spacetime heatmap, total energy vs time, and L² error vs time. PNG +
# Sixel.
#
#     julia --project bin/wave3d.jl --N 4 --M 8 --background minkowski
#     julia --project bin/wave3d.jl --background constant_shift --shift 0.1,0.0,0.0 \
#           --bc auto --ic gaussian --eps-ko 0.1
#
# Flags: --N --M --x0 --x1 --background (minkowski|constant_shift)
# --shift (e.g. 0.1,0.0,0.0) --ic (exact|gaussian|noise) --ic-width --bc
# (periodic|auto) --eps-ko --t1 --Nt --type --backend --out.

using WaveToySecondOrder
const W = WaveToySecondOrder
using CairoMakie
using SixelTerm
import KernelAbstractions

function _plot3d(res, out)
    fig = Figure(size = (800, 280))
    ax1 = Axis(fig[1, 1]; title = "Φ on x-slice (y=z=$(round(res.y_target, digits=2)))",
               xlabel = "x", ylabel = "t")
    heatmap!(ax1, res.xs_line, res.ts, res.Φs; colormap = :balance)
    ax2 = Axis(fig[1, 2]; title = "energy", xlabel = "t")
    lines!(ax2, res.ts_actual, res.energy)
    ax3 = Axis(fig[1, 3]; title = "L² error", xlabel = "t", yscale = log10)
    lines!(ax3, res.ts_actual, max.(res.l2_err, eps()))
    save(out, fig)
    println("wrote $out")
    display(fig)
end

function main3d(; out = joinpath(@__DIR__, "wave3d.png"), kwargs...)
    res = W.evolve3d(; formulation = :conservative, kwargs...)
    println("integrator = $(res.integrator_name)   background = $(res.background)   ",
            "dt = $(round(res.dt, sigdigits = 3))   max L² err = ",
            "$(round(maximum(res.l2_err), sigdigits = 4))   energy drift = ",
            "$(round(res.energy[end] / res.energy[1] - 1, sigdigits = 3))")
    _plot3d(res, out)
    return res
end

_parse_args(args) = begin
    o = Dict{String,String}(); i = 1
    while i ≤ length(args)
        startswith(args[i], "--") || error("wave3d.jl: expected --flag, got $(args[i])")
        o[args[i][3:end]] = args[i+1]; i += 2
    end
    o
end
_ptype(s) = s == "Float64" ? Float64 : s == "Float32" ? Float32 :
            error("wave3d.jl: --type must be Float64 or Float32")
function _pbackend(s, T)
    s == "cpu" && return KernelAbstractions.CPU()
    s == "metal" && (T === Float32 || error("Metal needs --type Float32");
                     @eval using Metal; return Base.invokelatest(()->Metal.MetalBackend()))
    s == "cuda" && (@eval using CUDA; return Base.invokelatest(()->CUDA.CUDABackend()))
    error("wave3d.jl: unknown --backend $s")
end

function main3d_cli(args)
    o = _parse_args(args)
    T = _ptype(get(o, "type", "Float64"))
    backend = _pbackend(get(o, "backend", "cpu"), T)
    shift = Tuple(parse.(Float64, split(get(o, "shift", "0.0,0.0,0.0"), ",")))
    bc = Symbol(get(o, "bc", "periodic"))
    # `_pbackend` may `@eval using Metal/CUDA` at runtime; run via
    # `invokelatest` so the freshly-loaded device methods are visible.
    return Base.invokelatest(main3d; T, backend,
        N = parse(Int, get(o, "N", "4")), M = parse(Int, get(o, "M", "8")),
        x0 = parse(Float64, get(o, "x0", "0")), x1 = parse(Float64, get(o, "x1", "1")),
        background = Symbol(get(o, "background", "minkowski")),
        shift, ic = Symbol(get(o, "ic", "exact")), bc,
        ic_width = parse(Float64, get(o, "ic-width", "0.15")),
        ε_KO = parse(Float64, get(o, "eps-ko", "0")),
        t1 = parse(Float64, get(o, "t1", "1.0")), Nt = parse(Int, get(o, "Nt", "200")),
        out = get(o, "out", joinpath(@__DIR__, "wave3d.png")))
end

if abspath(PROGRAM_FILE) == @__FILE__
    main3d_cli(ARGS)
end
