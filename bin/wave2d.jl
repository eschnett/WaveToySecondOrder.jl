# 2D scalar-wave app: evolve the conservative (Φ,Π) system on a 2+1
# ADM background (`W.evolve2d`) and render grid, a y-slice spacetime
# heatmap, total energy vs time, and L² error vs time. PNG + Sixel.
#
#     julia --project bin/wave2d.jl --N 4 --M 24 --background minkowski
#     julia --project bin/wave2d.jl --background gaugewave --A 0.1 --bc periodic
#     julia --project bin/wave2d.jl --background minkowski --bc auto --ic noise
#
# Flags: --N --M --mesh (cubical|cubed_square) --R (cubed-square radius)
# --x0 --x1 --background (minkowski|constant_shift|gaugewave) --A --d
# --shift (e.g. 0.05,0.0) --ic (exact|gaussian|noise) --ic-width --bc
# (periodic|auto) --eps-ko --t1 --Nt --type --backend --out.
#
#     julia --project bin/wave2d.jl --mesh cubed_square --ic gaussian \
#           --eps-ko 0.1 --t1 1.5

using CairoMakie
using KernelAbstractions
using SixelTerm
using WaveToySecondOrder

const W = WaveToySecondOrder

function plot2d(res; out = joinpath(@__DIR__, "wave2d.png"))
    fig = Figure(size = (800, 700))

    ax1 = Axis(fig[1, 1]; title = "grid: $(res.mesh.Ne) elements",
               xlabel = "x", ylabel = "y", aspect = DataAspect())
    QE = ((1,2),(2,3),(3,4),(4,1))
    for e in 1:res.mesh.Ne, (a, b) in QE
        va = res.mesh.vertex_idx[a, e]; vb = res.mesh.vertex_idx[b, e]
        lines!(ax1, [res.mesh.vertex_coords[1, va], res.mesh.vertex_coords[1, vb]],
                    [res.mesh.vertex_coords[2, va], res.mesh.vertex_coords[2, vb]];
               color = (:gray, 0.6), linewidth = 0.6)
    end

    ax2 = Axis(fig[1, 2]; title = "Φ(x, y=$(round(res.y_target,sigdigits=3)), t)",
               xlabel = "x", ylabel = "t")
    heatmap!(ax2, res.xs_line, collect(res.ts_actual), res.Φs; colormap = :balance)

    ax3 = Axis(fig[2, 1]; title = "total energy", xlabel = "t", ylabel = "E")
    lines!(ax3, res.ts_actual, res.energy)

    ax4 = Axis(fig[2, 2];
               title = res.ic === :exact ? "L² error vs exact" : "L² norm (noise)",
               xlabel = "t", ylabel = "‖Φ−Φ_exact‖", yscale = log10)
    lines!(ax4, res.ts_actual, max.(res.l2_err, eps(Float64)))

    save(out, fig)
    println("wrote $out")
    try; display(fig); catch; end
    return fig
end

function main2d(; out = joinpath(@__DIR__, "wave2d.png"), kwargs...)
    res = W.evolve2d(; kwargs...)
    println("integrator = $(res.integrator_name)   background = $(res.background)   ",
            "dt = $(round(res.dt, sigdigits=4))   ",
            "max L² err = $(round(maximum(res.l2_err), sigdigits=4))   ",
            "energy drift = $(round(res.energy[end]/res.energy[1]-1, sigdigits=3))")
    plot2d(res; out)
    return res
end

function _parse_args(args)
    o = Dict{String,String}(); i = 1
    while i ≤ length(args)
        startswith(args[i], "--") || error("wave2d.jl: expected --flag, got $(args[i])")
        o[args[i][3:end]] = args[i+1]; i += 2
    end
    return o
end

_ptype(s) = s == "Float64" ? Float64 : s == "Float32" ? Float32 :
            error("wave2d.jl: --type must be Float64 or Float32")
function _pbackend(s, T)
    s == "cpu" && return KernelAbstractions.CPU()
    s == "metal" && (T === Float32 || error("Metal needs --type Float32");
                     @eval using Metal; return Base.invokelatest(()->Metal.MetalBackend()))
    s == "cuda" && (@eval using CUDA; return Base.invokelatest(()->CUDA.CUDABackend()))
    error("wave2d.jl: unknown --backend $s")
end

function main2d_cli(args)
    o = _parse_args(args)
    T = _ptype(get(o, "type", "Float64"))
    backend = _pbackend(get(o, "backend", "cpu"), T)
    shift = Tuple(parse.(Float64, split(get(o, "shift", "0.0,0.0"), ",")))
    bcs = get(o, "bc", "periodic")
    bc = bcs in ("periodic", "auto") ? Symbol(bcs) : Symbol(bcs)
    return main2d(; T, backend,
        N = parse(Int, get(o, "N", "4")), M = parse(Int, get(o, "M", "16")),
        mesh_kind = Symbol(get(o, "mesh", "cubical")),
        R = parse(Float64, get(o, "R", "0.3")),
        ic_width = parse(Float64, get(o, "ic-width", "0.15")),
        x0 = parse(Float64, get(o, "x0", "0")), x1 = parse(Float64, get(o, "x1", "1")),
        background = Symbol(get(o, "background", "minkowski")),
        A = parse(Float64, get(o, "A", "0.1")), d = parse(Float64, get(o, "d", "1.0")),
        shift, ic = Symbol(get(o, "ic", "exact")), bc,
        ε_KO = parse(Float64, get(o, "eps-ko", "0")),
        t1 = parse(Float64, get(o, "t1", "1.0")), Nt = parse(Int, get(o, "Nt", "200")),
        out = get(o, "out", joinpath(@__DIR__, "wave2d.png")))
end

if abspath(PROGRAM_FILE) == @__FILE__
    main2d_cli(ARGS)
end
