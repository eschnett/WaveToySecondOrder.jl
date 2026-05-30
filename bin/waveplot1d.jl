using CairoMakie
using SixelTerm
using WaveToySecondOrder

const W = WaveToySecondOrder

# Thin plot wrapper around `W.evolve1d`. 1D analog of
# `bin/waveplot{2,3}d.jl`. The spacetime `u(x, t)` heatmap IS the full
# domain in 1D, so there's no separate snapshot plot.

function main1d(; kwargs...)
    res = W.evolve1d(; kwargs...)
    println("integrator = $(res.integrator_name)   ",
            "τ = $(res.params.τ)   ",
            "dt = $(round(res.dt, sigdigits=4))   ",
            "dx = $(round(res.dx, sigdigits=4))   ",
            "dt/dx = $(round(res.dt/res.dx, sigdigits=4))")

    T = eltype(res.u_final)
    N = size(res.u_final, 1)
    M = size(res.u_final, 2)

    fig = Figure(; size = (800, 800))

    plot_tag = "1D N=$(N), M=$(M)"
    ax1 = Axis(fig[1, 1];
               title  = "u(x, t)  [$plot_tag]",
               xlabel = "x", ylabel = "t")
    ax2 = Axis(fig[1, 3];
               title  = "u̇(x, t)  [$plot_tag]",
               xlabel = "x", ylabel = "t")
    ax3 = Axis(fig[2, 1:4];
               title  = "L² error vs analytic eigenmode  [$plot_tag]",
               xlabel = "t", ylabel = "‖u_num − u_exact‖_{L²}")
    ax4 = Axis(fig[3, 1:4];
               title  = "u(x, t = t1) vs t = t0  [$plot_tag]",
               xlabel = "x", ylabel = "u")

    hm1 = heatmap!(ax1, res.xs_line, res.ts, res.us; colormap = :plasma)
    Colorbar(fig[1, 2], hm1)

    hm2 = heatmap!(ax2, res.xs_line, res.ts, res.u̇s; colormap = :plasma)
    Colorbar(fig[1, 4], hm2)

    lines!(ax3, res.ts, res.l2_err; linewidth = 2)

    # Final-time profile next to the IC, for an eyeball check that the
    # standing wave returns to its IC at every full period.
    lines!(ax4, res.xs_line, res.us[:, 1];   label = "t = t0",
                                             color = :gray, linewidth = 1)
    lines!(ax4, res.xs_line, res.us[:, end]; label = "t = t1",
                                             color = :crimson, linewidth = 2)
    axislegend(ax4; position = :rb)

    display(fig)

    return fig
end

# Usage:
#
#     main1d(; N = 4, M = 32)               # default: sin(2π x) · cos(2π t), one period
#     main1d(; N = 5, M = 16, t1 = 2)       # two periods
#     main1d(; N = 3, M = 64, ic_wavenumber = 4π)   # higher-frequency mode
#
# Float32 (any backend):
#
#     main1d(; T = Float32, N = 4, M = 32)
#
# Apple Silicon GPU (Float32 + MetalBackend):
#
#     using Metal
#     main1d(; T = Float32, backend = MetalBackend(), N = 4, M = 32)

nothing
