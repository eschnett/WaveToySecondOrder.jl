using CairoMakie
using KernelAbstractions
using SixelTerm
using StaticArrays
using WaveToySecondOrder

const W = WaveToySecondOrder

# Thin plot wrapper around `W.evolve3d`. The driver — mesh build,
# integration, spacetime sampling, L²-error trace — lives in the
# package (`src/evolve.jl`); this script only assembles the figure
# from the returned NamedTuple.
#
# Default run: cubical mesh, Cartesian sin·sin·sin eigenmode IC on CPU.
# `mesh_kind` and `ic_kind` are independent — any pairing is allowed.
# See `?evolve3d` for all supported combinations.

function main3d(; kwargs...)
    res = W.evolve3d(; kwargs...)
    println("integrator = $(res.integrator_name)   ",
            "τ = $(res.params.τ)   ",
            "dt = $(round(res.dt, sigdigits=4))   ",
            "dx_min = $(round(res.dx, sigdigits=4))   ",
            "dt/dx_min = $(round(res.dt/res.dx, sigdigits=4))")

    T = eltype(res.u_final)
    N = size(res.u_final, 1)
    mesh, elem = res.mesh, res.elem
    x0, x1 = res.x0, res.x1

    # 2D slice of the final-time solution on the z = 0 plane.
    # `interpolate_field` does Newton iteration on the trilinear element
    # map per query point; the brute-force element search makes the loop
    # `O(Ng² · Ne)`. For 120² and Ne ≲ 5000 it takes a couple of seconds.
    # `default = NaN` for off-mesh points so the inflated-cube outer
    # sphere's exterior renders transparent in the heatmap below.
    Ng     = 120
    xs_xy  = range(x0, x1; length = Ng)
    ys_xy  = range(x0, x1; length = Ng)
    pts_xy = [SVector{3,T}(x, y, zero(T)) for x in xs_xy, y in ys_xy]
    u_xy   = W.interpolate_field(mesh, elem.xs, res.u_final, pts_xy;
                                   default = T(NaN))

    # Element-boundary segments lying on z = 0: collect the 12 edges
    # per element and keep only those whose endpoint vertices both sit
    # on the plane (Δz < tol). For meshes where the layer at z = 0
    # isn't a vertex layer (e.g. the cubed cube's ±z patches), nothing
    # is drawn for those patches — the plane cuts through their
    # interiors, not along an element edge.
    edge_x = T[]
    edge_y = T[]
    HEX_EDGES = ((1,2),(2,3),(3,4),(4,1),
                 (5,6),(6,7),(7,8),(8,5),
                 (1,5),(2,6),(3,7),(4,8))
    let atol = sqrt(eps(T))
        for e in 1:mesh.Ne, (a, b) in HEX_EDGES
            va = mesh.vertex_idx[a, e]
            vb = mesh.vertex_idx[b, e]
            if abs(mesh.vertex_coords[3, va]) < atol &&
               abs(mesh.vertex_coords[3, vb]) < atol
                push!(edge_x, mesh.vertex_coords[1, va],
                              mesh.vertex_coords[1, vb])
                push!(edge_y, mesh.vertex_coords[2, va],
                              mesh.vertex_coords[2, vb])
            end
        end
    end

    fig = Figure(; size = (800, 1000))

    plot_tag = "mesh=$(res.mesh_kind), ic=$(res.ic_kind)"
    slice_label = "y=$(round(res.y_target; digits=3)), z=$(round(res.z_target; digits=3))"
    ax1 = Axis(fig[1, 1];
               title  = "u(x, $slice_label, t)  [$plot_tag]",
               xlabel = "x", ylabel = "t",
               aspect = DataAspect())
    ax2 = Axis(fig[1, 3];
               title  = "u̇(x, $slice_label, t)  [$plot_tag]",
               xlabel = "x", ylabel = "t",
               aspect = DataAspect())
    ax3 = Axis(fig[2, 1:4];
               title  = "Physical L² error vs analytic eigenmode  [$plot_tag]",
               xlabel = "t", ylabel = "‖u_num − u_exact‖_{H_phys}")
    ax4 = Axis(fig[3, 1:3];
               title  = "u(x, y, z=0, t = t1)  [$plot_tag]",
               xlabel = "x", ylabel = "y",
               aspect = DataAspect())

    hm1 = heatmap!(ax1, res.xs_line, res.ts, res.us; colormap = :plasma)
    Colorbar(fig[1, 2], hm1)

    hm2 = heatmap!(ax2, res.xs_line, res.ts, res.u̇s; colormap = :plasma)
    Colorbar(fig[1, 4], hm2)

    lines!(ax3, res.ts, res.l2_err; linewidth = 2)

    umax_slice = maximum(abs, filter(isfinite, u_xy))
    hm4 = heatmap!(ax4, xs_xy, ys_xy, u_xy;
                   colormap   = :plasma,
                   colorrange = (-umax_slice, umax_slice))
    linesegments!(ax4, edge_x, edge_y; color = (:black, 0.6), linewidth = 0.6)
    Colorbar(fig[3, 4], hm4)

    display(fig)

    return fig
end

# Usage:
#
#     main(; N = 4, M = 8)
#     main(; mesh_kind = :cubed_cube, N = 4, M = 8, R = 0.1)
#     main(; mesh_kind = :inflated_cube, ic_kind = :radial, N = 4, M = 8)
#     main(; mesh_kind = :inflated_cube, outer_bc = :sommerfeld,
#            ic_kind = :outgoing, N = 4, M = 8)
#
# Apple Silicon GPU run (Float32, MetalBackend):
#
#     using Metal
#     main(; T = Float32, backend = MetalBackend(),
#            mesh_kind = :inflated_cube, ic_kind = :radial, N = 4, M = 8)

nothing
