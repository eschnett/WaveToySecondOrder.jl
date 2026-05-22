using CairoMakie
using SixelTerm
using StaticArrays
using WaveToySecondOrder

const W = WaveToySecondOrder

# Mesh choice mirrors `waveplot3d.jl`.
mesh_kind = :inflated_cube      # :cubical | :inflated_cube

N = 5                           # GLL nodes per element
M = 8                           # cubical: elements per axis
R = 0.1                         # inflated_cube: inner-patch radius

elem = W.make_element(Float64, N)

if mesh_kind === :cubical
    x0, x1 = 0.0, 1.0
    mesh = W.make_cubical_mesh(Float64, M, x0, x1)
elseif mesh_kind === :inflated_cube
    x0, x1 = -1.0, 1.0
    mesh = W.make_inflated_cube_mesh(Float64, N, R)
else
    error("unknown mesh_kind: $mesh_kind")
end
geom = W.make_geometry(mesh, elem)

# 3-extrema (2-node) sine IC in each direction. Normalising `(x − x0) /
# (x1 − x0)` makes the formula identical on both domains and ensures the
# function vanishes on the outer boundary.
A   = 1.0
k_  = 3π
ic(x, y, z) = A * sin(k_ * (x - x0) / (x1 - x0)) *
                  sin(k_ * (y - x0) / (x1 - x0)) *
                  sin(k_ * (z - x0) / (x1 - x0))

u = Array{Float64, 4}(undef, N, N, N, mesh.Ne)
for e in 1:mesh.Ne, k in 1:N, j in 1:N, i in 1:N
    u[i, j, k, e] = ic(geom.coords[1, i, j, k, e],
                       geom.coords[2, i, j, k, e],
                       geom.coords[3, i, j, k, e])
end
println("IC range: ", extrema(u))

# Two coordinate-plane slices, both at the geometric midpoint along the
# slice axis (where the 3-extrema mode reaches an extremum).
xmid = (x0 + x1) / 2

# (x, y) plane at z = xmid -------------------------------------------------
Ng = 120
xs = range(x0, x1; length = Ng)
ys = range(x0, x1; length = Ng)
pts_xy = [SVector(x, y, xmid) for x in xs, y in ys]
u_xy = W.interpolate_field(geom, elem, u, pts_xy)

# (x, z) plane at y = xmid -------------------------------------------------
zs = range(x0, x1; length = Ng)
pts_xz = [SVector(x, xmid, z) for x in xs, z in zs]
u_xz = W.interpolate_field(geom, elem, u, pts_xz)

# Figure -------------------------------------------------------------------
fig = Figure(; size = (1000, 500))
title_pad = "IC = sin(3π·X)·sin(3π·Y)·sin(3π·Z)   [$(mesh_kind), N=$N]"
Label(fig[0, 1:3], title_pad; fontsize = 16)

ax_xy = Axis(fig[1, 1]; title = "z = $(round(xmid; digits=3))",
             xlabel = "x", ylabel = "y", aspect = DataAspect())
hm_xy = heatmap!(ax_xy, xs, ys, u_xy; colormap = :balance,
                 colorrange = (-1, 1))
Colorbar(fig[1, 2], hm_xy)

ax_xz = Axis(fig[1, 3]; title = "y = $(round(xmid; digits=3))",
             xlabel = "x", ylabel = "z", aspect = DataAspect())
heatmap!(ax_xz, xs, zs, u_xz; colormap = :balance, colorrange = (-1, 1))

display(fig)

nothing
