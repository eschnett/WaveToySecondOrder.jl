using CairoMakie
using SixelTerm
using WaveToySecondOrder

const W = WaveToySecondOrder

# Mesh choice: same switch as `waveplot3d.jl`. Visualises the topology
# (large dots = the eight corner vertices of each hex) and the geometric
# discretisation (small dots = GLL collocation points produced by the
# trilinear element map).
mesh_kind = :cubed_cube      # :cubical | :cubed_cube

N = 4                           # GLL nodes per element
M = 4                           # subdivisions per patch axis (kept small to be legible)
R = 0.1                         # cubed_cube: inner-patch radius

elem = W.make_element(Float64, N)

if mesh_kind === :cubical
    mesh = W.make_cubical_mesh(Float64, M, 0.0, 1.0)
elseif mesh_kind === :cubed_cube
    mesh = W.make_cubed_cube_mesh(Float64, M, R)
else
    error("unknown mesh_kind: $mesh_kind")
end
geom = W.make_geometry(mesh, elem)

# Vertex point cloud: one column per distinct mesh vertex.
vx = mesh.vertex_coords[1, :]
vy = mesh.vertex_coords[2, :]
vz = mesh.vertex_coords[3, :]

# Element edges in Gmsh canonical ordering. A hex has 12 edges: four around
# the bottom face, four around the top face, and four vertical risers.
const HEX_EDGES = (
    (1, 2), (2, 3), (3, 4), (4, 1),
    (5, 6), (6, 7), (7, 8), (8, 5),
    (1, 5), (2, 6), (3, 7), (4, 8),
)
ex = Float64[]; ey = Float64[]; ez = Float64[]
for e in 1:mesh.Ne, (a, b) in HEX_EDGES
    va = mesh.vertex_idx[a, e]
    vb = mesh.vertex_idx[b, e]
    push!(ex, mesh.vertex_coords[1, va], mesh.vertex_coords[1, vb])
    push!(ey, mesh.vertex_coords[2, va], mesh.vertex_coords[2, vb])
    push!(ez, mesh.vertex_coords[3, va], mesh.vertex_coords[3, vb])
end

# Collocation point cloud: flatten the (3, N, N, N, Ne) coordinate array.
cx = vec(geom.coords[1, :, :, :, :])
cy = vec(geom.coords[2, :, :, :, :])
cz = vec(geom.coords[3, :, :, :, :])

fig = Figure(; size = (800, 800))
ax  = Axis3(fig[1, 1];
            title  = "mesh = $mesh_kind   (Ne=$(mesh.Ne), Nv=$(W.nv(mesh)))",
            xlabel = "x", ylabel = "y", zlabel = "z",
            aspect = (1, 1, 1))

linesegments!(ax, ex, ey, ez; color = :black, linewidth = 0.5,
              label = "element edges")
scatter!(ax, cx, cy, cz; markersize = 4,  color = (:steelblue, 0.6),
         label = "collocation points")
scatter!(ax, vx, vy, vz; markersize = 12, color = :crimson,
         label = "element vertices")

axislegend(ax; position = :lt)

display(fig)

nothing
