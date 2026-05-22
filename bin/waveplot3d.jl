using CairoMakie
using OrdinaryDiffEqSymplecticRK
using SixelTerm
using WaveToySecondOrder

const W = WaveToySecondOrder

# Evolve

# Mesh choice. `:cubical` evolves on [0,1]³ with a uniform grid (default;
# the analytic sin-eigenmode is exact). `:inflated_cube` evolves on the
# 7-patch inflated cube tiling [-1,+1]³ — useful for stress-testing the
# mesh-driven RHS on a non-uniform, non-axis-aligned topology.
#
# Caveat: `rhs3d!` currently scales each element by `1/h_e²` where
# `h_e = corner2.x − corner1.x` and assumes the element's local axis 1 is
# aligned with physical x. On the inflated cube's outer patches that
# assumption is false (the −x patch yields `h_e = 0`, +y/+z patches give a
# value unrelated to the element size), so the run blows up to NaN. A
# real fix needs per-element / per-node metric handling; until then this
# branch is wired up only to exercise the mesh plumbing.
mesh_kind = :cubical            # :cubical | :inflated_cube

N  = 5                          # GLL nodes per element
M  = 8                          # cubical: elements per axis
R  = 0.1                        # inflated_cube: inner-patch radius

elem = W.make_element(Float64, N)
ops  = W.make_operators(elem)

if mesh_kind === :cubical
    x0, x1 = 0.0, 1.0
    mesh = W.make_cubical_mesh(Float64, M, x0, x1)
elseif mesh_kind === :inflated_cube
    x0, x1 = -1.0, 1.0
    mesh = W.make_inflated_cube_mesh(Float64, N, R)
else
    error("unknown mesh_kind: $mesh_kind")
end
geom   = W.make_geometry(mesh, elem)         # coords + per-node Jacobian
coords = geom.coords                         # (3, N, N, N, Ne)

# Smallest GLL-node spacing across the whole mesh — drives the CFL dt.
# Use the full 3D Euclidean distance between neighbouring nodes along local
# axis 1; the x-coordinate difference alone is signed and can be zero or
# negative for elements whose local axis 1 is not aligned with physical x
# (e.g. outer patches of the inflated cube).
function min_node_spacing(coords)
    h = Inf
    @inbounds for e in 1:size(coords, 5), k in 1:size(coords, 4),
                  j in 1:size(coords, 3), i in 2:size(coords, 2)
        dxv = coords[1, i, j, k, e] - coords[1, i-1, j, k, e]
        dyv = coords[2, i, j, k, e] - coords[2, i-1, j, k, e]
        dzv = coords[3, i, j, k, e] - coords[3, i-1, j, k, e]
        h = min(h, sqrt(dxv*dxv + dyv*dyv + dzv*dzv))
    end
    return h
end
dx = min_node_spacing(coords)

u  = Array{Float64, 4}(undef, N, N, N, mesh.Ne)
u̇  = similar(u)

t0, t1 = 0.0, 1.0
A      = 1.0
# Initial condition: domain-normalised sine with 3 extrema (2 interior
# nodes) per direction. Identical functional form on both meshes; the
# (x − x0)/(x1 − x0) normalisation makes it vanish on the outer boundary
# of whichever domain is selected.
kx = ky = kz = 3π
L_     = x1 - x0
ω      = sqrt(kx^2 + ky^2 + kz^2) / L_
@inbounds for e in 1:mesh.Ne, k in 1:N, j in 1:N, i in 1:N
    X = (coords[1, i, j, k, e] - x0) / L_
    Y = (coords[2, i, j, k, e] - x0) / L_
    Z = (coords[3, i, j, k, e] - x0) / L_
    u[i, j, k, e]  = A * sin(kx*X) * sin(ky*Y) * sin(kz*Z)
    u̇[i, j, k, e] = 0.0
end

τ  = 3//2 * (N-1)^2             # SIPG threshold ~ 2·(N−1)² in unit coords
dt = (1//2 * dx) / sqrt(3)      # CFL: 3D spectral radius ≈ 3× the 1D one

bdry_values = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)    # homogeneous outer Dirichlet
f!(ü, u̇, u, p, t) = W.rhs3d!(ü, u, u̇, bdry_values; mesh, ops, τ)
prob = SecondOrderODEProblem(f!, u̇, u, (t0, t1))
sol  = solve(prob, KahanLi8(); dt)

# ----------------------------------------------------------------------
# Spacetime slice. We pick a target (y, z) line through the domain and at
# each timestep gather the GLL-node values whose (y, z) coordinates match
# it (within a tolerance). Each match is stored as `(e, i, j, k)` so the
# inner loop is a tight 4D index lookup. Duplicates from shared element
# faces are removed.
#
# Choice of `y_target`/`z_target`:
#  - Cubical [0, 1]³: y = z = 0.25 — a maximum of sin(2π·y/z) (= 1).
#  - Inflated [-1, +1]³: y = z = 0 — the central axis passes through the
#    inner cube and the ±x outer patches; on this line sin(π·y)·sin(π·z)
#    is zero, so we set the IC to a non-separable bump instead (below).
function build_slice(coords, y_target, z_target; atol)
    Ne = size(coords, 5)
    N  = size(coords, 2)
    idx_list = NTuple{4, Int}[]
    xs       = Float64[]
    for e in 1:Ne, kk in 1:N, jj in 1:N, ii in 1:N
        y = coords[2, ii, jj, kk, e]
        z = coords[3, ii, jj, kk, e]
        if abs(y - y_target) < atol && abs(z - z_target) < atol
            x = coords[1, ii, jj, kk, e]
            isnew = !any(x0 -> abs(x0 - x) < atol, xs)
            if isnew
                push!(idx_list, (e, ii, jj, kk))
                push!(xs, x)
            end
        end
    end
    perm = sortperm(xs)
    return idx_list[perm], xs[perm]
end

y_target = mesh_kind === :cubical ? 0.25 : 0.0
z_target = mesh_kind === :cubical ? 0.25 : 0.0
slice_idx, xs_line = build_slice(coords, y_target, z_target; atol = 1e-9)
isempty(xs_line) && error("slice at y=$y_target, z=$z_target hit no GLL nodes")

Nt = 200                        # number of time samples
ts = range(t0, t1, Nt)

# Buffers
n_dofs  = N^3 * mesh.Ne
Ns      = length(xs_line)
us      = Array{Float64}(undef, Ns, Nt)
u̇s      = Array{Float64}(undef, Ns, Nt)
l2_err  = Vector{Float64}(undef, Nt)
u_exact = similar(u)
u̇_exact = similar(u̇)

# L2-proxy: sqrt(mean square). Differs from the true H-weighted norm by a
# uniform constant for the cubical mesh.
l2(a) = sqrt(sum(abs2, a) / length(a))

for (n, t) in enumerate(ts)
    solt = sol(t)
    @assert all(isfinite, solt)
    # SecondOrderODEProblem state layout: [u̇; u]
    u̇_arr = reshape(solt[1:n_dofs],          N, N, N, mesh.Ne)
    u_arr  = reshape(solt[n_dofs+1:2n_dofs], N, N, N, mesh.Ne)

    # Spacetime slice along the x-axis at (y_target, z_target).
    for (p, (e, ii, jj, kk)) in enumerate(slice_idx)
        us[p, n] = u_arr[ii, jj, kk, e]
        u̇s[p, n] = u̇_arr[ii, jj, kk, e]
    end

    # Diagnostic: L2 error vs analytic sine eigenmode on the cubical mesh;
    # plain L2 of u (energy proxy) on the inflated cube where no closed-
    # form solution exists.
    if mesh_kind === :cubical
        ct = cos(ω * t)
        @inbounds for e in 1:mesh.Ne, k in 1:N, j in 1:N, i in 1:N
            X = (coords[1, i, j, k, e] - x0) / L_
            Y = (coords[2, i, j, k, e] - x0) / L_
            Z = (coords[3, i, j, k, e] - x0) / L_
            u_exact[i, j, k, e] = A * sin(kx*X) * sin(ky*Y) * sin(kz*Z) * ct
        end
        l2_err[n] = l2(u_arr .- u_exact)
    else
        l2_err[n] = l2(u_arr)
    end
end

# Figure

fig = Figure(; size=(800, 500))

slice_label = "y=$(round(y_target; digits=3)), z=$(round(z_target; digits=3))"
ax1 = Axis(fig[1, 1];
           title  = "u(x, $slice_label, t)  [$(mesh_kind)]",
           xlabel = "x", ylabel = "t",
           aspect = DataAspect())
ax2 = Axis(fig[1, 3];
           title  = "u̇(x, $slice_label, t)  [$(mesh_kind)]",
           xlabel = "x", ylabel = "t",
           aspect = DataAspect())
ax3_title  = mesh_kind === :cubical ? "L2 error vs time"        : "L2 norm of u vs time"
ax3_ylabel = mesh_kind === :cubical ? "‖u_num − u_exact‖₂"      : "‖u‖₂"
ax3 = Axis(fig[2, 1:4]; title = ax3_title, xlabel = "t", ylabel = ax3_ylabel)

hm1 = heatmap!(ax1, xs_line, ts, us; colormap=:plasma)
Colorbar(fig[1, 2], hm1)

hm2 = heatmap!(ax2, xs_line, ts, u̇s; colormap=:plasma)
Colorbar(fig[1, 4], hm2)

lines!(ax3, ts, l2_err; linewidth = 2)

display(fig)

nothing
