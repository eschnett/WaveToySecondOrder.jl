using CairoMakie
using OrdinaryDiffEqSymplecticRK
using SixelTerm
using WaveToySecondOrder

const W = WaveToySecondOrder

# Evolve

x0 = 0.0
x1 = 1.0
M  = 8                          # elements per axis (Mx = My = Mz = M)
N  = 5                          # GLL nodes per element

elem = W.make_element(Float64, N)
ops  = W.make_operators(elem)
mesh = W.make_cubical_mesh(Float64, M, x0, x1)
coords = W.element_coords(mesh, elem)        # (3, N, N, N, Ne)

# Element extent (= dom.h in the old code) — used to set dt via CFL.
dx = (coords[1, 2, 1, 1, 1] - coords[1, 1, 1, 1, 1])   # spacing between adjacent GLL nodes

u  = Array{Float64, 4}(undef, N, N, N, mesh.Ne)
u̇  = similar(u)

t0, t1 = 0.0, 1.0
A      = 1.0
kx = ky = kz = 2π
ω      = sqrt(kx^2 + ky^2 + kz^2)
W.initialize3d!(u, u̇, coords, t0; A, kx, ky, kz, ω)

τ  = 3//2 * (N-1)^2             # SIPG threshold ~ 2·(N−1)² in unit coords
dt = (1//2 * dx) / sqrt(3)      # CFL: 3D spectral radius ≈ 3× the 1D one

bdry_values = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)    # homogeneous outer Dirichlet
f!(ü, u̇, u, p, t) = W.rhs3d!(ü, u, u̇, bdry_values; mesh, ops, τ)
prob = SecondOrderODEProblem(f!, u̇, u, (t0, t1))
sol  = solve(prob, KahanLi8(); dt)

# ----------------------------------------------------------------------
# Spacetime line through the domain at y = z = 0.25. The middle of the
# domain (y = z = 0.5) is a zero of the sin(2π·y/z) eigenmode and would
# produce a flat-zero slice; y = z = 0.25 sits at a maximum (sin(π/2) = 1).
#
# Cubical-mesh element ordering: e = mx + (my-1)·M + (mz-1)·M².
elem_id(mx, my, mz) = mx + (my-1)*M + (mz-1)*M*M

# For a cubical axis-aligned mesh the y-coordinate of node (i, j, k) of
# element with column index `my` depends only on (j, my). Build a 2D
# coordinate table and find the (j, my) whose value is closest to 0.25.
function find_node_along(coords, axis::Int, target, M, N)
    table = Array{Float64}(undef, N, M)
    for my in 1:M
        e_ref = if axis == 1
            elem_id(my, 1, 1)
        elseif axis == 2
            elem_id(1, my, 1)
        else
            elem_id(1, 1, my)
        end
        for j in 1:N
            i_loc, j_loc, k_loc = axis == 1 ? (j, 1, 1) : axis == 2 ? (1, j, 1) : (1, 1, j)
            table[j, my] = coords[axis, i_loc, j_loc, k_loc, e_ref]
        end
    end
    idx = argmin(abs.(vec(table) .- target))
    j_loc = (idx - 1) %  N + 1
    m_idx = (idx - 1) ÷ N + 1
    return j_loc, m_idx
end
j_y, my_target = find_node_along(coords, 2, 0.25, M, N)   # y direction
k_z, mz_target = find_node_along(coords, 3, 0.25, M, N)   # z direction

# x-coordinates along the spacetime line: GLL nodes of every element along
# the x-row at (my_target, mz_target).
xs_line = Float64[]
for mx in 1:M
    e = elem_id(mx, my_target, mz_target)
    append!(xs_line, coords[1, :, j_y, k_z, e])
end

Nt = 200                        # number of time samples
ts = range(t0, t1, Nt)

# Buffers
n_dofs  = N^3 * mesh.Ne
us      = Array{Float64}(undef, length(xs_line), Nt)
u̇s      = Array{Float64}(undef, length(xs_line), Nt)
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

    # Spacetime slice along the x-axis at (j_y, k_z, my_target, mz_target).
    p = 1
    for mx in 1:M
        e = elem_id(mx, my_target, mz_target)
        for i in 1:N
            us[p, n] = u_arr[i, j_y, k_z, e]
            u̇s[p, n] = u̇_arr[i, j_y, k_z, e]
            p += 1
        end
    end

    # Error vs analytic eigenmode at this time.
    W.initialize3d!(u_exact, u̇_exact, coords, t; A, kx, ky, kz, ω)
    l2_err[n] = l2(u_arr .- u_exact)
end

# Figure

fig = Figure(; size=(800, 500))

ax1 = Axis(fig[1, 1];
           title  = "u(x, y=0.25, z=0.25, t)",
           xlabel = "x", ylabel = "t",
           aspect = DataAspect())
ax2 = Axis(fig[1, 3];
           title  = "u̇(x, y=0.25, z=0.25, t)",
           xlabel = "x", ylabel = "t",
           aspect = DataAspect())
ax3 = Axis(fig[2, 1:4];
           title  = "L2 error vs time",
           xlabel = "t", ylabel = "‖u_num − u_exact‖₂")

hm1 = heatmap!(ax1, xs_line, ts, us; colormap=:plasma)
Colorbar(fig[1, 2], hm1)

hm2 = heatmap!(ax2, xs_line, ts, u̇s; colormap=:plasma)
Colorbar(fig[1, 4], hm2)

lines!(ax3, ts, l2_err; linewidth = 2)

display(fig)

nothing
