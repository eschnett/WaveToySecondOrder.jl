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
dom  = W.make_domain(Float64, M, x0, x1)

x = [x + dom.h * a for a in elem.xs, x in dom.xs]
y = x
z = x
dx = dom.h * elem.h

u   = Array{Float64, 6}(undef, N, N, N, M, M, M)
u̇   = similar(u)

t0, t1 = 0.0, 1.0
A      = 1.0
kx = ky = kz = 2π
ω      = sqrt(kx^2 + ky^2 + kz^2)
W.initialize3d!(u, u̇, x, y, z, t0; A, kx, ky, kz, ω)

τ  = 3//2 * (N-1)^2             # SIPG threshold ~ 2·(N−1)² in unit coords
dt = (1//2 * dx) / sqrt(3)      # CFL: 3D spectral radius ≈ 3× the 1D one

bxL = bxR = byL = byR = bzL = bzR = 0.0    # homogeneous outer Dirichlet
f!(ü, u̇, u, p, t) =
    W.rhs3d!(ü, u, u̇, bxL, bxR, byL, byR, bzL, bzR; dom, ops, τ)
prob = SecondOrderODEProblem(f!, u̇, u, (t0, t1))
sol  = solve(prob, KahanLi8(); dt)

# Pick a spacetime line through the domain at y = z = 0.25. The middle of
# the domain (y = z = 0.5) is a zero of the sin(2π·y/z) eigenmode and would
# produce a flat-zero slice; y = z = 0.25 sits at a maximum (sin(π/2) = 1).
function _node_index(coords::AbstractMatrix, target)
    n, _ = size(coords)
    idx = argmin(abs.(vec(coords) .- target))   # global linear index
    j = (idx - 1) %  n + 1                      # local node index
    m = (idx - 1) ÷ n + 1                       # element index
    return j, m
end
j_mid, my_mid = _node_index(y, 0.25)
k_mid, mz_mid = _node_index(z, 0.25)

xs = vec(x)                     # length N·M
Nt = 200                        # number of time samples
ts = range(t0, t1, Nt)

# Buffers
n_dofs  = N^3 * M^3
us      = Array{Float64}(undef, length(xs), Nt)   # u   along the spacetime line
u̇s      = Array{Float64}(undef, length(xs), Nt)   # u̇ along the spacetime line
l2_err  = Vector{Float64}(undef, Nt)              # L2 norm of error vs time
u_exact = Array{Float64, 6}(undef, N, N, N, M, M, M)
u̇_exact = similar(u_exact)

# A reasonable proxy for the L2 norm on the GLL grid. (The proper H-weighted
# norm would multiply each node by w_i·w_j·w_k·h_x·h_y·h_z; for a uniform
# cubic mesh that's just an overall constant, so it does not change the
# *shape* of the error-vs-time curve, only its absolute scale.)
l2(a) = sqrt(sum(abs2, a) / length(a))

for (n, t) in enumerate(ts)
    solt = sol(t)
    @assert all(isfinite, solt)
    # SecondOrderODEProblem state layout: [u̇; u]
    u̇_arr = reshape(solt[1:n_dofs],            N, N, N, M, M, M)
    u_arr  = reshape(solt[n_dofs+1 : 2n_dofs], N, N, N, M, M, M)

    # Spacetime slice along the x-axis at the chosen (y_mid, z_mid) line.
    us[:, n] .= vec(u_arr[:, j_mid, k_mid, :, my_mid, mz_mid])
    u̇s[:, n] .= vec(u̇_arr[:, j_mid, k_mid, :, my_mid, mz_mid])

    # Error vs analytic eigenmode at this time.
    W.initialize3d!(u_exact, u̇_exact, x, y, z, t;
                    A, kx, ky, kz, ω)
    l2_err[n] = l2(u_arr .- u_exact)
end

# Figure

fig = Figure(; size=(800, 500))

ax1 = Axis(fig[1, 1];
           title  = "u(x, y_mid, z_mid, t)",
           xlabel = "x", ylabel = "t",
           aspect = DataAspect())
ax2 = Axis(fig[1, 3];
           title  = "u̇(x, y_mid, z_mid, t)",
           xlabel = "x", ylabel = "t",
           aspect = DataAspect())
ax3 = Axis(fig[2, 1:4];
           title  = "L2 error vs time",
           xlabel = "t", ylabel = "‖u_num − u_exact‖₂")

hm1 = heatmap!(ax1, xs, ts, us; colormap=:plasma)
Colorbar(fig[1, 2], hm1)

hm2 = heatmap!(ax2, xs, ts, u̇s; colormap=:plasma)
Colorbar(fig[1, 4], hm2)

lines!(ax3, ts, l2_err; linewidth = 2)

display(fig)

nothing
