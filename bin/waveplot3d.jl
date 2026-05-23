using CairoMakie
using OrdinaryDiffEqSymplecticRK
using ProgressMeter
using SixelTerm
using WaveToySecondOrder

const W = WaveToySecondOrder

# Pick a symplectic partitioned RK whose order matches the spatial
# polynomial order `N − 1` of the GLL element. Higher order = more
# stages = more RHS evaluations per step, so we want the time scheme
# only as accurate as the space scheme.
function pick_integrator(N::Integer)
    if     N ≤ 2;  return VelocityVerlet()  # 2nd-order (1 stage)
    elseif N == 3; return VelocityVerlet()  # 2nd-order
    elseif N == 4; return Ruth3()           # 3rd-order  (3 stages)
    elseif N == 5; return CandyRoz4()       # 4th-order  (4 stages)
    elseif N == 6; return McAte5()          # 5th-order  (6 stages)
    elseif N == 7; return KahanLi6()        # 6th-order  (9 stages)
    else           return KahanLi8()        # 8th-order  (17 stages)
    end
end

# Smallest GLL-node spacing across the whole mesh (3D Euclidean — handles
# curvilinear elements whose local axis 1 is not aligned with physical x).
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

# Locate GLL nodes whose (y, z) coordinates match a target line within
# tolerance. Returns the sorted-by-x list of `(e, i, j, k)` indices plus
# the corresponding x-coordinates. Duplicates from shared element faces
# are removed.
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

# ----------------------------------------------------------------------
# Main driver. Wrapped in a function so all locals are type-inferred and
# the RHS closure `f!` doesn't pay dynamic-dispatch costs on each of the
# integrator's per-step stage evaluations.
function main(; mesh_kind::Symbol = :cubical,
                N::Int = 5,
                M::Int = 8,
                R::Float64 = 0.1)

    elem = W.make_element(Float64, N)
    ops  = W.make_operators(elem)

    if mesh_kind === :cubical
        x0, x1 = 0.0, 1.0
        mesh = W.make_cubical_mesh(Float64, M, x0, x1)
    elseif mesh_kind === :inflated_cube
        x0, x1 = -1.0, 1.0
        mesh = W.make_inflated_cube_mesh(Float64, M, R)
    else
        error("unknown mesh_kind: $mesh_kind")
    end
    geom   = W.make_geometry(mesh, elem)         # coords + per-node Jacobian
    coords = geom.coords                         # (3, N, N, N, Ne)

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
    L_ = x1 - x0
    ω  = sqrt(kx^2 + ky^2 + kz^2) / L_
    @inbounds for e in 1:mesh.Ne, k in 1:N, j in 1:N, i in 1:N
        X = (coords[1, i, j, k, e] - x0) / L_
        Y = (coords[2, i, j, k, e] - x0) / L_
        Z = (coords[3, i, j, k, e] - x0) / L_
        u[i, j, k, e]  = A * sin(kx*X) * sin(ky*Y) * sin(kz*Z)
        u̇[i, j, k, e] = 0.0
    end

    # SIPG penalty constant. Rule of thumb: ~1.5·(N−1)² for axis-aligned
    # cubical meshes, ~8·(N−1)² for curvilinear / multi-patch meshes —
    # the higher value is needed to keep the operator negative semi-
    # definite on anisotropic outer-patch elements. (4·(N−1)² is right
    # at the NSD threshold for the inflated cube; the extra margin
    # prevents slow exponential growth from a few residual positive
    # eigenvalues.)
    τ  = mesh_kind === :cubical ? 1.5 * (N-1)^2 : 8.0 * (N-1)^2
    # `cfl_safety = 0.5` (vs the default 0.9) gives a margin for two
    # things at once: the higher-stage symplectic integrators (CandyRoz4
    # and up) have a tighter stability radius than Störmer–Verlet (the
    # baseline of the `recommended_dt` formula), and the power-iteration
    # estimate of `|λ_max|` can underestimate the true value by a small
    # factor when the spectrum is clustered.
    dt  = W.recommended_dt(geom, ops, τ; cfl_safety = 0.5)
    alg = pick_integrator(N)
    println("integrator = $(typeof(alg).name.name)   τ = $τ   ",
            "dt = $(round(dt, sigdigits=4))   dx_min = $(round(dx, sigdigits=4))")

    bdry_values = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)    # homogeneous outer Dirichlet
    f!(ü, u̇, u, p, t) = W.rhs3d!(ü, u, u̇, bdry_values; geom, ops, τ)
    prob = SecondOrderODEProblem(f!, u̇, u, (t0, t1))

    # Initialise the integrator. We turn off all in-solver state storage —
    # the sampling loop below pulls `integrator.u` directly at each desired
    # time, so there is no need for `sol.u` to grow with each step.
    integrator = init(prob, alg; dt,
                      save_everystep = false,
                      save_start     = false,
                      save_end       = false,
                      dense          = false)

    # Spacetime slice along the x-axis at the (y_target, z_target) line.
    y_target = mesh_kind === :cubical ? 0.25 : 0.0
    z_target = mesh_kind === :cubical ? 0.25 : 0.0
    slice_idx, xs_line = build_slice(coords, y_target, z_target; atol = 1e-9)
    isempty(xs_line) && error("slice at y=$y_target, z=$z_target hit no GLL nodes")

    Nt = 200                        # number of time samples
    ts = range(t0, t1, Nt)

    # Buffers (preallocated; the sample loop reuses them and does not allocate).
    Ns      = length(xs_line)
    us      = Array{Float64}(undef, Ns, Nt)
    u̇s      = Array{Float64}(undef, Ns, Nt)
    l2_err  = Vector{Float64}(undef, Nt)
    u_exact = similar(u)
    err_buf = similar(u)

    prog = Progress(length(ts);
                    desc = "Evolving + sampling (mesh=$(mesh_kind), τ=$τ): ",
                    barlen = 30, showspeed = true)
    for (n, t) in enumerate(ts)
        # Step the integrator forward to the next sample time.
        while integrator.t < t
            step!(integrator)
        end
        next!(prog)

        # `integrator.u` is the current `ArrayPartition([u̇; u])`. Read the
        # two partitions directly — no copy, no allocation.
        u̇_arr = integrator.u.x[1]
        u_arr  = integrator.u.x[2]
        @assert all(isfinite, u_arr) && all(isfinite, u̇_arr)

        # Spacetime slice along the x-axis at (y_target, z_target).
        for (p, (e, ii, jj, kk)) in enumerate(slice_idx)
            us[p, n] = u_arr[ii, jj, kk, e]
            u̇s[p, n] = u̇_arr[ii, jj, kk, e]
        end

        # Physical-mass-weighted L² norm of the error vs the analytic
        # sin·sin·sin·cos(ωt) eigenmode (which satisfies the wave equation
        # with homogeneous Dirichlet on either [0,1]³ or [-1,1]³). The error
        # is fully truncation/dispersion error from the discretisation.
        ct = cos(ω * t)
        @inbounds for e in 1:mesh.Ne, k in 1:N, j in 1:N, i in 1:N
            X = (coords[1, i, j, k, e] - x0) / L_
            Y = (coords[2, i, j, k, e] - x0) / L_
            Z = (coords[3, i, j, k, e] - x0) / L_
            u_exact[i, j, k, e] = A * sin(kx*X) * sin(ky*Y) * sin(kz*Z) * ct
        end
        err_buf .= u_arr .- u_exact
        l2_err[n] = W.discrete_l2_norm(err_buf, geom, ops)
    end
    finish!(prog)

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
    ax3 = Axis(fig[2, 1:4];
               title  = "Physical L² error vs analytic eigenmode  [$(mesh_kind)]",
               xlabel = "t", ylabel = "‖u_num − u_exact‖_{H_phys}")

    hm1 = heatmap!(ax1, xs_line, ts, us; colormap=:plasma)
    Colorbar(fig[1, 2], hm1)

    hm2 = heatmap!(ax2, xs_line, ts, u̇s; colormap=:plasma)
    Colorbar(fig[1, 4], hm2)

    lines!(ax3, ts, l2_err; linewidth = 2)

    display(fig)

    return fig
end

# Default run: cubical mesh, N = 5 GLL nodes per element, M = 8 elements per axis.
main(; mesh_kind = :inflated_cube, N = 5, M = 8, R = 0.1)

nothing
