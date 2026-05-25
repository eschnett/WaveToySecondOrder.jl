using CairoMakie
using KernelAbstractions
using OrdinaryDiffEqSymplecticRK
using ProgressMeter
using SixelTerm
using StaticArrays
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
function min_node_spacing(coords::AbstractArray{T}) where {T}
    h = typemax(T)
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
function build_slice(coords::AbstractArray{T}, y_target, z_target; atol) where {T}
    Ne = size(coords, 5)
    N  = size(coords, 2)
    idx_list = NTuple{4, Int}[]
    xs       = T[]
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
function main(; T::Type = Float64,
                backend = CPU(),
                mesh_kind::Symbol = :cubical,
                ic_kind::Symbol = :cartesian,
                N::Int = 5,
                M::Int = 8,
                R::Real  = 0.1,    # cubed_cube: inner-cube half-edge fraction
                L::Real  = 0.1,    # inflated_cube: inner-cube half-edge
                R1::Real = 0.3,    # inflated_cube: inflation→shell radius
                R2::Real = 1.0,    # inflated_cube: outer-sphere radius
                ic_wavenumber::Real = 3π,            # :cartesian IC: kx = ky = kz
                ic_radial_mode::Int  = 1,            # :radial IC: radial node index n ≥ 1
                ic_radius::Union{Nothing, Real} = nothing)  # :radial IC: sphere radius
                                                            # (defaults to half the bounding-box side)

    # When the caller asks for a non-CPU backend the geometry and state
    # are migrated to the device; only host-side analysis bits (slice
    # sampling, 2-D `interpolate_field` snapshot) ever materialise back
    # to host arrays via `Array(::AbstractGPUArray)`. Apple GPUs only
    # support `Float32`, so use `T = Float32, backend = MetalBackend()`.
    on_cpu = backend isa CPU
    on_cpu || T <: AbstractFloat ||
        error("non-CPU backend requires a floating-point T; got $T")

    elem = W.make_element(T, N)
    ops  = W.make_operators(elem)

    if mesh_kind === :cubical
        x0, x1 = zero(T), one(T)
        mesh = W.make_cubical_mesh(T, M, x0, x1)
    elseif mesh_kind === :cubed_cube
        x0, x1 = -one(T), one(T)
        mesh = W.make_cubed_cube_mesh(T, M, T(R))
    elseif mesh_kind === :inflated_cube
        # Bounding box is the smallest axis-aligned cube containing the
        # outer sphere |x| = R2. The IC and L²-error formulae below use
        # `(coords - x0) / L_ ∈ [0, 1]` as a normalised coordinate; the
        # `sin·sin·sin` analytic eigenmode does *not* satisfy the
        # outer-sphere Dirichlet BC, so the L²-error plot is informative
        # only as a measure of wave activity, not as a convergence
        # diagnostic.
        x0, x1 = -T(R2), T(R2)
        mesh = W.make_inflated_cube_mesh(T, T(L), T(R1), T(R2), M)
    else
        error("unknown mesh_kind: $mesh_kind")
    end

    # `geom_host` always lives on the CPU (vertex search,
    # `interpolate_field`, the IC loop and the per-sample slice
    # extraction all read its `coords` on the host). `geom` is what
    # the kernels see — equal to `geom_host` on CPU, a device-resident
    # copy on GPU.
    geom_host = W.make_geometry(mesh, elem)
    geom      = on_cpu ? geom_host : W.to_device(geom_host, backend)
    coords    = geom_host.coords                  # (3, N, N, N, Ne), host

    dx = min_node_spacing(coords)

    t0, t1 = zero(T), one(T) / 10
    L_ = x1 - x0

    # Initial-condition family. The IC is *independent* of the mesh:
    # any combination of `mesh_kind` × `ic_kind` is allowed. The
    # L²-error plot below is a true convergence diagnostic only when
    # the IC's natural boundary matches the mesh's outer boundary
    # (cartesian on cube domains, radial on the inflated cube);
    # otherwise it just tracks wave activity.
    if ic_kind === :cartesian
        # Standing-wave eigenmode on `[x0, x1]³`:
        #   u = A · sin(kx·X) sin(ky·Y) sin(kz·Z) · cos(ω t)
        # with `ω = √(3·k²) / (x1 − x0)`. Vanishes on cube faces.
        ic_k = T(ic_wavenumber)
        ic_ω = T(sqrt(3 * ic_wavenumber^2)) / L_
    elseif ic_kind === :radial
        # Spherically-symmetric Bessel eigenmode on a ball of radius
        # `ic_R` centred at the bounding-box centre:
        #   u = A · sinc(n · r / R) · cos(ω t),  ω = nπ / R.
        # Vanishes at `r = R`. Default `R` is half the bounding-box
        # side length — equals `R2` for the inflated cube and gives
        # the inscribed sphere for cube domains.
        ic_R = ic_radius === nothing ? L_ / 2 : T(ic_radius)
        ic_ω = T(ic_radial_mode) * T(π) / ic_R
        ic_k = ic_ω        # filler, only `k` for `Params3d` shape; unused by the radial IC
    else
        error("unknown ic_kind: $ic_kind (use :cartesian or :radial)")
    end

    # Bounding-box centre for the radial IC.
    ic_center = ((x0 + x1) / 2, (x0 + x1) / 2, (x0 + x1) / 2)

    # Bundle the system parameters. SIPG penalty rule of thumb scales
    # with the worst element aspect ratio:
    #
    #   :cubical       (AR ≈ 1)  → ~1.5·(N−1)²
    #   :cubed_cube    (AR ≲ 2)  → ~8·(N−1)²
    #   :inflated_cube (AR ≲ 10 on the outermost shell layer at r = R2)
    #                            → ~32·(N−1)²
    #
    # The inflated-cube value is empirical: a localised mode living on
    # the outermost shell elements has eigenvalue λ ≈ +30 000 at
    # τ = 8·(N−1)² with the default mesh constants and M = 8, growing
    # at ~177/unit time and tripping the integrator's `unstable_check`
    # before t = 0.15. Bumping to 32·(N−1)² drives that mode negative
    # at the cost of `dt ≈ 1/√τ → ~2×` smaller. See the discussion of
    # option (B) for a principled adaptive τ — TODO.
    τ_mesh = mesh_kind === :cubical       ? T(3//2) * (N-1)^2 :
             mesh_kind === :cubed_cube    ? T(8)    * (N-1)^2 :
             mesh_kind === :inflated_cube ? T(32)   * (N-1)^2 :
             error("unknown mesh_kind: $mesh_kind")
    params = W.Params3d(;
        A           = one(T),
        k           = (ic_k, ic_k, ic_k),
        ω           = ic_ω,
        τ           = τ_mesh,
        bdry_values = ntuple(_ -> zero(T), Val(6)),
    )

    # Build the IC into a host buffer, then `copyto!` to the device if
    # needed. The eigenmode broadcasts are GPU-compatible too, but
    # building on host keeps the seeding path uniform across backends.
    u_host = Array{T, 4}(undef, N, N, N, mesh.Ne)
    u̇_host = similar(u_host)
    if ic_kind === :cartesian
        W.eigenmode_cartesian!(u_host, u̇_host, coords, zero(T);
                                A = params.A,
                                kx = params.k[1], ky = params.k[2], kz = params.k[3],
                                ω = params.ω, x0 = x0, x1 = x1)
    else  # :radial
        W.eigenmode_radial!(u_host, u̇_host, coords, zero(T);
                             A = params.A, R = ic_R, n = ic_radial_mode,
                             center = ic_center)
    end
    if on_cpu
        u, u̇ = u_host, u̇_host
    else
        u  = KernelAbstractions.allocate(backend, T, size(u_host)...)
        u̇  = KernelAbstractions.allocate(backend, T, size(u̇_host)...)
        copyto!(u,  u_host)
        copyto!(u̇, u̇_host)
    end

    # `cfl_safety = 0.5` (vs the default 0.9) gives a margin for two
    # things at once: the higher-stage symplectic integrators (CandyRoz4
    # and up) have a tighter stability radius than Störmer–Verlet (the
    # baseline of the `recommended_dt` formula), and the power-iteration
    # estimate of `|λ_max|` can underestimate the true value by a small
    # factor when the spectrum is clustered.
    dt  = W.recommended_dt(geom, ops, params.τ; cfl_safety = T(1//2))
    cfl = dt / dx
    alg = pick_integrator(N)
    println("integrator = $(typeof(alg).name.name)   backend = $(typeof(backend).name.name)   ",
            "τ = $(params.τ)   ",
            "dt = $(round(dt, sigdigits=4))   dx_min = $(round(dx, sigdigits=4))   dt/dx_min = $(round(cfl, sigdigits=4))")

    f!(ü, u̇, u, p::W.Params3d, t) = W.rhs3d!(ü, u, u̇, p; geom, ops)
    prob = SecondOrderODEProblem(f!, u̇, u, (t0, t1), params)

    # Initialise the integrator. We turn off all in-solver state storage —
    # the sampling loop below pulls `integrator.u` directly at each desired
    # time, so there is no need for `sol.u` to grow with each step.
    integrator = init(prob, alg; dt,
                      save_everystep = false,
                      save_start     = false,
                      save_end       = false,
                      dense          = false)

    # Spacetime slice along the x-axis at the (y_target, z_target) line.
    y_target = mesh_kind === :cubical ? T(1//4) : zero(T)
    z_target = mesh_kind === :cubical ? T(1//4) : zero(T)
    slice_idx, xs_line = build_slice(coords, y_target, z_target;
                                     atol = sqrt(eps(T)))
    isempty(xs_line) && error("slice at y=$y_target, z=$z_target hit no GLL nodes")

    Nt = 200                        # number of time samples
    ts = range(t0, t1, Nt)

    # Buffers (preallocated; the sample loop reuses them and does not allocate).
    Ns      = length(xs_line)
    us      = Array{T}(undef, Ns, Nt)
    u̇s      = Array{T}(undef, Ns, Nt)
    l2_err  = Vector{T}(undef, Nt)
    u_exact  = similar(u)              # same backend as state
    u̇_exact = similar(u)              # filled by the eigenmode! call, otherwise unused
    err_buf  = similar(u)
    # Slice / finiteness / scalar-index work happens on host. We always
    # `copyto!` from the integrator's current state into these buffers —
    # even on CPU, because the symplectic-RK solver allocates its own
    # `ArrayPartition` and the input `u`/`u̇` don't alias `integrator.u`.
    # (Trying to alias `u_host` instead of copying here gives a slice that
    # is stuck at `t = 0` while the L²-error norm of `integrator.u` evolves
    # correctly — confusing visuals, no error message.)
    u_arr_host  = Array{T, 4}(undef, N, N, N, mesh.Ne)
    u̇_arr_host  = Array{T, 4}(undef, N, N, N, mesh.Ne)

    prog = Progress(length(ts);
                    desc = "Evolving + sampling (mesh=$(mesh_kind), ic=$(ic_kind), τ=$(params.τ)): ",
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

        # Bring slice / scalar-index work to host. On CPU this is a
        # host→host copy from the integrator's state; on GPU it's a
        # small device→host transfer. Either way, cheap vs the per-step
        # RHS evaluations.
        copyto!(u_arr_host,  u_arr)
        copyto!(u̇_arr_host, u̇_arr)
        @assert all(isfinite, u_arr_host) && all(isfinite, u̇_arr_host)

        # Spacetime slice along the x-axis at (y_target, z_target).
        for (p, (e, ii, jj, kk)) in enumerate(slice_idx)
            us[p, n] = u_arr_host[ii, jj, kk, e]
            u̇s[p, n] = u̇_arr_host[ii, jj, kk, e]
        end

        # Physical-mass-weighted L² norm of the error vs the analytic
        # eigenmode at time `t`. When the IC family matches the mesh's
        # outer boundary (Dirichlet exactly), this is a true
        # truncation/dispersion error; otherwise it tracks wave activity.
        # The eigenmode! call broadcasts over `geom.coords`, so it works
        # on host and device backends without extra copies.
        if ic_kind === :cartesian
            W.eigenmode_cartesian!(u_exact, u̇_exact, geom.coords, t;
                                    A = params.A,
                                    kx = params.k[1], ky = params.k[2], kz = params.k[3],
                                    ω = params.ω, x0 = x0, x1 = x1)
        else  # :radial
            W.eigenmode_radial!(u_exact, u̇_exact, geom.coords, t;
                                 A = params.A, R = ic_R, n = ic_radial_mode,
                                 center = ic_center)
        end
        err_buf .= u_arr .- u_exact
        l2_err[n] = W.discrete_l2_norm(err_buf, geom, ops)
    end
    finish!(prog)

    # 2D slice of the final-time solution on the z = 0 plane.
    # `interpolate_field` does Newton iteration on the trilinear element
    # map per query point; the brute-force element search makes the loop
    # `O(Ng² · Ne)`. For 120² and Ne ≲ 5000 it takes a couple of seconds.
    Ng     = 120
    xs_xy  = range(x0, x1; length = Ng)
    ys_xy  = range(x0, x1; length = Ng)
    pts_xy = [SVector{3,T}(x, y, zero(T)) for x in xs_xy, y in ys_xy]
    # `interpolate_field` is host-only (Newton iteration with brute-
    # force vertex search). Materialise `u_final` on host first if it
    # lives on a device.
    u_final = on_cpu ? integrator.u.x[2] : Array(integrator.u.x[2])
    u_xy    = W.interpolate_field(mesh, elem, u_final, pts_xy)

    # Element-boundary segments lying on z = 0: collect the 12 edges per
    # element and keep only those whose endpoint vertices both sit on the
    # plane (Δz < tol). For meshes where the layer at z = 0 isn't a vertex
    # layer (e.g. the cubed cube's ±z patches), nothing is drawn for
    # those patches, which is the correct behaviour — the plane cuts
    # through their interiors, not along an element edge.
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

    # Figure
    fig = Figure(; size = (800, 1000))

    plot_tag = "mesh=$(mesh_kind), ic=$(ic_kind)"
    slice_label = "y=$(round(y_target; digits=3)), z=$(round(z_target; digits=3))"
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

    hm1 = heatmap!(ax1, xs_line, ts, us; colormap = :plasma)
    Colorbar(fig[1, 2], hm1)

    hm2 = heatmap!(ax2, xs_line, ts, u̇s; colormap = :plasma)
    Colorbar(fig[1, 4], hm2)

    lines!(ax3, ts, l2_err; linewidth = 2)

    umax_slice = maximum(abs, filter(isfinite, u_xy))
    hm4 = heatmap!(ax4, xs_xy, ys_xy, u_xy;
                   colormap   = :plasma,
                   colorrange = (-umax_slice, umax_slice))
    linesegments!(ax4, edge_x, edge_y; color = (:black, 0.6), linewidth = 0.6)
    Colorbar(fig[3, 4], hm4)

    display(fig)

    return fig
end

# Default run: cubical mesh, Cartesian sin·sin·sin eigenmode IC, on CPU.
# `mesh_kind` and `ic_kind` are independent — any pairing is allowed.
#
# Cubed cube + Cartesian IC (the cube outer boundary matches the IC's
# Dirichlet zeros, so the L² error is a convergence diagnostic):
#
#     main(; mesh_kind = :cubed_cube, N = 4, M = 8, R = 0.1)
#
# Inflated cube + radial Bessel IC (the outer sphere matches the IC's
# Dirichlet zero at `r = R2`, so the L² error is again convergence,
# this time on the spherical domain):
#
#     main(; mesh_kind = :inflated_cube, ic_kind = :radial, N = 4, M = 8)
#
# Apple Silicon GPU run (Float32, MetalBackend):
#
#     using Metal
#     main(; T = Float32, backend = MetalBackend(),
#            mesh_kind = :inflated_cube, ic_kind = :radial, N = 4, M = 8)

# main(; N = 4, M = 8)
# main(; mesh_kind = :cubed_cube, N = 4, M = 8, R = 0.1)
# main(; mesh_kind = :inflated_cube, ic_kind = :radial, N = 4, M = 8)

main(; mesh_kind = :inflated_cube, ic_kind = :radial, N = 4, M = 8, L = 0.1, R1 = 0.3, R2 = 1.0)

nothing
