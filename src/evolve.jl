# High-level evolution drivers for the 1D / 2D / 3D wave equation.
#
# Each `evolve{1,2,3}d` function builds a mesh, a `MeshGeometry`
# (or `Domain1d` NamedTuple), a workspace, an integrator, and a set of
# analytic-IC closures, then runs a `SecondOrderODEProblem` forward
# while sampling:
#
#   * a 1-D spacetime slice `u(x, …, t)` of `u` and `u̇`,
#   * the physical-mass-weighted L² error vs the analytic eigenmode at
#     each sample time,
#   * the full-domain snapshot at `t = t1` (interpolated onto a uniform
#     grid for plotting).
#
# The returned `NamedTuple` is consumed by `bin/waveplot{1,2,3}d.jl` to
# assemble the figure. Equation-free output: no Makie / CairoMakie /
# SixelTerm dependency in the package proper; plotting stays in
# `bin/`.

################################################################################
# Shared helpers

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

# Smallest GLL-node spacing across the mesh, Euclidean. Handles
# curvilinear elements whose local axis 1 is not aligned with physical x.
function _min_node_spacing_3d(coords::AbstractArray{T}) where {T}
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

function _min_node_spacing_2d(coords::AbstractArray{T}) where {T}
    h = typemax(T)
    @inbounds for e in 1:size(coords, 4), j in 1:size(coords, 3),
                  i in 2:size(coords, 2)
        dxv = coords[1, i, j, e] - coords[1, i-1, j, e]
        dyv = coords[2, i, j, e] - coords[2, i-1, j, e]
        h = min(h, sqrt(dxv*dxv + dyv*dyv))
    end
    return h
end

# Locate GLL nodes whose (y, z) coordinates match a target line within
# tolerance. Returns the sorted-by-x list of `(e, i, j, k)` indices plus
# the corresponding x-coordinates. Duplicates from shared element faces
# are removed.
function _build_slice_3d(coords::AbstractArray{T}, y_target, z_target; atol) where {T}
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

function _build_slice_2d(coords::AbstractArray{T}, y_target; atol) where {T}
    Ne = size(coords, 4)
    N  = size(coords, 2)
    idx_list = NTuple{3, Int}[]
    xs       = T[]
    for e in 1:Ne, jj in 1:N, ii in 1:N
        y = coords[2, ii, jj, e]
        if abs(y - y_target) < atol
            x = coords[1, ii, jj, e]
            isnew = !any(x0 -> abs(x0 - x) < atol, xs)
            if isnew
                push!(idx_list, (e, ii, jj))
                push!(xs, x)
            end
        end
    end
    perm = sortperm(xs)
    return idx_list[perm], xs[perm]
end

################################################################################
# evolve1d

"""
    evolve1d(; T = Float64, N = 5, M = 32, x0 = 0, x1 = 1,
                ic_wavenumber = 2π, τ_mult = 3//2,
                t0 = 0, t1 = 1, Nt = 200, cfl_safety = 1//2) → NamedTuple

Run the 1D wave equation `u_tt = u_xx` on `[x0, x1]` with homogeneous
Dirichlet BC and a sin·cos analytic eigenmode IC. Returns a NamedTuple
holding:

* `ts :: AbstractRange` — sample times.
* `xs_line :: Vector{T}` — physical x-coordinates of the GLL nodes
  along the 1D mesh (linearised).
* `us, u̇s :: Matrix{T}` of shape `(N·M, Nt)` — spacetime sample of
  the state.
* `l2_err :: Vector{T}` — physical-L² error at each sample time.
* `u_final, u̇_final :: Matrix{T}` of shape `(N, M)` — final-time state.
* `dom`, `elem`, `ops` — operator-level handles (for further analysis).
* `params :: Params1d{T}` — the wave-equation parameter bundle used.
* `x0, x1, dt, dx` — scalars echoed back for the plot title.
"""
function evolve1d(; T::Type = Float64,
                    backend = CPU(),
                    N::Int = 5,
                    M::Int = 32,
                    x0::Real = 0,
                    x1::Real = 1,
                    ic_wavenumber::Real = 2π,
                    τ_mult::Real = 3//2,
                    t0::Real = 0,
                    t1::Real = 1,
                    Nt::Int = 200,
                    cfl_safety::Real = 1//2)

    on_cpu = backend isa CPU
    on_cpu || T <: AbstractFloat ||
        error("evolve1d: non-CPU backend requires a floating-point T; got $T")

    elem = make_element(T, N)
    ops  = make_operators(elem)
    dom  = make_domain(T, M, T(x0), T(x1))

    # Element-shaped (N, M) physical-coordinate grid. Each column holds
    # one element's N local node positions in physical space —
    # `elem.xs ∈ [0, 1]` scaled to element width `dom.h` and offset by
    # the element's left edge `dom.xs[m]`.
    x_grid = T[dom.xs[m] + dom.h * xn for xn in elem.xs, m in 1:M]
    dx     = dom.h * elem.h
    L_     = T(x1) - T(x0)

    ic_k = T(ic_wavenumber)
    ic_ω = ic_k                              # 1D dispersion: ω = k

    params = Params1d(; A = one(T),
                        k = ic_k,
                        ω = ic_ω,
                        τ = T(τ_mult) * (N - 1)^2,
                        bL = zero(T), bR = zero(T))

    # `x_grid` lives on the chosen backend so the analytic-IC
    # broadcast `sin(k·x_grid)·cos(ω·t)` runs on-device.
    if on_cpu
        x_grid_dev = x_grid
    else
        x_grid_dev = KernelAbstractions.allocate(backend, T, N, M)
        copyto!(x_grid_dev, x_grid)
    end

    # Build IC directly on the backend.
    u  = on_cpu ? Array{T, 2}(undef, N, M) :
                  KernelAbstractions.allocate(backend, T, N, M)
    u̇  = similar(u)
    initialize!(u, u̇, x_grid_dev, zero(T);
                 A = params.A, k = params.k, ω = params.ω)

    dt  = recommended_dt(dom, ops, params.τ; cfl_safety = T(cfl_safety))
    alg = pick_integrator(N)

    f!(ü, u̇, u, p::Params1d, t) = rhs_wave1d!(ü, u, u̇, p; dom, ops)
    prob = SecondOrderODEProblem(f!, u̇, u, (T(t0), T(t1)), params)

    integrator = init(prob, alg; dt,
                      save_everystep = false,
                      save_start     = false,
                      save_end       = false,
                      dense          = false)

    ts = range(T(t0), T(t1), Nt)
    # 1D state laid out as `(N, M)` matrix; linearise for the spacetime
    # heatmap so the x-axis sweeps left-to-right across all elements.
    Ns       = N * M
    xs_line  = vec(x_grid)
    perm     = sortperm(xs_line)
    xs_line  = xs_line[perm]
    us       = Array{T}(undef, Ns, Nt)
    u̇s       = Array{T}(undef, Ns, Nt)
    l2_err   = Vector{T}(undef, Nt)
    # Analytic-reference buffers live on the same backend as the state
    # so the broadcast `err_buf .= u - u_exact` plus the sum-reduction
    # both run on-device when applicable.
    u_exact  = similar(u)
    u̇_exact  = similar(u)
    err_buf  = similar(u)
    # Host scratch — used to copy the spacetime slice back from device
    # before scalar-index sampling.
    u_arr_host  = Array{T, 2}(undef, N, M)
    u̇_arr_host  = Array{T, 2}(undef, N, M)
    # Quadrature weight per node = H_ref[i] · dom.h. Built once.
    # On device, migrate so the L²-error `mapreduce` stays on-backend.
    H_ref      = SVector{N, T}(ntuple(i -> ops.H[i, i], Val(N)))
    w_node_host = Array{T, 2}(undef, N, M)
    @inbounds for m in 1:M, i in 1:N
        w_node_host[i, m] = H_ref[i] * dom.h
    end
    if on_cpu
        w_node = w_node_host
    else
        w_node = KernelAbstractions.allocate(backend, T, N, M)
        copyto!(w_node, w_node_host)
    end

    prog = Progress(Nt;
                    desc = "evolve1d (N=$N, M=$M, backend=$(typeof(backend).name.name), τ=$(params.τ)): ",
                    barlen = 30, showspeed = true)
    for (n, t) in enumerate(ts)
        while integrator.t < t
            step!(integrator)
        end
        next!(prog)

        u̇_arr = integrator.u.x[1]
        u_arr  = integrator.u.x[2]

        copyto!(u_arr_host,  u_arr)
        copyto!(u̇_arr_host, u̇_arr)
        @assert all(isfinite, u_arr_host) && all(isfinite, u̇_arr_host)

        u_lin  = vec(u_arr_host)[perm]
        u̇_lin  = vec(u̇_arr_host)[perm]
        @inbounds for p in 1:Ns
            us[p, n]  = u_lin[p]
            u̇s[p, n] = u̇_lin[p]
        end

        # Physical-L² error using GLL quadrature weights. Works on
        # device because `u_exact` lives on the same backend and
        # `mapreduce` is GPU-portable through GPUArrays.
        initialize!(u_exact, u̇_exact, x_grid_dev, t;
                     A = params.A, k = params.k, ω = params.ω)
        err_buf .= u_arr .- u_exact
        l2_err[n] = sqrt(mapreduce((e, w) -> e * e * w, +, err_buf, w_node;
                                    init = zero(T)))
    end
    finish!(prog)

    u_final  = copy(u_arr_host)
    u̇_final  = copy(u̇_arr_host)

    return (; ts, xs_line, us, u̇s, l2_err,
              u_final, u̇_final,
              dom, elem, ops, params,
              x0 = T(x0), x1 = T(x1), dt, dx,
              integrator_name = nameof(typeof(alg)))
end

################################################################################
# evolve2d

"""
    evolve2d(; T = Float64, backend = CPU(), mesh_kind = :cubical,
                ic_kind = :cartesian, N = 5, M = 8,
                R = 0.1, L = 0.1, R1 = 0.3, R2 = 1.0,
                ic_wavenumber = 3π, ic_radial_mode = 1, ic_radius = nothing,
                outer_bc = :dirichlet,
                t0 = 0, t1 = 1, Nt = 200, cfl_safety = 1//2,
                slice_y = nothing) → NamedTuple

2D wave-equation driver. `mesh_kind ∈ {:cubical, :cubed_square,
:inflated_square}`; `ic_kind ∈ {:cartesian, :radial, :outgoing}`;
`outer_bc ∈ {:dirichlet, :sommerfeld}` (only valid on
`:inflated_square`). Returns the sampled spacetime slice, the L² error
trace, and the final-time snapshot for downstream plotting.

`:outgoing` uses the Hankel-transform Gaussian-pulse solution from
[`outgoing_pulse_2d!`](@ref) — the closest analytic analog to a
smooth, localized, outgoing radial wave that exists in 2D. The
Gaussian width is controlled by `ic_pulse_width` (default
`L_/12` where `L_` is the bounding-box side); the quadrature order
of the Hankel integral is `ic_pulse_n_quad` (default `128`, accurate
to ~14 digits for `t · σ ≲ 10`). The pulse spreads outward but
leaves a wake — the 2D wave equation isn't Huygens. Pair with
`outer_bc = :sommerfeld` to absorb the leading edge at the outer
circle.

With `outer_bc = :sommerfeld` the outer-circle faces are tagged `7`
and `rhs_wave2d!`'s post-pass adds the BGT-0 (plane-wave) dissipative
drag. See the docstring on `rhs_wave2d!` for the 2D-specific physics
caveat (no exact BGT-1 in 2D).

Returned NamedTuple keys mirror `evolve3d` minus the `z_target` /
`sommerfeld_R` triple (2D has `sommerfeld_R` too).
"""
function evolve2d(; T::Type = Float64,
                    backend = CPU(),
                    mesh_kind::Symbol = :cubical,
                    ic_kind::Symbol = :cartesian,
                    N::Int = 5,
                    M::Int = 8,
                    R::Real  = 0.1,
                    L::Real  = 0.1,
                    R1::Real = 0.3,
                    R2::Real = 1.0,
                    ic_wavenumber::Real = 3π,
                    ic_radial_mode::Int  = 1,
                    ic_radius::Union{Nothing, Real} = nothing,
                    ic_pulse_width::Union{Nothing, Real} = nothing,
                    ic_pulse_n_quad::Int = 128,
                    outer_bc::Symbol = :dirichlet,
                    t0::Real = 0,
                    t1::Real = 1,
                    Nt::Int = 200,
                    cfl_safety::Real = 1//2,
                    slice_y::Union{Nothing, Real} = nothing)

    on_cpu = backend isa CPU
    on_cpu || T <: AbstractFloat ||
        error("non-CPU backend requires a floating-point T; got $T")
    if outer_bc !== :dirichlet && mesh_kind !== :inflated_square
        error("evolve2d: outer_bc = :$outer_bc only supported on mesh_kind = :inflated_square")
    end

    elem = make_element(T, N)
    ops  = make_operators(elem)

    if mesh_kind === :cubical
        x0, x1 = zero(T), one(T)
        mesh = make_uniform_quad(T, M, x0, x1)
    elseif mesh_kind === :cubed_square
        x0, x1 = -one(T), one(T)
        mesh = make_cubed_square_mesh(T, M, T(R))
    elseif mesh_kind === :inflated_square
        x0, x1 = -T(R2), T(R2)
        mesh = make_inflated_square_mesh(T, T(L), T(R1), T(R2), M; outer_bc)
    else
        error("evolve2d: unknown mesh_kind: $mesh_kind (use :cubical, :cubed_square, :inflated_square)")
    end

    geom_host = make_geometry(mesh, elem)
    geom      = on_cpu ? geom_host : to_device(geom_host, backend)
    work      = make_workspace(geom)
    coords    = geom_host.coords

    dx = _min_node_spacing_2d(coords)
    L_ = x1 - x0

    # Build IC parameters.
    ic_center = ((x0 + x1) / 2, (x0 + x1) / 2)
    if ic_kind === :cartesian
        ic_k = T(ic_wavenumber)
        ic_ω = T(sqrt(2 * ic_wavenumber^2)) / L_
        ic_R = zero(T)
        ic_σ = zero(T)
    elseif ic_kind === :radial
        ic_R = ic_radius === nothing ? L_ / 2 : T(ic_radius)
        ic_ω = T(WaveToySecondOrder._J0_ZEROS[ic_radial_mode]) / ic_R
        ic_k = ic_ω
        ic_σ = zero(T)
    elseif ic_kind === :outgoing
        # Default Gaussian width: bounding-box side / 12. On the
        # `:inflated_square` mesh (L_ = 2 R2) this gives σ = R2/6,
        # which puts the FWHM (≈ 2.35 σ) at ≈ 0.4 R2 — well-localized
        # near the origin yet not so sharp that 128-node Gauss-
        # Legendre under-resolves the integrand.
        ic_σ = ic_pulse_width === nothing ? L_ / 12 : T(ic_pulse_width)
        ic_k = zero(T); ic_ω = zero(T); ic_R = zero(T)
    else
        error("evolve2d: unknown ic_kind: $ic_kind (use :cartesian, :radial, or :outgoing)")
    end

    # Cache the Hankel-transform Bessel table once if we'll be sampling
    # the analytic `:outgoing` reference at every step. `nothing` for
    # other IC families avoids the ~MB allocation when it isn't needed.
    pulse_cache = ic_kind === :outgoing ?
        outgoing_pulse_2d_cache(coords; σ = ic_σ, center = ic_center,
                                 n_quad = ic_pulse_n_quad) :
        nothing

    # `sommerfeld_R = R2` on the inflated-square outer circle would
    # plug into the BGT-1 `+u/R` term, but 2D BGT-1 isn't exact (see
    # `rhs_wave2d!` docstring). Use `Inf` here so the post-pass runs as
    # plane-wave BGT-0, which is the safe default on a curved boundary
    # in 2D. The mesh still gets tagged `7` via `outer_bc`, which is
    # what triggers the dissipative kernel.
    sommerfeld_R = T(Inf)

    τ_mult = mesh_kind === :cubical ? T(3//2) : T(8)
    params = Params2d(; A = one(T),
                        k = (ic_k, ic_k),
                        ω = ic_ω,
                        τ = τ_mult * (N - 1)^2,
                        bdry_values = ntuple(_ -> zero(T), Val(4)),
                        sommerfeld_R = sommerfeld_R)

    # IC into host buffer, then `copyto!` to device.
    u_host = Array{T, 3}(undef, N, N, mesh.Ne)
    u̇_host = similar(u_host)
    if ic_kind === :cartesian
        eigenmode_cartesian_2d!(u_host, u̇_host, coords, zero(T);
                                 A = params.A,
                                 kx = params.k[1], ky = params.k[2],
                                 ω = params.ω, x0 = x0, x1 = x1)
    elseif ic_kind === :radial
        eigenmode_radial_2d!(u_host, u̇_host, coords, zero(T);
                              A = params.A, R = ic_R, n = ic_radial_mode,
                              center = ic_center)
    else  # :outgoing
        outgoing_pulse_2d!(u_host, u̇_host, pulse_cache, zero(T);
                            A = params.A)
    end
    if on_cpu
        u, u̇ = u_host, u̇_host
    else
        u  = KernelAbstractions.allocate(backend, T, size(u_host)...)
        u̇  = KernelAbstractions.allocate(backend, T, size(u̇_host)...)
        copyto!(u,  u_host)
        copyto!(u̇, u̇_host)
    end

    dt  = recommended_dt(geom, ops, params.τ; cfl_safety = T(cfl_safety))
    alg = pick_integrator(N)

    f!(ü, u̇, u, p::Params2d, t) = rhs_wave2d!(ü, u, u̇, p; geom, ops, work)
    prob = SecondOrderODEProblem(f!, u̇, u, (T(t0), T(t1)), params)
    integrator = init(prob, alg; dt,
                      save_everystep = false,
                      save_start     = false,
                      save_end       = false,
                      dense          = false)

    y_target = T(slice_y === nothing ?
                 (mesh_kind === :cubical ? 1//4 : 0) :
                 slice_y)
    slice_idx, xs_line = _build_slice_2d(coords, y_target; atol = sqrt(eps(T)))
    isempty(xs_line) && error("evolve2d: slice at y=$y_target hit no GLL nodes")

    ts       = range(T(t0), T(t1), Nt)
    Ns       = length(xs_line)
    us       = Array{T}(undef, Ns, Nt)
    u̇s       = Array{T}(undef, Ns, Nt)
    l2_err   = Vector{T}(undef, Nt)
    u_exact  = similar(u)
    u̇_exact  = similar(u)
    err_buf  = similar(u)
    u_arr_host = Array{T, 3}(undef, N, N, mesh.Ne)
    u̇_arr_host = Array{T, 3}(undef, N, N, mesh.Ne)

    prog = Progress(Nt;
                    desc = "evolve2d (mesh=$(mesh_kind), ic=$(ic_kind), τ=$(params.τ)): ",
                    barlen = 30, showspeed = true)
    for (n, t) in enumerate(ts)
        while integrator.t < t
            step!(integrator)
        end
        next!(prog)

        u̇_arr = integrator.u.x[1]
        u_arr  = integrator.u.x[2]

        copyto!(u_arr_host,  u_arr)
        copyto!(u̇_arr_host, u̇_arr)
        @assert all(isfinite, u_arr_host) && all(isfinite, u̇_arr_host)

        for (p, (e, ii, jj)) in enumerate(slice_idx)
            us[p, n] = u_arr_host[ii, jj, e]
            u̇s[p, n] = u̇_arr_host[ii, jj, e]
        end

        if ic_kind === :cartesian
            eigenmode_cartesian_2d!(u_exact, u̇_exact, geom.coords, t;
                                     A = params.A,
                                     kx = params.k[1], ky = params.k[2],
                                     ω = params.ω, x0 = x0, x1 = x1)
        elseif ic_kind === :radial
            eigenmode_radial_2d!(u_exact, u̇_exact, geom.coords, t;
                                  A = params.A, R = ic_R, n = ic_radial_mode,
                                  center = ic_center)
        else  # :outgoing
            outgoing_pulse_2d!(u_exact, u̇_exact, pulse_cache, t;
                                A = params.A)
        end
        err_buf .= u_arr .- u_exact
        l2_err[n] = discrete_l2_norm(err_buf, geom, ops)
    end
    finish!(prog)

    u_final = on_cpu ? copy(integrator.u.x[2]) : Array(integrator.u.x[2])

    return (; ts, xs_line, us, u̇s, l2_err,
              u_final,
              mesh, geom = geom_host, elem, ops, params,
              x0, x1, dt, dx, y_target,
              sommerfeld_R, ic_kind, mesh_kind, outer_bc,
              integrator_name = nameof(typeof(alg)))
end

################################################################################
# evolve3d

"""
    evolve3d(; T = Float64, backend = CPU(), mesh_kind = :cubical,
                ic_kind = :cartesian, N = 5, M = 8,
                R = 0.1, L = 0.1, R1 = 0.3, R2 = 1.0,
                ic_wavenumber = 3π, ic_radial_mode = 1,
                ic_radius = nothing,
                ic_pulse_offset = nothing, ic_pulse_width = nothing,
                outer_bc = :dirichlet,
                t0 = 0, t1 = 1, Nt = 200, cfl_safety = 1//2,
                slice_y = nothing, slice_z = nothing) → NamedTuple

3D wave-equation driver — moved out of `bin/waveplot3d.jl`. Supports
the three mesh families (`:cubical, :cubed_cube, :inflated_cube`), the
three IC families (`:cartesian, :radial, :outgoing`), and the
Sommerfeld outer BC option on `:inflated_cube`.

Returned NamedTuple keys mirror `evolve2d`'s plus `z_target` and
`sommerfeld_R`.
"""
function evolve3d(; T::Type = Float64,
                    backend = CPU(),
                    mesh_kind::Symbol = :cubical,
                    ic_kind::Symbol = :cartesian,
                    N::Int = 5,
                    M::Int = 8,
                    R::Real  = 0.1,
                    L::Real  = 0.1,
                    R1::Real = 0.3,
                    R2::Real = 1.0,
                    ic_wavenumber::Real = 3π,
                    ic_radial_mode::Int  = 1,
                    ic_radius::Union{Nothing, Real} = nothing,
                    ic_pulse_offset::Union{Nothing, Real} = nothing,
                    ic_pulse_width::Union{Nothing, Real} = nothing,
                    outer_bc::Symbol = :dirichlet,
                    t0::Real = 0,
                    t1::Real = 1,
                    Nt::Int = 200,
                    cfl_safety::Real = 1//2,
                    slice_y::Union{Nothing, Real} = nothing,
                    slice_z::Union{Nothing, Real} = nothing,
                    inner_bc::Symbol = :excision)

    on_cpu = backend isa CPU
    on_cpu || T <: AbstractFloat ||
        error("non-CPU backend requires a floating-point T; got $T")
    if outer_bc !== :dirichlet &&
       !(mesh_kind === :inflated_cube || mesh_kind === :radial_shell)
        error("evolve3d: outer_bc = :$outer_bc only supported on " *
              "mesh_kind ∈ (:inflated_cube, :radial_shell)")
    end

    elem = make_element(T, N)
    ops  = make_operators(elem)

    if mesh_kind === :cubical
        x0, x1 = zero(T), one(T)
        mesh = make_uniform_hex(T, M, x0, x1)
    elseif mesh_kind === :cubed_cube
        x0, x1 = -one(T), one(T)
        mesh = make_cubed_cube_mesh(T, M, T(R))
    elseif mesh_kind === :inflated_cube
        x0, x1 = -T(R2), T(R2)
        mesh = make_inflated_cube_mesh(T, T(L), T(R1), T(R2), M; outer_bc)
    elseif mesh_kind === :radial_shell
        # Pure 6-patch spherical shell R1 ≤ |x| ≤ R2 — for BH excision
        # (inner sphere R1 is the excision surface). Default
        # `inner_bc = :excision` triggers the no-SAT branch in
        # `wave_strong_rhs_element!`.
        x0, x1 = -T(R2), T(R2)
        mesh = make_radial_shell_mesh(T, T(R1), T(R2), M;
                                        outer_bc, inner_bc)
    else
        error("evolve3d: unknown mesh_kind: $mesh_kind " *
              "(use :cubical, :cubed_cube, :inflated_cube, :radial_shell)")
    end

    geom_host = make_geometry(mesh, elem)
    geom      = on_cpu ? geom_host : to_device(geom_host, backend)
    work      = make_workspace(geom)
    coords    = geom_host.coords

    dx = _min_node_spacing_3d(coords)
    L_ = x1 - x0

    if ic_kind === :cartesian
        ic_k = T(ic_wavenumber)
        ic_ω = T(sqrt(3 * ic_wavenumber^2)) / L_
        ic_R = zero(T); ic_s0 = zero(T); ic_σ = zero(T)
    elseif ic_kind === :radial
        ic_R = ic_radius === nothing ? L_ / 2 : T(ic_radius)
        ic_ω = T(ic_radial_mode) * T(π) / ic_R
        ic_k = ic_ω
        ic_s0 = zero(T); ic_σ = zero(T)
    elseif ic_kind === :outgoing
        ic_s0 = ic_pulse_offset === nothing ? L_ / 4 : T(ic_pulse_offset)
        ic_σ  = ic_pulse_width  === nothing ? ic_s0 / 5 : T(ic_pulse_width)
        ic_k = zero(T); ic_ω = zero(T); ic_R = zero(T)
    else
        error("evolve3d: unknown ic_kind: $ic_kind (use :cartesian, :radial, or :outgoing)")
    end
    ic_center = ((x0 + x1) / 2, (x0 + x1) / 2, (x0 + x1) / 2)

    sommerfeld_R = (mesh_kind in (:inflated_cube, :radial_shell) &&
                     outer_bc === :sommerfeld) ?
                       T(R2) : T(Inf)
    τ_mult = mesh_kind === :cubical ? T(3//2) : T(8)
    params = Params3d(; A = one(T),
                        k = (ic_k, ic_k, ic_k),
                        ω = ic_ω,
                        τ = τ_mult * (N - 1)^2,
                        bdry_values = ntuple(_ -> zero(T), Val(6)),
                        sommerfeld_R = sommerfeld_R)

    u_host = Array{T, 4}(undef, N, N, N, mesh.Ne)
    u̇_host = similar(u_host)
    if ic_kind === :cartesian
        eigenmode_cartesian!(u_host, u̇_host, coords, zero(T);
                              A = params.A,
                              kx = params.k[1], ky = params.k[2], kz = params.k[3],
                              ω = params.ω, x0 = x0, x1 = x1)
    elseif ic_kind === :radial
        eigenmode_radial!(u_host, u̇_host, coords, zero(T);
                           A = params.A, R = ic_R, n = ic_radial_mode,
                           center = ic_center)
    else  # :outgoing
        outgoing_pulse!(u_host, u̇_host, coords, zero(T);
                         A = params.A, s0 = ic_s0, σ = ic_σ,
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

    dt  = recommended_dt(geom, ops, params.τ; cfl_safety = T(cfl_safety))
    alg = pick_integrator(N)

    f!(ü, u̇, u, p::Params3d, t) = rhs_wave3d!(ü, u, u̇, p; geom, ops, work)
    prob = SecondOrderODEProblem(f!, u̇, u, (T(t0), T(t1)), params)
    integrator = init(prob, alg; dt,
                      save_everystep = false,
                      save_start     = false,
                      save_end       = false,
                      dense          = false)

    y_target = T(slice_y === nothing ? (mesh_kind === :cubical ? 1//4 : 0) : slice_y)
    z_target = T(slice_z === nothing ? (mesh_kind === :cubical ? 1//4 : 0) : slice_z)
    slice_idx, xs_line = _build_slice_3d(coords, y_target, z_target;
                                          atol = sqrt(eps(T)))
    isempty(xs_line) && error("evolve3d: slice at y=$y_target, z=$z_target hit no GLL nodes")

    ts       = range(T(t0), T(t1), Nt)
    Ns       = length(xs_line)
    us       = Array{T}(undef, Ns, Nt)
    u̇s       = Array{T}(undef, Ns, Nt)
    l2_err   = Vector{T}(undef, Nt)
    u_exact  = similar(u)
    u̇_exact  = similar(u)
    err_buf  = similar(u)
    u_arr_host = Array{T, 4}(undef, N, N, N, mesh.Ne)
    u̇_arr_host = Array{T, 4}(undef, N, N, N, mesh.Ne)

    prog = Progress(Nt;
                    desc = "evolve3d (mesh=$(mesh_kind), ic=$(ic_kind), bc=$(outer_bc), τ=$(params.τ)): ",
                    barlen = 30, showspeed = true)
    for (n, t) in enumerate(ts)
        while integrator.t < t
            step!(integrator)
        end
        next!(prog)

        u̇_arr = integrator.u.x[1]
        u_arr  = integrator.u.x[2]

        copyto!(u_arr_host,  u_arr)
        copyto!(u̇_arr_host, u̇_arr)
        @assert all(isfinite, u_arr_host) && all(isfinite, u̇_arr_host)

        for (p, (e, ii, jj, kk)) in enumerate(slice_idx)
            us[p, n] = u_arr_host[ii, jj, kk, e]
            u̇s[p, n] = u̇_arr_host[ii, jj, kk, e]
        end

        if ic_kind === :cartesian
            eigenmode_cartesian!(u_exact, u̇_exact, geom.coords, t;
                                  A = params.A,
                                  kx = params.k[1], ky = params.k[2], kz = params.k[3],
                                  ω = params.ω, x0 = x0, x1 = x1)
        elseif ic_kind === :radial
            eigenmode_radial!(u_exact, u̇_exact, geom.coords, t;
                               A = params.A, R = ic_R, n = ic_radial_mode,
                               center = ic_center)
        else
            outgoing_pulse!(u_exact, u̇_exact, geom.coords, t;
                             A = params.A, s0 = ic_s0, σ = ic_σ,
                             center = ic_center)
        end
        err_buf .= u_arr .- u_exact
        l2_err[n] = discrete_l2_norm(err_buf, geom, ops)
    end
    finish!(prog)

    u_final = on_cpu ? copy(integrator.u.x[2]) : Array(integrator.u.x[2])

    return (; ts, xs_line, us, u̇s, l2_err,
              u_final,
              mesh, geom = geom_host, elem, ops, params,
              x0, x1, dt, dx, y_target, z_target,
              sommerfeld_R, ic_kind, mesh_kind, outer_bc,
              integrator_name = nameof(typeof(alg)))
end
