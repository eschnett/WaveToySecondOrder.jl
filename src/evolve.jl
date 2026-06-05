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
# only as accurate as the space scheme. Used by the 2D/3D
# `SecondOrderODEProblem` drivers.
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

# Explicit-RK pick for the first-order ADM system (1D driver). The
# variable-β system is not Hamiltonian, so symplectic integrators are
# not appropriate; the spatial operator is (nearly) skew, so we need
# explicit RK schemes whose stability region covers a stretch of the
# imaginary axis. Order again matches the spatial order `N − 1`.
function pick_integrator_first_order(N::Integer)
    if     N ≤ 4;  return RK4()     # classic 4th-order
    elseif N ≤ 6;  return Tsit5()   # 5th-order
    else           return Vern7()   # 7th-order
    end
end

# Smallest GLL-node spacing across the mesh, Euclidean. Handles
# curvilinear elements whose local axis 1 is not aligned with physical x.
@inline _node_dist3(c, a, b) =
    sqrt((c[1,a...] - c[1,b...])^2 + (c[2,a...] - c[2,b...])^2 +
         (c[3,a...] - c[3,b...])^2)

# Smallest physical spacing between reference-axis-adjacent nodes, taken
# over ALL reference directions (ξ, η, ζ) — not just axis 1. On curved
# meshes the angular spacing can be smaller than the radial one, so an
# axis-1-only minimum overestimates h and yields a too-large CFL dt.
function _min_node_spacing_3d(coords::AbstractArray{T}) where {T}
    N1, N2, N3, Ne = size(coords, 2), size(coords, 3), size(coords, 4), size(coords, 5)
    h = typemax(T)
    @inbounds for e in 1:Ne, k in 1:N3, j in 1:N2, i in 1:N1
        i > 1 && (h = min(h, _node_dist3(coords, (i,j,k,e), (i-1,j,k,e))))
        j > 1 && (h = min(h, _node_dist3(coords, (i,j,k,e), (i,j-1,k,e))))
        k > 1 && (h = min(h, _node_dist3(coords, (i,j,k,e), (i,j,k-1,e))))
    end
    return h
end

# (i, j, e) of a representative node on boundary side `f` (1=−x, 2=+x,
# 3=−y, 4=+y) of an axis-aligned mesh — the first boundary element on
# that side, at the side's row and the mid tangential node. Used to
# classify each side from a node that actually lies on it.
function _side_node_2d(geom, f, N)
    bdry = geom.conn.bdry
    a_idx = (f + 1) ÷ 2
    row = isodd(f) ? 1 : N
    mid = (N + 1) ÷ 2
    @inbounds for e in 1:geom.Ne
        bdry[f, e] == 0 && continue
        return a_idx == 1 ? (row, mid, e) : (mid, row, e)
    end
    return (1, 1, 1)
end

@inline _node_dist2(c, a, b) =
    sqrt((c[1,a...] - c[1,b...])^2 + (c[2,a...] - c[2,b...])^2)

function _min_node_spacing_2d(coords::AbstractArray{T}) where {T}
    N1, N2, Ne = size(coords, 2), size(coords, 3), size(coords, 4)
    h = typemax(T)
    @inbounds for e in 1:Ne, j in 1:N2, i in 1:N1
        i > 1 && (h = min(h, _node_dist2(coords, (i,j,e), (i-1,j,e))))
        j > 1 && (h = min(h, _node_dist2(coords, (i,j,e), (i,j-1,e))))
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

# Built-in 1D backgrounds with their exact scalar-wave solutions
# (used as IC, as the L²-error reference, and as boundary data). Each
# entry returns `(bg :: Background1D, Φ_exact(t, x), Π_exact(t, x),
# DΦ_exact(t, x), max_speed)`. `Π = (√γ/α)(∂_t Φ − β ∂_x Φ)`;
# `DΦ = ∂_x Φ` (needed to assemble characteristic boundary data);
# `max_speed = max |β| + α/√γ` bounds the coordinate characteristic
# speeds `−β ± α/√γ`.
function _background1d(kind::Symbol, ::Type{T};
                       A::Real, d::Real, shift::Real,
                       k_w::Real) where {T}
    k₀ = T(k_w)
    if kind === :minkowski || kind === :constant_shift
        β₀ = kind === :minkowski ? zero(T) : T(shift)
        bg = AnalyticBackground1D((t, x) -> one(typeof(x)),
                                  _ConstFn(β₀),
                                  (t, x) -> one(typeof(x)))
        # Right-mover Φ = sin(k(x − c₊ t)), c₊ = 1 − β. With α = γ = 1:
        # Π = ∂_t Φ − β ∂_x Φ = −k cos(k(x − c₊ t)).
        c₊ = one(T) - β₀
        Φe = (t, x) -> sin(k₀ * (x - c₊ * t))
        Πe = (t, x) -> -k₀ * cos(k₀ * (x - c₊ * t))
        De = (t, x) -> k₀ * cos(k₀ * (x - c₊ * t))
        return bg, Φe, Πe, De, abs(β₀) + one(T)
    elseif kind === :gaugewave
        # AwA gauge wave: α = √H, β = 0, γ_xx = H. Exact solution
        # Φ = sin(k₀(x̂ − t̂)) with x̂ − t̂ = x − t + 2C cos(2π(x−t)/d);
        # ∂_x(x̂ − t̂) = 1 − A sin(2π(x−t)/d) = H.
        Aᵥ, dᵥ = T(A), T(d)
        kᵥ = 2 * T(π) / dᵥ
        C = Aᵥ * dᵥ / (4 * T(π))
        bg = MetricBackground1D(SpacetimeMetrics.GaugeWave(Aᵥ, dᵥ))
        ψ = (t, x) -> x - t + 2C * cos(kᵥ * (x - t))
        Φe = (t, x) -> sin(k₀ * ψ(t, x))
        Πe = (t, x) -> -k₀ * (1 - Aᵥ * sin(kᵥ * (x - t))) * cos(k₀ * ψ(t, x))
        De = (t, x) -> k₀ * (1 - Aᵥ * sin(kᵥ * (x - t))) * cos(k₀ * ψ(t, x))
        return bg, Φe, Πe, De, one(T)        # α/√γ = 1, β = 0
    elseif kind === :sineshift
        # Sine shift: α = 1, β = −Ac/(1+Ac), γ_xx = (1+Ac)²,
        # c = cos(2π(x−t)/d). Exact Φ = sin(k₀ ψ), ψ = x + C sin(…) − t;
        # ∂_xψ = 1 + A cos(2π(x−t)/d) = √γ.
        Aᵥ, dᵥ = T(A), T(d)
        kᵥ = 2 * T(π) / dᵥ
        C = Aᵥ * dᵥ / (2 * T(π))
        bg = MetricBackground1D(SpacetimeMetrics.SineShift(Aᵥ, dᵥ))
        ψ = (t, x) -> x + C * sin(kᵥ * (x - t)) - t
        Φe = (t, x) -> sin(k₀ * ψ(t, x))
        Πe = (t, x) -> -k₀ * (1 + Aᵥ * cos(kᵥ * (x - t))) * cos(k₀ * ψ(t, x))
        De = (t, x) -> k₀ * (1 + Aᵥ * cos(kᵥ * (x - t))) * cos(k₀ * ψ(t, x))
        # max |β| + α/√γ = A/(1−A) + 1/(1−A).
        return bg, Φe, Πe, De, (Aᵥ + 1) / (1 - Aᵥ)
    else
        error("evolve1d: unknown background $kind " *
              "(expected :minkowski, :constant_shift, :gaugewave, :sineshift)")
    end
end

# ADM coefficients (a = α/√γ, β) of a Background1D at a single point —
# host-side helper for boundary-face classification and data assembly.
function _bg_point(bg::Background1D, t, x)
    α, β, γ = WaveToySecondOrder._bg_adm(bg, t, x)
    sγ = sqrt(γ)
    return α / sγ, β
end

# Per-stage boundary bundle for `evolve1d`: classify both faces from
# the background at time `t` (must match the setup-time classes —
# time-dependent backgrounds may not change a face's characteristic
# class mid-run), then assemble the scalar data from the exact-solution
# closures (`g ≡ 0` for noise runs).
function _assemble_bc1d(bg, t, xL, xR, kindL, kindR, classL0, classR0,
                        Φe, Πe, De, withdata::Bool, ::Type{T}) where {T}
    aL, βL = _bg_point(bg, t, xL)
    aR, βR = _bg_point(bg, t, xR)
    classL = classify_face1d(aL, βL, -1)
    classR = classify_face1d(aR, βR, +1)
    (classL == classL0 && classR == classR0) ||
        throw(ArgumentError("evolve1d: a boundary face changed its " *
            "characteristic class at t = $t (left: " *
            "$(WaveToySecondOrder._face_class_name(classL0)) → " *
            "$(WaveToySecondOrder._face_class_name(classL)), right: " *
            "$(WaveToySecondOrder._face_class_name(classR0)) → " *
            "$(WaveToySecondOrder._face_class_name(classR))); the " *
            "chosen boundary conditions are no longer admissible"))

    # Dirichlet data slot is the field-radiation residual evaluated on
    # the exact solution, r = Π + (n̂ + β/a)·∂_xΦ (matches the kernel's
    # `r`); driving the kernel residual to it injects the exact
    # incoming wave while leaving outgoing waves free.
    g_in(x, a, β, n̂) = !withdata ? zero(T) :
        T(Πe(t, x) + (n̂ + β / a) * De(t, x))

    g1L = kindL == BC_DIRICHLET      ? g_in(xL, aL, βL, -1) :
          kindL == BC_FULL_DIRICHLET ? (withdata ? T(Φe(t, xL)) : zero(T)) :
          zero(T)
    g2L = kindL == BC_FULL_DIRICHLET ? (withdata ? T(Πe(t, xL)) : zero(T)) :
          zero(T)
    g1R = kindR == BC_DIRICHLET      ? g_in(xR, aR, βR, +1) :
          kindR == BC_FULL_DIRICHLET ? (withdata ? T(Φe(t, xR)) : zero(T)) :
          zero(T)
    g2R = kindR == BC_FULL_DIRICHLET ? (withdata ? T(Πe(t, xR)) : zero(T)) :
          zero(T)
    return make_bc1d(kindL, kindR; g1L, g2L, g1R, g2R)
end

# Constant-value closure as a callable struct so the background stays
# isbits when captured into GPU kernels (a `let`-captured `T(shift)`
# closure would be fine too, but this is explicit).
struct _ConstFn{T}
    v :: T
end
(f::_ConstFn)(t, x) = f.v

"""
    evolve1d(; T = Float64, backend = CPU(), N = 4, M = 32,
               x0 = 0, x1 = 1,
               background = :sineshift, A = 0.3, d = 1, shift = 0.5,
               ic = :exact, ic_wavenumber = 2π, noise_amp = √eps,
               ε_KO = 0, t0 = 0, t1 = 1, Nt = 200,
               cfl = 1//10) → NamedTuple

Run the 1D scalar wave on a 1+1 ADM background (`wave1d_curved_rhs!`)
over the periodic interval `[x0, x1]`, integrating the first-order
(Φ, Π) system with an explicit RK scheme from OrdinaryDiffEq
(`pick_integrator_first_order(N)`; fixed CFL-derived `dt`).

* `background ∈ {:minkowski, :constant_shift, :gaugewave, :sineshift}`
  — built-in backgrounds with exact solutions (`:constant_shift` uses
  `shift`; `:gaugewave` / `:sineshift` use amplitude `A` and period
  `d`).
* `ic ∈ {:exact, :noise}` — exact-solution IC of wavenumber
  `ic_wavenumber`, or √eps-amplitude noise (robust-stability mode; the
  L² error is reported against the zero solution).
* `ε_KO` — Kreiss-Oliger coefficient (also tightens the `dt` choice).
* `bc` — outer boundary treatment:
  - `:periodic` (default): periodic ring mesh, no outer boundary.
  - `:auto`: classify each face from the background at `t0` and pick
    the natural admissible condition — on subluminal faces Dirichlet
    (exact data) for `ic = :exact` / Sommerfeld for `ic = :noise`;
    excision on superluminal outflow faces; full-state Dirichlet on
    superluminal inflow faces.
  - `(left = :sym, right = :sym)` with symbols from `:dirichlet`,
    `:sommerfeld`, `:excision`, `:full_dirichlet` — validated against
    each face's characteristic class (see `boundaries1d.jl`);
    inadmissible combinations throw an `ArgumentError`. Dirichlet
    data come from the background's exact solution for `ic = :exact`
    and are homogeneous for `ic = :noise`.

Returns a NamedTuple with sample times `ts`, sorted node line
`xs_line` + permutation, spacetime samples `Φs`/`Πs :: (N·M, Nt)`,
`l2_err`, total energy `energy`, final state, and the operator-level
handles (`mesh`, `geom`, `elem`, `ops`).
"""
function evolve1d(; T::Type = Float64,
                    backend = CPU(),
                    N::Int = 4,
                    M::Int = 32,
                    x0::Real = 0,
                    x1::Real = 1,
                    background::Symbol = :sineshift,
                    A::Real = 0.3,
                    d::Real = 1,
                    shift::Real = 0.5,
                    ic::Symbol = :exact,
                    ic_wavenumber::Real = 2π,
                    noise_amp::Real = sqrt(eps(Float64)),
                    ε_KO::Real = 0,
                    bc = :periodic,
                    t0::Real = 0,
                    t1::Real = 1,
                    Nt::Int = 200,
                    cfl::Real = 1//10)

    on_cpu = backend isa CPU
    on_cpu || T <: AbstractFloat ||
        error("evolve1d: non-CPU backend requires a floating-point T; got $T")

    periodic = bc === :periodic
    mesh = make_uniform_line(T, M, T(x0), T(x1); periodic)
    elem = make_element(T, N)
    ops  = make_operators(elem)
    geom_host = make_geometry(mesh, elem)
    geom = on_cpu ? geom_host : to_device(geom_host, backend)
    ws   = make_wave1d_workspace(geom, ops)

    x_grid = reshape(copy(geom_host.coords), N, M)
    if on_cpu
        x_grid_dev = x_grid
    else
        x_grid_dev = KernelAbstractions.allocate(backend, T, N, M)
        copyto!(x_grid_dev, x_grid)
    end

    bg, Φ_exact_fn, Π_exact_fn, DΦ_exact_fn, max_speed =
        _background1d(background, T; A, d, shift, k_w = ic_wavenumber)

    # Boundary-condition setup: classify the two outer faces from the
    # background at t0, resolve :auto, and validate the requested kinds
    # against the characteristic classes.
    xL, xR = T(x0), T(x1)
    local kindL::Int, kindR::Int, classL0::Int, classR0::Int
    if !periodic
        aL, βL = _bg_point(bg, T(t0), xL)
        aR, βR = _bg_point(bg, T(t0), xR)
        classL0 = classify_face1d(aL, βL, -1)
        classR0 = classify_face1d(aR, βR, +1)
        # :auto picks the natural admissible condition per face. At
        # subluminal faces: Dirichlet (exact-solution data) for
        # ic = :exact so the analytic reference keeps entering the
        # domain; Sommerfeld (absorbing) for ic = :noise.
        auto(class) = class == FACE_SUBLUMINAL ?
                          (ic === :exact ? BC_DIRICHLET : BC_SOMMERFELD) :
                      class == FACE_OUTFLOW    ? BC_EXCISION :
                                                 BC_FULL_DIRICHLET
        if bc === :auto
            kindL, kindR = auto(classL0), auto(classR0)
        elseif bc isa NamedTuple && haskey(bc, :left) && haskey(bc, :right)
            kindL, kindR = bc1d_kind(bc.left), bc1d_kind(bc.right)
        else
            throw(ArgumentError("evolve1d: bc must be :periodic, :auto, " *
                "or (left = :sym, right = :sym); got $bc"))
        end
        validate_bc1d(classL0, kindL, "left (−x)")
        validate_bc1d(classR0, kindR, "right (+x)")
    end

    # CFL-derived fixed dt: wave limit `cfl · dx_min / max_speed`,
    # plus the exact KO-term limit when ε_KO ≠ 0. With the μ⁻⁵
    # normalisation the KO spectral radius is exactly `ε_KO · ws.μ`
    # (RK4's negative-real-axis reach is ≈ 2.8, halved for safety), so
    # this branch only binds for ε_KO ≳ 1.
    h_elem  = T(geom_host.jac[1, 1, 1, 1])
    ξs      = elem.xs
    dx_min  = minimum(ξs[i+1] - ξs[i] for i in 1:N-1) * h_elem
    dt      = T(cfl) * dx_min / max_speed
    if ε_KO != 0
        dt = min(dt, T(1.4) / (T(ε_KO) * ws.μ))
    end

    # IC on the host grid, then migrate.
    Φ0_host = Matrix{T}(undef, N, M)
    Π0_host = Matrix{T}(undef, N, M)
    if ic === :exact
        @. Φ0_host = Φ_exact_fn(T(t0), x_grid)
        @. Π0_host = Π_exact_fn(T(t0), x_grid)
    elseif ic === :noise
        amp = T(noise_amp)
        Φ0_host .= amp .* randn(T, N, M)
        Π0_host .= amp .* randn(T, N, M)
    else
        error("evolve1d: unknown ic $ic (expected :exact or :noise)")
    end
    Φ0 = on_cpu ? Φ0_host : copyto!(similar(x_grid_dev), Φ0_host)
    Π0 = on_cpu ? Π0_host : copyto!(similar(x_grid_dev), Π0_host)

    # Parameter bundle for the RHS: backgrounds are sampled into the
    # preallocated coefficient fields at every integrator stage time;
    # for non-periodic meshes the boundary bundle (face classes
    # re-checked, data scalars from the exact closures) is assembled
    # host-side per stage.
    withdata = ic === :exact
    p = (; geom, ops, ws, bg, xgrid = x_grid_dev,
         a = similar(Φ0), β = similar(Φ0), sγ = similar(Φ0),
         ε_KO = T(ε_KO))
    function rhs!(du, u, p, t)
        Φ, Π = u.x[1], u.x[2]
        Φ̇, Π̇ = du.x[1], du.x[2]
        sample_background!(p.a, p.β, p.sγ, p.bg, t, p.xgrid)
        bc1d = periodic ? nothing :
            _assemble_bc1d(p.bg, t, xL, xR, kindL, kindR,
                           classL0, classR0,
                           Φ_exact_fn, Π_exact_fn, DΦ_exact_fn,
                           withdata, T)
        wave1d_curved_rhs!(Φ̇, Π̇, Φ, Π, p.a, p.β;
                           p.geom, p.ops, p.ws, ε_KO = p.ε_KO, bc1d)
        return nothing
    end

    alg  = pick_integrator_first_order(N)
    prob = ODEProblem(rhs!, ArrayPartition(Φ0, Π0), (T(t0), T(t1)), p)
    integrator = init(prob, alg; dt,
                      adaptive       = false,
                      save_everystep = false,
                      save_start     = false,
                      save_end       = false,
                      dense          = false)

    ts = range(T(t0), T(t1), Nt)
    Ns      = N * M
    xs_line = vec(x_grid)
    perm    = sortperm(xs_line)
    xs_line = xs_line[perm]
    Φs       = Array{T}(undef, Ns, Nt)
    Πs       = Array{T}(undef, Ns, Nt)
    ts_actual = Vector{T}(undef, Nt)
    l2_err   = Vector{T}(undef, Nt)
    energy   = Vector{T}(undef, Nt)
    Φ_host = Matrix{T}(undef, N, M)
    Π_host = Matrix{T}(undef, N, M)
    Φ_ref  = Matrix{T}(undef, N, M)
    Hphys_host = geom_host.Hphys
    sγ_host    = Matrix{T}(undef, N, M)
    ws_host    = on_cpu ? ws : make_wave1d_workspace(geom_host, ops)

    prog = Progress(Nt;
                    desc = "evolve1d (N=$N, M=$M, bg=$background, " *
                           "backend=$(typeof(backend).name.name)): ",
                    barlen = 30, showspeed = true)
    for (n, t) in enumerate(ts)
        while integrator.t < t
            step!(integrator)
        end
        next!(prog)
        # Fixed dt overshoots the sample time by < dt; record and use
        # the actual time for the analytic reference.
        ta = T(integrator.t)
        ts_actual[n] = ta

        copyto!(Φ_host, integrator.u.x[1])
        copyto!(Π_host, integrator.u.x[2])
        @assert all(isfinite, Φ_host) && all(isfinite, Π_host)
        Φs[:, n] = vec(Φ_host)[perm]
        Πs[:, n] = vec(Π_host)[perm]

        # Physical-L² error vs the exact solution (zero for :noise).
        if ic === :exact
            @. Φ_ref = Φ_exact_fn(ta, x_grid)
        else
            fill!(Φ_ref, zero(T))
        end
        l2_err[n] = sqrt(sum(@. (Φ_host - Φ_ref)^2 * Hphys_host))

        # Total ADM energy (host-side; the state was already copied).
        sample_background!(p.a, p.β, p.sγ, bg, ta, x_grid_dev)
        copyto!(sγ_host, p.sγ)
        energy[n] = wave1d_energy(Φ_host, Π_host, sγ_host;
                                  geom = geom_host, ops, ws = ws_host)
    end
    finish!(prog)

    return (; ts, ts_actual, xs_line, perm, Φs, Πs, l2_err, energy,
              Φ_final = copy(Φ_host), Π_final = copy(Π_host),
              mesh, geom = geom_host, elem, ops, background, ic, bc,
              x0 = T(x0), x1 = T(x1), dt, dx = dx_min,
              integrator_name = nameof(typeof(alg)))
end

################################################################################
# evolve2d

"""
    evolve2d(; T = Float64, backend = CPU(), N = 4, M = 16,
               x0 = 0, x1 = 1, background = :minkowski,
               A = 0.1, d = 1, shift = (0.0, 0.0),
               ic = :exact, bc = :periodic, ε_KO = 0,
               t0 = 0, t1 = 1, Nt = 200, cfl = 1//10) → NamedTuple

2D scalar wave on a 2+1 ADM background (`wave2d_curved_rhs!`) on the
uniform_quad domain `[x0,x1]²`, first-order (Φ,Π) system integrated
with explicit RK (`pick_integrator_first_order`). Mirrors `evolve1d`.

* `background ∈ {:minkowski, :constant_shift, :gaugewave}` — built-in
  backgrounds with exact solutions (`:constant_shift` uses the
  2-vector `shift`; `:gaugewave` uses amplitude `A`, period `d`,
  propagating in x).
* `ic ∈ {:exact, :noise}`.
* `bc` — `:periodic`, `:auto` (per-side classification: subluminal →
  absorbing Sommerfeld, superluminal outflow → excision, superluminal
  inflow → full-state Dirichlet with exact data), or a 4-tuple of
  symbols for the (−x,+x,−y,+y) sides. The radiative (Sommerfeld) BC
  is the characteristic-free field-radiation SAT, valid for small
  shift; see `boundaries2d.jl`.

Returns `(; ts, ts_actual, xs_line, perm, Φs, Πs, l2_err, energy,
Φ_final, Π_final, mesh, geom, elem, ops, background, ic, bc, x0, x1,
dt, dx, slice_y, integrator_name)` for the `bin/wave2d.jl` plot app.
"""
function evolve2d(; T::Type = Float64,
                    backend = CPU(),
                    N::Int = 4,
                    M::Int = 16,
                    x0::Real = 0,
                    x1::Real = 1,
                    mesh_kind::Symbol = :cubical,
                    R::Real = 0.3,
                    L::Real = 0.2, R1::Real = 0.5, R2::Real = 1.0,
                    ic_width::Real = 0.15,
                    background::Symbol = :minkowski,
                    A::Real = 0.1,
                    d::Real = 1,
                    shift = (0.0, 0.0),
                    ic::Symbol = :exact,
                    bc = :periodic,
                    noise_amp::Real = sqrt(eps(Float64)),
                    ε_KO::Real = 0,
                    t0::Real = 0,
                    t1::Real = 1,
                    Nt::Int = 200,
                    cfl::Real = 1//10,
                    slice_y::Union{Nothing, Real} = nothing)

    on_cpu = backend isa CPU
    on_cpu || T <: AbstractFloat ||
        error("evolve2d: non-CPU backend requires a floating-point T; got $T")
    curv = mesh_kind === :cubed_square || mesh_kind === :inflated_square ||
           mesh_kind === :annulus
    periodic = (bc === :periodic) && !curv

    # `:cubical` → axis-aligned affine uniform_quad (per-axis operator);
    # `:cubed_square` / `:inflated_square` → curvilinear FILLED disk;
    # `:annulus` → curvilinear ring R1 ≤ |x| ≤ R2 with the inner circle
    # an excision surface (tag 8) and the outer circle the computational
    # boundary — the 2D BH-excision setup. All curvilinear kinds use the
    # discrete metric terms and the free-stream-preserving conservative
    # operator + physical-normal boundary.
    if mesh_kind === :cubed_square
        mesh = make_cubed_square_mesh(T, M, T(R))
        x0, x1 = -one(T), one(T)
    elseif mesh_kind === :inflated_square
        mesh = make_inflated_square_mesh(T, T(L), T(R1), T(R2), M)
        x0, x1 = -T(R2), T(R2)
    elseif mesh_kind === :annulus
        mesh = make_annulus_mesh(T, T(R1), T(R2), M;
                                 inner_bc = :excision, outer_bc = :sommerfeld)
        x0, x1 = -T(R2), T(R2)
    else
        mesh = make_uniform_quad(T, M, M, T(x0), T(x1); periodic)
    end
    elem = make_element(T, N); ops = make_operators(elem)
    geom_host = make_geometry(mesh, elem)
    geom = on_cpu ? geom_host : to_device(geom_host, backend)
    ws   = make_wave2d_workspace(geom, ops)
    coef = make_coef2d(geom)
    # Discrete metric terms are computed on the HOST geom (host scalar
    # loop); the device evolution uses a migrated copy, while the host
    # monitoring loop uses `metric_host`.
    metric_host = curv ? make_metric_terms2d(geom_host, ops) : nothing
    metric = !curv ? nothing :
             on_cpu ? metric_host : metric_to_device(metric_host, backend)

    xg = reshape(copy(geom_host.coords[1, :, :, :]), N, N, mesh.Ne)
    yg = reshape(copy(geom_host.coords[2, :, :, :]), N, N, mesh.Ne)
    xg_d = on_cpu ? xg : copyto!(similar(coef.alpha), xg)
    yg_d = on_cpu ? yg : copyto!(similar(coef.alpha), yg)

    bg, Φe, Πe, Dxe, Dye, max_speed =
        _background2d(background, T; A, d, shift, R1 = T(R1), R2 = T(R2))

    # Host-resident background coefficients, for boundary-face
    # classification (below) and the per-output diagnostics (energy / L²).
    # Sampled on the host grids so all reads are host-side even on GPU.
    coef_h = on_cpu ? coef : make_coef2d(geom_host)

    # Smallest physical node spacing (handles curved elements).
    dx_min = _min_node_spacing_2d(geom_host.coords)
    dt = T(cfl) * dx_min / max_speed
    if ε_KO != 0
        dt = min(dt, T(1.4) / (T(ε_KO) * ws.μ))
    end

    # Boundary setup. Curvilinear: a single Sommerfeld kind on the
    # whole outer circle. Rectangular: classify the four sides at t0,
    # resolve :auto, validate. Side axis/sign: 1→−x,2→+x,3→−y,4→+y.
    local kinds::NTuple{4,Int}
    if curv
        # Single BC kind on the whole curved outer circle: Sommerfeld
        # (absorbing, default) or Dirichlet (injects the exact solution
        # — requires ic = :exact). `bc === :periodic` is the unset
        # default and maps to Sommerfeld here.
        ck = bc === :periodic ? :sommerfeld : bc
        ck === :sommerfeld || ck === :dirichlet ||
            throw(ArgumentError("evolve2d: curvilinear ($mesh_kind) bc " *
                "must be :sommerfeld or :dirichlet; got $bc"))
        ck === :dirichlet && ic !== :exact &&
            throw(ArgumentError("evolve2d: curvilinear bc=:dirichlet " *
                "requires ic=:exact (it injects the exact solution)"))
        kinds = ntuple(_ -> bc1d_kind(ck), 4)
    elseif !periodic
        # Classify each side from a node that actually lies on it, using
        # the HOST coefficients (host scalar reads — safe on GPU).
        sample_background2d!(coef_h, bg, T(t0), xg, yg)
        side_axis = (1, 1, 2, 2); side_sign = (-1, 1, -1, 1)
        classes = ntuple(4) do f
            i, j, e = _side_node_2d(geom_host, f, N)
            classify_face2d(coef_h.alpha[i,j,e], coef_h.b1[i,j,e],
                            coef_h.b2[i,j,e], coef_h.gu11[i,j,e],
                            coef_h.gu22[i,j,e], side_axis[f], side_sign[f])
        end
        autopick(c) = c == FACE_SUBLUMINAL ? BC_SOMMERFELD :
                      c == FACE_OUTFLOW    ? BC_EXCISION : BC_FULL_DIRICHLET
        if bc === :auto
            kinds = ntuple(f -> autopick(classes[f]), 4)
        elseif bc isa Tuple || bc isa NamedTuple
            syms = bc isa NamedTuple ? (bc.mx, bc.px, bc.my, bc.py) : bc
            kinds = ntuple(f -> bc1d_kind(syms[f]), 4)
        else
            throw(ArgumentError("evolve2d: bc must be :periodic, :auto, " *
                "or a 4-tuple of side symbols; got $bc"))
        end
        for f in 1:4
            validate_bc1d(classes[f], kinds[f],
                          ("−x","+x","−y","+y")[f] * " side")
        end
    end

    # IC.
    Φ0 = Array{T,3}(undef, N, N, mesh.Ne); Π0 = similar(Φ0)
    if ic === :exact
        @. Φ0 = Φe(T(t0), xg, yg); @. Π0 = Πe(T(t0), xg, yg)
    elseif ic === :gaussian
        w = T(ic_width)
        @. Φ0 = exp(-(xg^2 + yg^2) / (2 * w^2)); fill!(Π0, zero(T))
    elseif ic === :noise
        Φ0 .= T(noise_amp) .* randn(T, N, N, mesh.Ne)
        Π0 .= T(noise_amp) .* randn(T, N, N, mesh.Ne)
    else
        error("evolve2d: unknown ic $ic")
    end
    Φdev = on_cpu ? Φ0 : copyto!(similar(coef.alpha), Φ0)
    Πdev = on_cpu ? Π0 : copyto!(similar(coef.alpha), Π0)

    withdata = ic === :exact
    # Boundary data buffers, allocated only when a data-carrying BC is
    # active. Rectangular full-state Dirichlet uses (gΦ, gΠ); the curved
    # field-radiation Dirichlet uses (gΠ, gDx, gDy) = exact (Π, ∂_xΦ,
    # ∂_yΦ). Refilled each stage in `rhs!`.
    curv_dir = curv && kinds[1] == BC_DIRICHLET
    needdata = (!periodic && any(==(BC_FULL_DIRICHLET), kinds)) || curv_dir
    # Allocate on the same backend as the state (device on GPU, host on
    # CPU) so the BC kernel reads them in place; filled each stage in rhs!.
    _gbuf() = fill!(similar(coef.alpha), zero(T))
    gΦ  = needdata ? _gbuf() : nothing
    gΠ  = needdata ? _gbuf() : nothing
    gDx = curv_dir ? _gbuf() : nothing
    gDy = curv_dir ? _gbuf() : nothing
    # Annulus inner circle is tagged excision (8): the curvilinear BC
    # pass gives those faces no SAT (pure outflow) while `kinds[1]`
    # drives the outer circle.
    exc_tag = mesh_kind === :annulus ? 8 : 0

    p = (; geom, ops, ws, coef, bg, metric, xg = xg_d, yg = yg_d)
    function rhs!(du, u, p, t)
        Φ, Π = u.x[1], u.x[2]; Φ̇, Π̇ = du.x[1], du.x[2]
        sample_background2d!(p.coef, p.bg, t, p.xg, p.yg)
        bc2d = nothing
        if !periodic
            if needdata && withdata
                # Fill on the device grids (== host grids on CPU) so the
                # buffers match the backend the BC kernel reads from.
                @. gΠ = Πe(t, xg_d, yg_d)
                if curv_dir
                    @. gDx = Dxe(t, xg_d, yg_d); @. gDy = Dye(t, xg_d, yg_d)
                else
                    @. gΦ = Φe(t, xg_d, yg_d)
                end
            end
            bc2d = make_bc2d(kinds; gΦ, gΠ, gDx, gDy, excision_tag = exc_tag)
        end
        wave2d_curved_rhs!(Φ̇, Π̇, Φ, Π, p.coef; p.geom, p.ops, p.ws,
                           ε_KO = T(ε_KO), bc2d, metric = p.metric)
        return nothing
    end

    alg  = pick_integrator_first_order(N)
    prob = ODEProblem(rhs!, ArrayPartition(Φdev, Πdev), (T(t0), T(t1)), p)
    integrator = init(prob, alg; dt, adaptive = false,
                      save_everystep = false, save_start = false,
                      save_end = false, dense = false)

    # Slice along y for the spacetime plot.
    y_target = T(slice_y === nothing ? (x0 + x1) / 2 : slice_y)
    slice_idx, xs_line = _build_slice_2d(geom_host.coords, y_target;
                                         atol = sqrt(eps(T)))
    isempty(xs_line) && error("evolve2d: slice y=$y_target hit no nodes")
    perm = sortperm(xs_line); xs_line = xs_line[perm]
    sidx = slice_idx[perm]

    ts = range(T(t0), T(t1), Nt)
    Ns = length(xs_line)
    Φs = Array{T}(undef, Ns, Nt); Πs = similar(Φs)
    ts_actual = Vector{T}(undef, Nt)
    l2_err = Vector{T}(undef, Nt); energy = Vector{T}(undef, Nt)
    Φh = Array{T,3}(undef, N, N, mesh.Ne); Πh = similar(Φh)
    Φref = similar(Φh)
    Hphys_h = geom_host.Hphys
    ws_h = on_cpu ? ws : make_wave2d_workspace(geom_host, ops)
    # coef_h was allocated above (used for boundary classification too).

    prog = Progress(Nt; desc = "evolve2d (M=$M, bg=$background, " *
                    "backend=$(typeof(backend).name.name)): ",
                    barlen = 30, showspeed = true)
    for (n, t) in enumerate(ts)
        while integrator.t < t; step!(integrator); end
        next!(prog)
        ta = T(integrator.t); ts_actual[n] = ta
        copyto!(Φh, integrator.u.x[1]); copyto!(Πh, integrator.u.x[2])
        @assert all(isfinite, Φh) && all(isfinite, Πh)
        for (q, (e, ii, jj)) in enumerate(sidx)
            Φs[q, n] = Φh[ii, jj, e]; Πs[q, n] = Πh[ii, jj, e]
        end
        if ic === :exact
            @. Φref = Φe(ta, xg, yg)
        else
            fill!(Φref, zero(T))
        end
        Hw = curv ? metric_host.Hd : Hphys_h
        l2_err[n] = sqrt(sum(@. (Φh - Φref)^2 * Hw))
        sample_background2d!(coef_h, bg, ta, xg, yg)
        energy[n] = wave2d_energy(Φh, Πh, coef_h; geom = geom_host, ops,
                                  ws = ws_h, metric = metric_host)
    end
    finish!(prog)

    return (; ts, ts_actual, xs_line, perm, Φs, Πs, l2_err, energy,
              Φ_final = copy(Φh), Π_final = copy(Πh),
              mesh, geom = geom_host, elem, ops, background, ic, bc, mesh_kind,
              x0 = T(x0), x1 = T(x1), dt, dx = dx_min, y_target,
              integrator_name = nameof(typeof(alg)))
end

# Built-in 2D backgrounds. Returns
# (bg::Background2D, Φe, Πe, Dxe, Dye, max_speed). Backgrounds with an
# exact scalar-wave solution fill the closures; `:radial_shift` has none
# (use ic=:noise) and returns zero closures.
function _background2d(kind::Symbol, ::Type{T}; A, d, shift, R1 = 0, R2 = 1) where {T}
    if kind === :minkowski || kind === :constant_shift
        bx, by = kind === :minkowski ? (zero(T), zero(T)) :
                 (T(shift[1]), T(shift[2]))
        bg = AnalyticBackground2D(_Const3(one(T)), _ConstVec2(bx, by),
                                  _ConstMet2(one(T), zero(T), one(T)))
        # Diagonal plane wave Φ = sin(2π(x+y) − ωt). For ∂_tΦ=β·∇Φ+Π,
        # ∂_tΠ=∇·(βΠ+∇Φ) the dispersion is (ω+β·k)²=|k|², so the physical
        # branch is ω = −β·k + |k| (NOT +β·k): with k=2π(1,1),
        # ω = −2π(bx+by) + 2π√2.
        k = 2 * T(π); ω = -k * (bx + by) + k * sqrt(T(2))
        Φe = (t,x,y) -> sin(k*(x+y) - ω*t)
        # Π = ∂_tΦ − βⁱ∂_iΦ = (−ω − k(bx+by))cos = −k√2·cos (β-independent).
        Πe = (t,x,y) -> (-ω - k*(bx+by)) * cos(k*(x+y) - ω*t)
        Dxe = (t,x,y) -> k * cos(k*(x+y) - ω*t)      # ∂_xΦ = ∂_yΦ
        return bg, Φe, Πe, Dxe, Dxe, abs(bx) + abs(by) + sqrt(T(2))
    elseif kind === :gaugewave
        Aᵥ, dᵥ = T(A), T(d); kᵥ = 2*T(π)/dᵥ; k₀ = 2*T(π); C = Aᵥ*dᵥ/(4*T(π))
        bg = MetricBackground2D(SpacetimeMetrics.GaugeWave(Aᵥ, dᵥ))
        ψ = (t,x) -> x - t + 2C*cos(kᵥ*(x-t))
        Φe = (t,x,y) -> sin(k₀*ψ(t,x))
        Πe = (t,x,y) -> -k₀*(1 - Aᵥ*sin(kᵥ*(x-t)))*cos(k₀*ψ(t,x))
        # ∂_xΦ = k₀·∂_xψ·cos(k₀ψ), ∂_xψ = 1 − 2C kᵥ sin(kᵥ(x−t)); ∂_yΦ = 0.
        Dxe = (t,x,y) -> k₀*(1 - 2C*kᵥ*sin(kᵥ*(x-t)))*cos(k₀*ψ(t,x))
        Dye = (t,x,y) -> zero(T)
        return bg, Φe, Πe, Dxe, Dye, one(T)
    elseif kind === :radial_shift
        # Flat space (α=1, γ=I) with a radial shift whose magnitude ramps
        # LINEARLY in r from `V` (>1) at the inner radius R1 to `V_out`
        # (<0.1) at the outer radius R2:
        #   β_r(r) = V + (V_out − V)·(r − R1)/(R2 − R1),  β = β_r·(x,y)/r.
        # The radial characteristic speeds are dr/dt = −(β_r ± a) (this
        # solver advects with +βⁱ∂_iΦ; a = α√γ^rr = 1). At R1 both are
        # < 0 (V > 1 ⇒ both characteristics fall into the hole) → the
        # inner circle is SUPERLUMINAL OUTFLOW, correctly handled by
        # EXCISION (no SAT). At R2 the face is subluminal ⇒ Sommerfeld.
        # Because dr/dt = −(β_r ± a), infall (outflow at the inner
        # circle) corresponds to β_r > 0 — the shift VECTOR points
        # radially outward even though matter falls inward; the opposite
        # sign (β_r < −1) would be superluminal INFLOW (full-Dirichlet),
        # which is out of scope. A linear ramp is used so the shift is
        # well resolved on the grid (a steep 1/r² profile would be
        # under-resolved and drive a spurious variable-β instability). No
        # analytic solution → use ic=:noise. `A` sets V. max_speed = V+1.
        V = T(A); R1v = T(R1); R2v = T(R2); Vout = T(1)/20
        bg = AnalyticBackground2D(_Const3(one(T)),
                                  _RadialShift2(V, Vout, R1v, R2v),
                                  _ConstMet2(one(T), zero(T), one(T)))
        z = (t,x,y) -> zero(T)
        return bg, z, z, z, z, V + one(T)
    else
        error("evolve2d: unknown background $kind (:minkowski, " *
              ":constant_shift, :gaugewave, :radial_shift)")
    end
end

# Radial shift with magnitude ramping linearly in r from `Vin` at `R1`
# to `Vout` at `R2`: β = β_r(r)·(x,y)/r. With β_r > 0 the radial
# characteristic speeds −(β_r ± a) are negative (matter falls inward),
# making the inner circle a superluminal-outflow / excision surface.
struct _RadialShift2{T}; Vin::T; Vout::T; R1::T; R2::T; end
function (f::_RadialShift2)(t, x, y)
    r = sqrt(x*x + y*y)
    βr = f.Vin + (f.Vout - f.Vin) * (r - f.R1) / (f.R2 - f.R1)
    s = βr / r
    return (s * x, s * y)
end

struct _RadialShift3{T}; Vin::T; Vout::T; R1::T; R2::T; end
function (f::_RadialShift3)(t, x, y, z)
    r = sqrt(x*x + y*y + z*z)
    βr = f.Vin + (f.Vout - f.Vin) * (r - f.R1) / (f.R2 - f.R1)
    s = βr / r
    return (s * x, s * y, s * z)
end

# isbits callable closures so the backgrounds pass into GPU kernels.
struct _Const3{T}; v::T; end
(f::_Const3)(t, x, y) = f.v
struct _ConstVec2{T}; b1::T; b2::T; end
(f::_ConstVec2)(t, x, y) = (f.b1, f.b2)
struct _ConstMet2{T}; g11::T; g12::T; g22::T; end
(f::_ConstMet2)(t, x, y) = (f.g11, f.g12, f.g22)

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
function _evolve3d_strong(; T::Type = Float64,
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

"""
    evolve3d(; formulation = :strong, kwargs...)

3D wave driver. `formulation = :strong` (default) is the second-order
Laplacian wave (`_evolve3d_strong`, symplectic). `formulation =
:conservative` is the first-order ADM (Φ,Π) wave (`_evolve3d_conservative`,
explicit RK), the 3D analog of `evolve2d`. The two take different kwargs.
"""
function evolve3d(; formulation::Symbol = :strong, kwargs...)
    formulation === :conservative && return _evolve3d_conservative(; kwargs...)
    formulation === :strong ||
        throw(ArgumentError("evolve3d: formulation must be :strong or " *
                            ":conservative; got $formulation"))
    return _evolve3d_strong(; kwargs...)
end

# (i,j,k,e) of a representative node on boundary side `f` (1=−x..6=+z) of
# an axis-aligned 3D mesh — the first boundary element on that side, at
# the side's row and mid tangential node.
function _side_node_3d(geom, f, N)
    bdry = geom.conn.bdry
    a_idx = (f + 1) ÷ 2
    row = isodd(f) ? 1 : N
    mid = (N + 1) ÷ 2
    @inbounds for e in 1:geom.Ne
        bdry[f, e] == 0 && continue
        return a_idx == 1 ? (row, mid, mid, e) :
               a_idx == 2 ? (mid, row, mid, e) : (mid, mid, row, e)
    end
    return (1, 1, 1, 1)
end

# Built-in 3D ADM backgrounds. Returns
# (bg::Background3D, Φe, Πe, Dxe, Dye, Dze, max_speed).
function _background3d(kind::Symbol, ::Type{T}; shift, R1 = 0, R2 = 1) where {T}
    if kind === :minkowski || kind === :constant_shift
        bx, by, bz = kind === :minkowski ? (zero(T), zero(T), zero(T)) :
                     (T(shift[1]), T(shift[2]), T(shift[3]))
        bg = AnalyticBackground3D(_Const4(one(T)), _ConstVec3(bx, by, bz),
                                  _ConstMet3(one(T), zero(T), zero(T),
                                             one(T), zero(T), one(T)))
        # Φ = sin(k(x+y+z) − ωt); dispersion (ω+β·k)²=|k|² ⇒ ω = −β·k + |k|,
        # k = 2π(1,1,1): ω = −2π(bx+by+bz) + 2π√3.
        k = 2 * T(π); s = bx + by + bz; ω = -k * s + k * sqrt(T(3))
        Φe = (t,x,y,z) -> sin(k*(x+y+z) - ω*t)
        Πe = (t,x,y,z) -> (-ω - k*s) * cos(k*(x+y+z) - ω*t)   # = −k√3·cos
        De = (t,x,y,z) -> k * cos(k*(x+y+z) - ω*t)            # ∂_xΦ=∂_yΦ=∂_zΦ
        return bg, Φe, Πe, De, De, De, abs(bx)+abs(by)+abs(bz)+sqrt(T(3))
    elseif kind === :radial_shift
        # 3D analog of the 2D annulus excision: flat space (α=1, γ=I)
        # with a radial shift ramping LINEARLY in r from `V` (>1) at the
        # inner radius R1 to `V_out` (<0.1) at the outer radius R2,
        #   β_r(r) = V + (V_out − V)·(r − R1)/(R2 − R1),  β = β_r·x⃗/r.
        # Radial characteristic speeds dr/dt = −(β_r ± a), a = α√γ^rr = 1.
        # At R1 (V > 1) both are < 0 ⇒ inner sphere is SUPERLUMINAL
        # OUTFLOW → EXCISION (no SAT); at R2 the face is subluminal ⇒
        # Sommerfeld. A linear ramp keeps the shift well resolved (a
        # steep 1/r² profile would be under-resolved). No analytic
        # solution → use ic=:noise. `shift[1]` sets V. max_speed = V+1.
        V = T(shift[1]); Vout = T(1)/20
        bg = AnalyticBackground3D(_Const4(one(T)),
                                  _RadialShift3(V, Vout, T(R1), T(R2)),
                                  _ConstMet3(one(T), zero(T), zero(T),
                                             one(T), zero(T), one(T)))
        z = (t,x,y,z) -> zero(T)
        return bg, z, z, z, z, z, V + one(T)
    else
        error("evolve3d (conservative): unknown background $kind " *
              "(:minkowski, :constant_shift, :radial_shift)")
    end
end

# Conservative first-order (Φ,Π) ADM 3D wave driver — the 3D analog of
# evolve2d. Milestone 1: axis-aligned affine `uniform_hex` only.
function _evolve3d_conservative(; T::Type = Float64,
                    backend = CPU(),
                    N::Int = 4,
                    M::Int = 8,
                    x0::Real = 0, x1::Real = 1,
                    mesh_kind::Symbol = :cubical,
                    R::Real = 0.3, L::Real = 0.2, R1::Real = 0.5, R2::Real = 1.0,
                    background::Symbol = :minkowski,
                    shift = (0.0, 0.0, 0.0),
                    ic::Symbol = :exact,
                    bc = :periodic,
                    ic_width::Real = 0.15,
                    noise_amp::Real = sqrt(eps(Float64)),
                    ε_KO::Real = 0,
                    t0::Real = 0, t1::Real = 1, Nt::Int = 200,
                    cfl::Real = 1//10,
                    slice_y::Union{Nothing, Real} = nothing,
                    slice_z::Union{Nothing, Real} = nothing)
    on_cpu = backend isa CPU
    on_cpu || T <: AbstractFloat ||
        error("evolve3d: non-CPU backend requires a floating-point T; got $T")
    curv = mesh_kind !== :cubical
    periodic = (bc === :periodic) && !curv
    outer = (bc === :periodic || bc === :dirichlet) ? :dirichlet : :sommerfeld

    if mesh_kind === :cubical
        mesh = make_uniform_hex(T, M, T(x0), T(x1); periodic = periodic)
    elseif mesh_kind === :cubed_cube
        mesh = make_cubed_cube_mesh(T, M, T(R)); x0, x1 = -one(T), one(T)
    elseif mesh_kind === :inflated_cube
        mesh = make_inflated_cube_mesh(T, T(L), T(R1), T(R2), M; outer_bc = outer)
        x0, x1 = -T(R2), T(R2)
    elseif mesh_kind === :radial_shell
        mesh = make_radial_shell_mesh(T, T(R1), T(R2), M;
                                      inner_bc = :excision, outer_bc = outer)
        x0, x1 = -T(R2), T(R2)
    else
        error("evolve3d conservative: unknown mesh_kind $mesh_kind " *
              "(:cubical, :cubed_cube, :inflated_cube, :radial_shell)")
    end
    elem = make_element(T, N); ops = make_operators(elem)
    geom_host = make_geometry(mesh, elem)
    geom = on_cpu ? geom_host : to_device(geom_host, backend)
    ws   = make_wave3d_workspace(geom, ops)
    coef = make_coef3d(geom)
    metric_host = curv ? make_metric_terms3d(geom_host, ops) : nothing
    metric = !curv ? nothing :
             on_cpu ? metric_host : metric_to_device(metric_host, backend)

    xg = reshape(copy(geom_host.coords[1,:,:,:,:]), N, N, N, mesh.Ne)
    yg = reshape(copy(geom_host.coords[2,:,:,:,:]), N, N, N, mesh.Ne)
    zg = reshape(copy(geom_host.coords[3,:,:,:,:]), N, N, N, mesh.Ne)
    xg_d = on_cpu ? xg : copyto!(similar(coef.alpha), xg)
    yg_d = on_cpu ? yg : copyto!(similar(coef.alpha), yg)
    zg_d = on_cpu ? zg : copyto!(similar(coef.alpha), zg)

    bg, Φe, Πe, Dxe, Dye, Dze, max_speed =
        _background3d(background, T; shift, R1 = T(R1), R2 = T(R2))
    coef_h = on_cpu ? coef : make_coef3d(geom_host)

    dx_min = _min_node_spacing_3d(geom_host.coords)
    dt = T(cfl) * dx_min / max_speed
    if ε_KO != 0
        dt = min(dt, T(1.4) / (T(ε_KO) * ws.μ))
    end

    # Boundary setup. Curvilinear: a single outer kind (Sommerfeld /
    # Dirichlet); radial-shell additionally excises the inner sphere
    # (tag 8). Rectangular: classify the 6 sides on the host.
    local kinds::NTuple{6,Int}
    if curv
        ck = bc === :periodic ? :sommerfeld : bc
        ck === :sommerfeld || ck === :dirichlet ||
            throw(ArgumentError("evolve3d: curvilinear ($mesh_kind) bc must " *
                "be :sommerfeld or :dirichlet; got $bc"))
        ck === :dirichlet && ic !== :exact &&
            throw(ArgumentError("evolve3d: curvilinear bc=:dirichlet requires " *
                "ic=:exact (it injects the exact solution)"))
        kinds = ntuple(_ -> bc1d_kind(ck), 6)
    elseif !periodic
        sample_background3d!(coef_h, bg, T(t0), xg, yg, zg)
        side_axis = (1,1,2,2,3,3); side_sign = (-1,1,-1,1,-1,1)
        classes = ntuple(6) do f
            i,j,k,e = _side_node_3d(geom_host, f, N)
            classify_face3d(coef_h.alpha[i,j,k,e], coef_h.b1[i,j,k,e],
                            coef_h.b2[i,j,k,e], coef_h.b3[i,j,k,e],
                            coef_h.gu11[i,j,k,e], coef_h.gu22[i,j,k,e],
                            coef_h.gu33[i,j,k,e], side_axis[f], side_sign[f])
        end
        autopick(c) = c == FACE_SUBLUMINAL ? BC_SOMMERFELD :
                      c == FACE_OUTFLOW    ? BC_EXCISION : BC_FULL_DIRICHLET
        if bc === :auto
            kinds = ntuple(f -> autopick(classes[f]), 6)
        elseif bc isa Tuple
            kinds = ntuple(f -> bc1d_kind(bc[f]), 6)
        else
            throw(ArgumentError("evolve3d: bc must be :periodic, :auto, or " *
                                "a 6-tuple; got $bc"))
        end
        for f in 1:6
            validate_bc1d(classes[f], kinds[f],
                          ("−x","+x","−y","+y","−z","+z")[f] * " side")
        end
    else
        kinds = ntuple(_ -> BC_SOMMERFELD, 6)
    end

    # IC.
    Φ0 = Array{T,4}(undef, N, N, N, mesh.Ne); Π0 = similar(Φ0)
    if ic === :exact
        @. Φ0 = Φe(T(t0), xg, yg, zg); @. Π0 = Πe(T(t0), xg, yg, zg)
    elseif ic === :gaussian
        w = T(ic_width); c0 = T((x0 + x1) / 2)
        @. Φ0 = exp(-((xg-c0)^2 + (yg-c0)^2 + (zg-c0)^2) / (2 * w^2))
        fill!(Π0, zero(T))
    elseif ic === :noise
        Φ0 .= T(noise_amp) .* randn(T, N, N, N, mesh.Ne)
        Π0 .= T(noise_amp) .* randn(T, N, N, N, mesh.Ne)
    else
        error("evolve3d: unknown ic $ic")
    end
    Φdev = on_cpu ? Φ0 : copyto!(similar(coef.alpha), Φ0)
    Πdev = on_cpu ? Π0 : copyto!(similar(coef.alpha), Π0)

    withdata = ic === :exact
    curv_dir = curv && kinds[1] == BC_DIRICHLET
    needdata = (!periodic && !curv && any(==(BC_FULL_DIRICHLET), kinds)) || curv_dir
    _gbuf() = fill!(similar(coef.alpha), zero(T))
    gΦ  = needdata ? _gbuf() : nothing
    gΠ  = needdata ? _gbuf() : nothing
    gDx = curv_dir ? _gbuf() : nothing
    gDy = curv_dir ? _gbuf() : nothing
    gDz = curv_dir ? _gbuf() : nothing
    # radial-shell inner sphere is tagged excision (8): the curvilinear
    # BC gives those faces no SAT while kinds[1] drives the outer sphere.
    exc_tag = mesh_kind === :radial_shell ? 8 : 0

    p = (; geom, ops, ws, coef, bg, metric, xg = xg_d, yg = yg_d, zg = zg_d)
    function rhs!(du, u, p, t)
        Φ, Π = u.x[1], u.x[2]; Φ̇, Π̇ = du.x[1], du.x[2]
        sample_background3d!(p.coef, p.bg, t, p.xg, p.yg, p.zg)
        bc3d = nothing
        if !periodic
            if needdata && withdata
                @. gΠ = Πe(t, xg_d, yg_d, zg_d)
                if curv_dir
                    @. gDx = Dxe(t, xg_d, yg_d, zg_d)
                    @. gDy = Dye(t, xg_d, yg_d, zg_d)
                    @. gDz = Dze(t, xg_d, yg_d, zg_d)
                else
                    @. gΦ = Φe(t, xg_d, yg_d, zg_d)
                end
            end
            bc3d = make_bc3d(kinds; gΦ, gΠ, gDx, gDy, gDz, excision_tag = exc_tag)
        end
        wave3d_curved_rhs!(Φ̇, Π̇, Φ, Π, p.coef; p.geom, p.ops, p.ws,
                           ε_KO = T(ε_KO), bc3d, metric = p.metric)
        return nothing
    end

    alg  = pick_integrator_first_order(N)
    prob = ODEProblem(rhs!, ArrayPartition(Φdev, Πdev), (T(t0), T(t1)), p)
    integrator = init(prob, alg; dt, adaptive = false,
                      save_everystep = false, save_start = false,
                      save_end = false, dense = false)

    y_target = T(slice_y === nothing ? (x0 + x1) / 2 : slice_y)
    z_target = T(slice_z === nothing ? (x0 + x1) / 2 : slice_z)
    slice_idx, xs_line = _build_slice_3d(geom_host.coords, y_target, z_target;
                                         atol = sqrt(eps(T)))
    # Diagnostic 1-D slice; curvilinear meshes often have no node exactly
    # on the x-axis, in which case the slice is simply empty (not fatal).
    local perm, sidx
    if isempty(xs_line)
        curv || error("evolve3d: slice y=$y_target z=$z_target hit no nodes")
        perm = Int[]; sidx = slice_idx
    else
        perm = sortperm(xs_line); xs_line = xs_line[perm]; sidx = slice_idx[perm]
    end

    ts = range(T(t0), T(t1), Nt); Ns = length(xs_line)
    Φs = Array{T}(undef, Ns, Nt); Πs = similar(Φs)
    ts_actual = Vector{T}(undef, Nt)
    l2_err = Vector{T}(undef, Nt); energy = Vector{T}(undef, Nt)
    Φh = Array{T,4}(undef, N, N, N, mesh.Ne); Πh = similar(Φh); Φref = similar(Φh)
    ws_h = on_cpu ? ws : make_wave3d_workspace(geom_host, ops)

    prog = Progress(Nt; desc = "evolve3d-cons (M=$M, bg=$background, " *
                    "backend=$(typeof(backend).name.name)): ",
                    barlen = 30, showspeed = true)
    for (n, t) in enumerate(ts)
        while integrator.t < t; step!(integrator); end
        next!(prog)
        ta = T(integrator.t); ts_actual[n] = ta
        copyto!(Φh, integrator.u.x[1]); copyto!(Πh, integrator.u.x[2])
        @assert all(isfinite, Φh) && all(isfinite, Πh)
        for (q, (e, ii, jj, kk)) in enumerate(sidx)
            Φs[q, n] = Φh[ii, jj, kk, e]; Πs[q, n] = Πh[ii, jj, kk, e]
        end
        if ic === :exact
            @. Φref = Φe(ta, xg, yg, zg)
        else
            fill!(Φref, zero(T))
        end
        Hw = curv ? metric_host.Hd : geom_host.Hphys
        l2_err[n] = sqrt(sum(@. (Φh - Φref)^2 * Hw))
        sample_background3d!(coef_h, bg, ta, xg, yg, zg)
        energy[n] = wave3d_energy(Φh, Πh, coef_h; geom = geom_host, ops,
                                  ws = ws_h, metric = curv ? metric_host : nothing)
    end
    finish!(prog)

    return (; ts, ts_actual, xs_line, perm, Φs, Πs, l2_err, energy,
              Φ_final = copy(Φh), Π_final = copy(Πh),
              mesh, geom = geom_host, elem, ops, background, ic, bc, mesh_kind,
              x0 = T(x0), x1 = T(x1), dt, dx = dx_min, y_target, z_target,
              integrator_name = nameof(typeof(alg)))
end
