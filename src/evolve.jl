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

    # Dirichlet data slot is the exact ingoing characteristic at the
    # face: u_R = ∂_xΦ − Π if s_R·n̂ < 0, else u_L = ∂_xΦ + Π
    # (mirrors the kernel's mode selection).
    g_in(x, a, β, n̂) = !withdata ? zero(T) :
        ((a - β) * n̂ < 0 ? T(De(t, x) - Πe(t, x)) : T(De(t, x) + Πe(t, x)))

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
