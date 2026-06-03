# 1D scalar wave on a 1+1 ADM background with space- and time-varying
# lapse α(t, x), shift β(t, x), and spatial metric γ_xx(t, x).
# State: (Φ, Π) with the *densitised* momentum
#
#     Π := (√γ_xx / α) · (∂_t Φ − β · ∂_x Φ),
#
# i.e. Π = √γ · (n^μ ∂_μ Φ) with n the unit normal of the foliation.
# The covariant wave equation ∂_μ(√|g| g^{μν} ∂_ν Φ) = 0 with
# √|g| = α·√γ_xx reduces to the flux-conservative first-order system
#
#     ∂_t Φ = β · ∂_x Φ + (α/√γ_xx) · Π
#     ∂_t Π = ∂_x ( (α/√γ_xx) · ∂_x Φ + β · Π ).
#
# The single ∂_x of the combined flux automatically supplies both the
# `β · ∂_x Π` advection term and the `(∂_x β) · Π` source term that a
# non-conservative ADM form would have to add by hand. For α ≡ 1 this
# reduces to the densitised form Π = √γ (∂_t Φ − β ∂_x Φ); for
# α = γ = 1, β = const it is the textbook constant-shift wave equation.
#
# Only the combination a := α/√γ_xx (and β) enters the evolution — the
# kernel takes `a` and `β` as its two coefficient fields; α and √γ
# individually are needed only for initial data and diagnostics.
#
# (CAPITAL Greek `Φ` and `Π` rather than lowercase `π`: Julia's
# `π = 3.14…` constant is the lowercase letter.)
#
# Discretisation: ONE consistent first-derivative operator everywhere —
# `HexSBPSAT.apply_D!`, reference SBP-G + centred-flux SAT at every
# element interface, with the neighbour relation read from the
# `HexMeshes.Mesh{1}` connectivity (periodic ring via
# `make_uniform_line(...; periodic = true)`). With this SAT the
# assembled operator `H · D` is exactly skew, so the RHS spectrum is
# purely imaginary up to background variation. RK + skew operator is
# marginally stable; Kreiss-Oliger dissipation patches this:
#
#     u̇ += ε · h^{2p+1} · D^{2p+2} · u,    here p = 2:  +ε · h⁵ · D⁶ u.
#
# Since `D` is skew, `D⁶` has eigenvalues `−μ⁶ ≤ 0` (dissipative), and
# the contribution on smooth data scales as `ε · h⁵ · k⁶ = O(h^{2p−1})`
# — KO does not degrade the formal order.

using HexSBPSAT: MeshGeometry, SBPOps, apply_D!
using KernelAbstractions: @kernel, @index, @Const, get_backend, allocate
using SpacetimeMetrics: adm_decompose
using StaticArrays: SVector

################################################################################
# ADM backgrounds: the kernel consumes the coefficient fields
# a = α/√γ_xx and β = βˣ sampled on the collocation points at the
# current integrator stage time; √γ_xx is sampled alongside for
# initial data and diagnostics.

"""
    Background1D

Abstract 1+1 ADM background (α, βˣ, γ_xx) for the 1D scalar wave.
Concrete subtypes implement `_bg_adm(bg, t, x) → (α, βˣ, γ_xx)`;
sample onto the grid with [`sample_background!`](@ref).
"""
abstract type Background1D end

"""
    AnalyticBackground1D(α_fn, β_fn, γ_fn) <: Background1D

Background given by closures `α_fn(t, x)`, `β_fn(t, x)`, `γ_fn(t, x)`
(the latter returns `γ_xx`). Use for backgrounds that are not (or
cannot be) expressed as a 4-metric — e.g. superluminal-shift stress
tests, which are coordinate effects no Lorentz boost can produce —
and for number types whose transcendental functions a metric type
doesn't support.
"""
struct AnalyticBackground1D{Fα, Fβ, Fγ} <: Background1D
    α_fn :: Fα
    β_fn :: Fβ
    γ_fn :: Fγ
end

@inline _bg_adm(bg::AnalyticBackground1D, t, x) =
    (bg.α_fn(t, x), bg.β_fn(t, x), bg.γ_fn(t, x))

"""
    MetricBackground1D(m) <: Background1D

Background sourced from a `SpacetimeMetrics.AbstractMetric` `m`:
ADM variables are extracted via `adm_decompose(m, (t, x, 0, 0))`
at every collocation point, so any 4-metric whose restriction to the
(t, x) plane is sensible works — e.g. `Minkowski()`,
`GaugeWave(A, d)` (genuinely varying lapse α = √H).
"""
struct MetricBackground1D{M} <: Background1D
    metric :: M
end

@inline function _bg_adm(bg::MetricBackground1D, t, x)
    α, β, γ = adm_decompose(bg.metric, SVector(t, x, zero(x), zero(x)))
    return α, β[1], γ[1, 1]
end

@kernel function _sample_background_kernel!(a, β, sqrtγ, bg, t::T,
                                            @Const(xgrid)) where {T}
    idx = @index(Global, Linear)
    @inbounds begin
        αv, βv, γv = _bg_adm(bg, t, xgrid[idx])
        sγ = sqrt(T(γv))
        sqrtγ[idx] = sγ
        a[idx]     = T(αv) / sγ
        β[idx]     = T(βv)
    end
end

"""
    sample_background!(a, β, sqrtγ, bg::Background1D, t, xgrid)

Fill the `(N, M)` coefficient fields `a = α/√γ_xx`, `β = βˣ`, and
`sqrtγ = √γ_xx` from `bg` at time `t` on the collocation grid
`xgrid :: (N, M)` (e.g. `reshape(geom.coords, N, M)`). Runs as a
KernelAbstractions kernel on whatever backend holds `a` —
allocation-free and GPU-compatible, suitable for calling at every
integrator stage time.
"""
function sample_background!(a::AbstractMatrix{T}, β::AbstractMatrix{T},
                            sqrtγ::AbstractMatrix{T},
                            bg::Background1D, t, xgrid) where {T}
    @assert size(a) == size(β) == size(sqrtγ) == size(xgrid)
    backend = get_backend(a)
    _sample_background_kernel!(backend)(a, β, sqrtγ, bg, T(t), xgrid;
                                        ndrange = length(a))
    return nothing
end

"""
    Wave1DWorkspace{T}

Preallocated scratch for [`wave1d_curved_rhs!`](@ref) — five `(N, M)`
buffers (`DΦ`, `F`, `DF`, two KO ping-pong buffers) plus the
per-node Kreiss-Oliger scale `h⁵ = (∂x/∂ξ)⁵`. Allocate one per
independent RHS evaluation context via [`make_wave1d_workspace`](@ref);
contents are overwritten on every call.
"""
struct Wave1DWorkspace{T, AT <: AbstractMatrix{T}}
    DΦ :: AT
    F  :: AT
    DF :: AT
    s1 :: AT
    s2 :: AT
    h5 :: AT   # per-node (element width)⁵ for the KO term
end

"""
    make_wave1d_workspace(geom::MeshGeometry{1, T, N}) → Wave1DWorkspace{T}

Allocate the RHS scratch on the same backend as `geom`.
"""
function make_wave1d_workspace(geom::MeshGeometry{1, T, N}) where {T, N}
    backend = get_backend(geom.coords)
    Ne = geom.Ne
    bufs = ntuple(_ -> allocate(backend, T, N, Ne), 5)
    h5 = allocate(backend, T, N, Ne)
    # geom.jac[1, 1, i, e] is the element width for affine line elements.
    copyto!(h5, reshape(geom.jac, N, Ne) .^ 5)
    return Wave1DWorkspace{T, typeof(h5)}(bufs..., h5)
end

"""
    wave1d_curved_rhs!(Φ̇, Π̇, Φ, Π, a, β;
                       geom, ops, ws, ε_KO = 0.1) → (Φ̇, Π̇)

1D scalar wave RHS on a 1+1 ADM background with space- and
time-varying lapse, shift, and spatial metric. Densitised-momentum
convention `Π := (√γ_xx/α) · (∂_t Φ − β · ∂_x Φ)`; see the file-level
comment for the derivation. The kernel evolves

    ∂_t Φ = β · ∂_x Φ + a · Π
    ∂_t Π = ∂_x ( a · ∂_x Φ + β · Π ),        a := α/√γ_xx.

Inputs `a`, `β` are `(N, M)` coefficient fields evaluated at the
current integrator stage time. `ws` is a [`Wave1DWorkspace`](@ref);
no allocations occur per call. `ε_KO` is the Kreiss-Oliger
coefficient (`0.1` is the standard NR default; `0.0` disables the
KO passes entirely).
"""
function wave1d_curved_rhs!(Φ̇::AbstractMatrix{T}, Π̇::AbstractMatrix{T},
                            Φ::AbstractMatrix{T}, Π::AbstractMatrix{T},
                            a::AbstractMatrix{T}, β::AbstractMatrix{T};
                            geom::MeshGeometry{1, T, N},
                            ops::SBPOps{N, T},
                            ws::Wave1DWorkspace{T},
                            ε_KO::Real = 0.1) where {N, T}
    @assert size(Φ) == size(Π) == size(Φ̇) == size(Π̇) == (N, geom.Ne)
    @assert size(a) == size(β) == (N, geom.Ne)
    εT = T(ε_KO)
    (; DΦ, F, DF, s1, s2, h5) = ws

    apply_D!(DΦ, Φ; geom, ops)
    @. F = a * DΦ + β * Π
    apply_D!(DF, F; geom, ops)

    @. Φ̇ = β * DΦ + a * Π
    @. Π̇ = DF

    if εT != 0
        # D⁶Φ: D¹Φ is already in DΦ; five more applications, ping-pong.
        apply_D!(s1, DΦ; geom, ops)   # D²Φ
        apply_D!(s2, s1; geom, ops)   # D³Φ
        apply_D!(s1, s2; geom, ops)   # D⁴Φ
        apply_D!(s2, s1; geom, ops)   # D⁵Φ
        apply_D!(s1, s2; geom, ops)   # D⁶Φ
        @. Φ̇ += εT * h5 * s1
        # D⁶Π.
        apply_D!(s1, Π;  geom, ops)   # D¹Π
        apply_D!(s2, s1; geom, ops)   # D²Π
        apply_D!(s1, s2; geom, ops)   # D³Π
        apply_D!(s2, s1; geom, ops)   # D⁴Π
        apply_D!(s1, s2; geom, ops)   # D⁵Π
        apply_D!(s2, s1; geom, ops)   # D⁶Π
        @. Π̇ += εT * h5 * s2
    end
    return Φ̇, Π̇
end

"""
    wave1d_energy(Φ, Π, sqrtγ; geom, ops, ws) → E

Discrete ADM energy

    E = ∫ ½ [ (Π/√γ)² + (∂_x Φ)²/γ ] · √γ dx
      ≈ Σ ½ [ (Π/√γ)² + (DΦ)²/γ ] · √γ · Hphys.

Exactly conserved by the continuum system only for static backgrounds
(∂_t α = ∂_t β = ∂_t γ = 0); use as a drift monitor otherwise.
Overwrites `ws.DΦ`.
"""
function wave1d_energy(Φ::AbstractMatrix{T}, Π::AbstractMatrix{T},
                       sqrtγ::AbstractMatrix{T};
                       geom::MeshGeometry{1, T, N},
                       ops::SBPOps{N, T},
                       ws::Wave1DWorkspace{T}) where {N, T}
    apply_D!(ws.DΦ, Φ; geom, ops)
    return sum(@. (
        ((Π / sqrtγ)^2 + (ws.DΦ / sqrtγ)^2) * sqrtγ * geom.Hphys / 2))
end
