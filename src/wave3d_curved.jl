# 3D scalar wave on a 3+1 ADM background with lapse α(t,x,y,z), shift
# βⁱ(t,x,y,z), and spatial metric γ_ij(t,x,y,z). State (Φ, Π),
# Π := (√γ/α)(∂_tΦ − βⁱ∂_iΦ). The covariant wave equation
# ∂_μ(√|g| g^{μν} ∂_νΦ) = 0 (√|g| = α√γ) gives the flux-conservative
# first-order system
#
#     ∂_t Φ = βⁱ ∂_i Φ + (α/√γ) Π
#     ∂_t Π = ∂_i ( βⁱ Π + α√γ γ^{ij} ∂_j Φ ),     i, j ∈ {x, y, z}
#
# The 3D analog of `wave2d_curved.jl`. The flux Fⁱ = βⁱΠ + α√γ γ^{ij}∂_jΦ
# is differentiated with the per-axis `HexSBPSAT.apply_D!(·, d)` (affine
# meshes). Kreiss–Oliger dissipation ε·μ⁻⁵·D⁶ is applied per axis.
# Curvilinear meshes (free-stream-preserving conservative-curl operators)
# are Milestone 2.
#
# CAPITAL Greek Φ, Π.

using HexSBPSAT: MeshGeometry, SBPOps, MeshWorkspace, make_workspace, apply_D!
using KernelAbstractions: @kernel, @index, @Const, get_backend, allocate
using SpacetimeMetrics: adm_decompose
using StaticArrays: SVector, SMatrix

################################################################################
# ADM backgrounds (3D)

"""
    Background3D

Abstract 3+1 ADM background (α, βⁱ, γ_ij) for the 3D scalar wave.
Concrete subtypes implement
`_bg_adm3(bg, t, x, y, z) → (α, β1, β2, β3, γ11, γ12, γ13, γ22, γ23, γ33)`
(covariant spatial metric); sample onto the grid with
[`sample_background3d!`].
"""
abstract type Background3D end

"""
    AnalyticBackground3D(α_fn, β_fn, γ_fn) <: Background3D

Closures `α_fn(t,x,y,z)→α`, `β_fn(t,x,y,z)→(β1,β2,β3)`,
`γ_fn(t,x,y,z)→(γ11,γ12,γ13,γ22,γ23,γ33)`. Closures must not capture
`Type` objects if the background is to be passed into GPU kernels.
"""
struct AnalyticBackground3D{Fα, Fβ, Fγ} <: Background3D
    α_fn :: Fα
    β_fn :: Fβ
    γ_fn :: Fγ
end

@inline function _bg_adm3(bg::AnalyticBackground3D, t, x, y, z)
    γ11, γ12, γ13, γ22, γ23, γ33 = bg.γ_fn(t, x, y, z)
    β1, β2, β3 = bg.β_fn(t, x, y, z)
    return bg.α_fn(t, x, y, z), β1, β2, β3, γ11, γ12, γ13, γ22, γ23, γ33
end

"""
    MetricBackground3D(m) <: Background3D

Background from a `SpacetimeMetrics.AbstractMetric`; ADM variables via
`adm_decompose(m, (t,x,y,z))`.
"""
struct MetricBackground3D{M} <: Background3D
    metric :: M
end

@inline function _bg_adm3(bg::MetricBackground3D, t, x, y, z)
    α, β, γ = adm_decompose(bg.metric, SVector(t, x, y, z))
    return α, β[1], β[2], β[3],
           γ[1,1], γ[1,2], γ[1,3], γ[2,2], γ[2,3], γ[3,3]
end

# Coefficient fields the RHS consumes, per node: lapse, √γ, shift, and the
# contravariant spatial metric γ^{ij} (6 independent components).
@kernel function _sample_bg3d_kernel!(alpha, sqrtγ, b1, b2, b3,
                                      gu11, gu12, gu13, gu22, gu23, gu33,
                                      bg, t::T, @Const(xg), @Const(yg),
                                      @Const(zg)) where {T}
    idx = @index(Global, Linear)
    @inbounds begin
        αv, β1, β2, β3, g11, g12, g13, g22, g23, g33 =
            _bg_adm3(bg, t, xg[idx], yg[idx], zg[idx])
        g11 = T(g11); g12 = T(g12); g13 = T(g13)
        g22 = T(g22); g23 = T(g23); g33 = T(g33)
        det = g11*(g22*g33 - g23*g23) - g12*(g12*g33 - g23*g13) +
              g13*(g12*g23 - g22*g13)
        alpha[idx] = T(αv); sqrtγ[idx] = sqrt(det)
        b1[idx] = T(β1); b2[idx] = T(β2); b3[idx] = T(β3)
        gu11[idx] = (g22*g33 - g23*g23) / det
        gu12[idx] = (g13*g23 - g12*g33) / det
        gu13[idx] = (g12*g23 - g13*g22) / det
        gu22[idx] = (g11*g33 - g13*g13) / det
        gu23[idx] = (g12*g13 - g11*g23) / det
        gu33[idx] = (g11*g22 - g12*g12) / det
    end
end

"""
    sample_background3d!(coef, bg::Background3D, t, xg, yg, zg)

Fill the coefficient NamedTuple `coef = (; alpha, sqrtγ, b1, b2, b3,
gu11, gu12, gu13, gu22, gu23, gu33)` (each `(N,N,N,Ne)`) from `bg` at
time `t` on the collocation grids. A KernelAbstractions kernel —
allocation-free, GPU-compatible.
"""
function sample_background3d!(coef, bg::Background3D, t, xg, yg, zg)
    backend = get_backend(coef.alpha)
    T = eltype(coef.alpha)
    _sample_bg3d_kernel!(backend)(coef.alpha, coef.sqrtγ, coef.b1, coef.b2,
                                  coef.b3, coef.gu11, coef.gu12, coef.gu13,
                                  coef.gu22, coef.gu23, coef.gu33,
                                  bg, T(t), xg, yg, zg;
                                  ndrange = length(coef.alpha))
    return nothing
end

# Allocate the coefficient bundle on the same backend as `geom`.
function make_coef3d(geom::MeshGeometry{3, T, N}) where {T, N}
    backend = get_backend(geom.coords)
    mk() = allocate(backend, T, N, N, N, geom.Ne)
    return (; alpha = mk(), sqrtγ = mk(), b1 = mk(), b2 = mk(), b3 = mk(),
            gu11 = mk(), gu12 = mk(), gu13 = mk(),
            gu22 = mk(), gu23 = mk(), gu33 = mk())
end

################################################################################
# RHS workspace + kernel

"""
    Wave3DWorkspace{T}

Scratch for [`wave3d_curved_rhs!`](@ref): nine `(N,N,N,Ne)` buffers
(`DΦ1, DΦ2, DΦ3, F1, F2, F3, s1, s2, s3`), the operator face-trace
workspace `mw`, and the first-derivative spectral radius `μ` with the
Kreiss–Oliger scale `μ⁻⁵`.
"""
struct Wave3DWorkspace{T, AT <: AbstractArray{T,4}, MW}
    DΦ1 :: AT
    DΦ2 :: AT
    DΦ3 :: AT
    F1  :: AT
    F2  :: AT
    F3  :: AT
    s1  :: AT
    s2  :: AT
    s3  :: AT
    mw  :: MW
    μ      :: T
    inv_μ5 :: T
end

"""
    make_wave3d_workspace(geom::MeshGeometry{3, T, N}, ops) → Wave3DWorkspace

Allocate scratch and compute the spectral radius `μ` of the per-axis
`apply_D!` by power iteration on `D²` along axis 1.
"""
function make_wave3d_workspace(geom::MeshGeometry{3, T, N}, ops) where {T, N}
    backend = get_backend(geom.coords)
    Ne = geom.Ne
    bufs = ntuple(_ -> allocate(backend, T, N, N, N, Ne), 9)
    mw = make_workspace(geom)
    v, w = bufs[8], bufs[9]
    v_host = T[isodd(i+j+k+m) ? one(T) : -one(T)
               for i in 1:N, j in 1:N, k in 1:N, m in 1:Ne]
    copyto!(v, v_host)
    μ² = zero(T)
    for _ in 1:30
        apply_D!(w, v, 1; geom, ops, work = mw)
        apply_D!(v, w, 1; geom, ops, work = mw)
        μ² = sqrt(sum(abs2, v))
        μ² > 0 || break
        v ./= μ²
    end
    μ = sqrt(μ²)
    inv_μ5 = μ > 0 ? inv(μ)^5 : zero(T)
    return Wave3DWorkspace{T, typeof(v), typeof(mw)}(bufs..., mw, μ, inv_μ5)
end

# Six-fold per-axis KO application: out += koT · D_d⁶ field, ping-pong
# through s1/s2. `field` is not modified.
@inline function _ko_axis3!(dst, field, d, koT, ws, geom, ops)
    s1, s2 = ws.s1, ws.s2; mw = ws.mw
    apply_D!(s1, field, d; geom, ops, work = mw)
    apply_D!(s2, s1, d; geom, ops, work = mw)
    apply_D!(s1, s2, d; geom, ops, work = mw)
    apply_D!(s2, s1, d; geom, ops, work = mw)
    apply_D!(s1, s2, d; geom, ops, work = mw)
    apply_D!(s2, s1, d; geom, ops, work = mw)   # D⁶ field in s2
    @. dst += koT * s2
    return nothing
end

"""
    wave3d_curved_rhs!(Φ̇, Π̇, Φ, Π, coef; geom, ops, ws, ε_KO = 0.1,
                       bc3d = nothing, metric = nothing) → (Φ̇, Π̇)

3D scalar-wave RHS on a 3+1 ADM background. `coef` from
[`sample_background3d!`] at the current stage time. Flux
`Fⁱ = βⁱΠ + α√γ γ^{ij}∂_jΦ`, evolved as `∂_tΦ = βⁱ∂_iΦ + (α/√γ)Π`,
`∂_tΠ = ∂_iFⁱ`. `metric === nothing` selects the axis-aligned affine
path (per-axis `apply_D!`); curvilinear (`metric` provided) is Milestone
2. Allocation-free.
"""
function wave3d_curved_rhs!(Φ̇::AbstractArray{T,4}, Π̇::AbstractArray{T,4},
                            Φ::AbstractArray{T,4}, Π::AbstractArray{T,4},
                            coef;
                            geom::MeshGeometry{3, T, N},
                            ops::SBPOps{N, T},
                            ws::Wave3DWorkspace{T},
                            ε_KO::Real = 0.1,
                            bc3d = nothing,
                            metric = nothing) where {N, T}
    (; DΦ1, DΦ2, DΦ3, F1, F2, F3, s1, s2, s3) = ws
    (; alpha, sqrtγ, b1, b2, b3, gu11, gu12, gu13, gu22, gu23, gu33) = coef
    εT = T(ε_KO); koT = εT * ws.inv_μ5; mw = ws.mw

    metric === nothing ||
        error("wave3d_curved_rhs!: curvilinear path is Milestone 2")

    apply_D!(DΦ1, Φ, 1; geom, ops, work = mw)   # affine per-axis gradient
    apply_D!(DΦ2, Φ, 2; geom, ops, work = mw)
    apply_D!(DΦ3, Φ, 3; geom, ops, work = mw)

    # Flux Fⁱ = βⁱΠ + α√γ γ^{ij}∂_jΦ.
    @. F1 = b1 * Π + alpha * sqrtγ * (gu11 * DΦ1 + gu12 * DΦ2 + gu13 * DΦ3)
    @. F2 = b2 * Π + alpha * sqrtγ * (gu12 * DΦ1 + gu22 * DΦ2 + gu23 * DΦ3)
    @. F3 = b3 * Π + alpha * sqrtγ * (gu13 * DΦ1 + gu23 * DΦ2 + gu33 * DΦ3)

    apply_D!(s1, F1, 1; geom, ops, work = mw)
    apply_D!(s2, F2, 2; geom, ops, work = mw)
    apply_D!(s3, F3, 3; geom, ops, work = mw)
    @. Π̇ = s1 + s2 + s3
    @. Φ̇ = b1 * DΦ1 + b2 * DΦ2 + b3 * DΦ3 + (alpha / sqrtγ) * Π

    if εT != 0
        _ko_axis3!(Φ̇, Φ, 1, koT, ws, geom, ops)
        _ko_axis3!(Φ̇, Φ, 2, koT, ws, geom, ops)
        _ko_axis3!(Φ̇, Φ, 3, koT, ws, geom, ops)
        _ko_axis3!(Π̇, Π, 1, koT, ws, geom, ops)
        _ko_axis3!(Π̇, Π, 2, koT, ws, geom, ops)
        _ko_axis3!(Π̇, Π, 3, koT, ws, geom, ops)
    end

    if bc3d !== nothing
        _apply_bc3d!(Φ̇, Π̇, Φ, Π, coef, ws; geom, ops, bc3d)
    end
    return Φ̇, Π̇
end

"""
    wave3d_energy(Φ, Π, coef; geom, ops, ws) → E

Discrete ADM energy `E = Σ ½[ Π²/√γ + √γ γ^{ij}∂_iΦ∂_jΦ ]·Hphys`. The
physical energy of the normal observers (exactly conserved only for
static backgrounds with α≡1; otherwise a drift monitor). Overwrites
`ws.DΦ1, ws.DΦ2, ws.DΦ3`.
"""
function wave3d_energy(Φ::AbstractArray{T,4}, Π::AbstractArray{T,4}, coef;
                       geom::MeshGeometry{3, T, N}, ops::SBPOps{N, T},
                       ws::Wave3DWorkspace{T}) where {N, T}
    (; DΦ1, DΦ2, DΦ3) = ws
    (; sqrtγ, gu11, gu12, gu13, gu22, gu23, gu33) = coef
    mw = ws.mw
    apply_D!(DΦ1, Φ, 1; geom, ops, work = mw)
    apply_D!(DΦ2, Φ, 2; geom, ops, work = mw)
    apply_D!(DΦ3, Φ, 3; geom, ops, work = mw)
    H = geom.Hphys
    return sum(@. (Π^2 / sqrtγ +
                   sqrtγ * (gu11*DΦ1^2 + gu22*DΦ2^2 + gu33*DΦ3^2 +
                            2*gu12*DΦ1*DΦ2 + 2*gu13*DΦ1*DΦ3 +
                            2*gu23*DΦ2*DΦ3)) * H / 2)
end

# isbits callable closures so the backgrounds pass into GPU kernels.
struct _Const4{T}; v::T; end
(f::_Const4)(t, x, y, z) = f.v
struct _ConstVec3{T}; b1::T; b2::T; b3::T; end
(f::_ConstVec3)(t, x, y, z) = (f.b1, f.b2, f.b3)
struct _ConstMet3{T}; g11::T; g12::T; g13::T; g22::T; g23::T; g33::T; end
(f::_ConstMet3)(t, x, y, z) = (f.g11, f.g12, f.g13, f.g22, f.g23, f.g33)
