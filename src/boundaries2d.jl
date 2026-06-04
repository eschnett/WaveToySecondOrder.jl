# Outer boundary conditions for the 2D ADM scalar wave
# (`wave2d_curved_rhs!`), mirroring boundaries1d.jl. Faces are
# classified from the *normal* characteristic structure: at a face
# with axis-aligned outward normal n̂·e_{dₙ}, the normal speeds are
# c± = −β^n ± a_n with β^n = n̂·β^{dₙ}, a_n = α√(γ^{dₙdₙ}) — the 1D
# analysis applied to the normal direction. So classification and
# admissibility reuse the 1D helpers (`classify_face1d`,
# `validate_bc1d`, `BC_*`).
#
# Radiative (Sommerfeld) / Dirichlet faces use the same
# characteristic-free field-radiation SAT as 1D: impose
# ∂_tΦ + a_n·∂_nΦ = data on the field, giving (per boundary node) the
# normalised residual
#     r = Π + ((β^n + a_n)/a)·(n̂·∂_nΦ),   a = α/√γ,
# driven to its target (0 = absorbing). The penalty acts on Π̇ with
# coefficient σ·|s_in|·invjac[dₙ,dₙ]/H_1d[face] (the per-axis weight of
# the interior `apply_D!` SAT), σ = 1. Confirmed energy-stable by the
# dense-operator spectrum tests (flat / small shift / anisotropic γ).
# Superluminal outflow → excision (no term); superluminal inflow →
# full-state Dirichlet (pin Φ, Π to data). Axis-aligned affine meshes.
#
# The BC pass reads ∂_dΦ from the workspace (`ws.DΦ1`, `ws.DΦ2`),
# which the RHS leaves populated (the KO chain uses separate scratch).
# CPU only for now; periodic 2D runs on GPU, non-periodic on CPU.

using HexSBPSAT: MeshGeometry, SBPOps
using KernelAbstractions: get_backend

# 2D face → (normal axis dₙ, outward sign n̂, along-axis node index).
@inline _face_geom2d(f, ::Val{N}) where {N} =
    (((f + 1) ÷ 2), (isodd(f) ? -1 : 1), (isodd(f) ? 1 : N))

"""
    classify_face2d(α, β1, β2, γ::SMatrix{2,2}, n̂axis, n̂sign;
                    sonic_tol) → Int

Characteristic class (`FACE_*`) of an axis-aligned 2D boundary face
with outward normal `n̂sign·e_{n̂axis}`, from the normal speed
`a_n = α√(γ^{n̂axis n̂axis})` and normal shift `β^{n̂axis}`. Delegates to
[`classify_face1d`](@ref).
"""
function classify_face2d(α::T, β1::T, β2::T, gu11::T, gu22::T,
                         n̂axis::Int, n̂sign::Int;
                         sonic_tol = eps(T)^(1//4)) where {T}
    a_n = α * sqrt(n̂axis == 1 ? gu11 : gu22)
    βax = n̂axis == 1 ? β1 : β2
    return classify_face1d(a_n, βax, n̂sign; sonic_tol)
end

"""
    make_bc2d(kinds::NTuple{4}; σ = 1, gΦ = nothing, gΠ = nothing)

Boundary bundle for [`wave2d_curved_rhs!`] (kwarg `bc2d`). `kinds`
gives the `BC_*` code (or Symbol) for each of the four face
directions (−x, +x, −y, +y) of the rectangular domain; only faces
tagged `bdry ≠ 0` are touched. `gΦ`/`gΠ` are optional `(N,N,Ne)` data
arrays (boundary-node entries only): `gΦ` is the field-radiation
target for `:dirichlet`, and `(gΦ, gΠ)` the state target for
`:full_dirichlet`. `:sommerfeld` (absorbing) and `:excision` need no
data.
"""
function make_bc2d(kinds; σ = 1, gΦ = nothing, gΠ = nothing)
    codes = ntuple(i -> (kinds[i] isa Symbol ? bc1d_kind(kinds[i]) :
                         Int(kinds[i])), 4)
    return (; kinds = codes, σ, gΦ, gΠ)
end

# Per-element boundary pass over the four faces (CPU). Reads ∂_dΦ from
# ws.DΦ1/DΦ2 (populated by the RHS).
function _apply_bc2d!(Φ̇::AbstractArray{T,3}, Π̇::AbstractArray{T,3},
                      Φ::AbstractArray{T,3}, Π::AbstractArray{T,3},
                      coef, ws; geom::MeshGeometry{2, T, N},
                      ops::SBPOps{N, T}, bc2d) where {N, T}
    get_backend(Φ̇) isa KernelAbstractions.CPU ||
        error("_apply_bc2d!: non-CPU boundary pass not implemented " *
              "(periodic meshes run on GPU; non-periodic on CPU)")
    DΦ1, DΦ2 = ws.DΦ1, ws.DΦ2
    H1 = ops.H
    σ  = T(bc2d.σ)
    conn = geom.conn
    @inbounds for m in 1:geom.Ne, f in 1:4
        conn.bdry[f, m] == 0 && continue
        kind = bc2d.kinds[f]
        kind == BC_EXCISION && continue
        dn, n̂, fn = _face_geom2d(f, Val(N))
        for t in 1:N
            i = dn == 1 ? fn : t
            j = dn == 1 ? t  : fn
            α  = coef.alpha[i, j, m]; sγ = coef.sqrtγ[i, j, m]
            gunn = dn == 1 ? coef.gu11[i, j, m] : coef.gu22[i, j, m]
            βax  = dn == 1 ? coef.b1[i, j, m]   : coef.b2[i, j, m]
            ij   = dn == 1 ? geom.invjac[1, 1, i, j, m] :
                             geom.invjac[2, 2, i, j, m]
            a   = α / sγ
            a_n = α * sqrt(gunn)
            βn  = n̂ * βax
            wt  = ij / H1[fn, fn]            # per-axis face weight
            if kind == BC_FULL_DIRICHLET
                # Both modes enter: pin the state to data.
                τ = σ * (abs(a_n - βax) + abs(a_n + βax)) * wt
                gΦv = bc2d.gΦ === nothing ? zero(T) : bc2d.gΦ[i, j, m]
                gΠv = bc2d.gΠ === nothing ? zero(T) : bc2d.gΠ[i, j, m]
                Φ̇[i, j, m] += -τ * (Φ[i, j, m] - gΦv)
                Π̇[i, j, m] += -τ * (Π[i, j, m] - gΠv)
            else
                # Field-radiation (Sommerfeld absorbing / Dirichlet).
                DΦn = dn == 1 ? DΦ1[i, j, m] : DΦ2[i, j, m]
                q   = n̂ * DΦn
                r   = Π[i, j, m] + ((βn + a_n) / a) * q
                g   = kind == BC_SOMMERFELD ? zero(T) :
                      (bc2d.gΦ === nothing ? zero(T) : bc2d.gΦ[i, j, m])
                s_in = a_n + βn
                Π̇[i, j, m] += -σ * s_in * wt * (r - g)
            end
        end
    end
    return nothing
end
