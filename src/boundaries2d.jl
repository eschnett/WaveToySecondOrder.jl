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
data. On curvilinear meshes, `:dirichlet` instead takes the exact
solution's boundary data `gΠ` (Π) and `gDx`/`gDy` (∂_xΦ, ∂_yΦ); the
pass forms the field-radiation target with its own physical normal.
"""
function make_bc2d(kinds; σ = 1, gΦ = nothing, gΠ = nothing,
                   gDx = nothing, gDy = nothing)
    codes = ntuple(i -> (kinds[i] isa Symbol ? bc1d_kind(kinds[i]) :
                         Int(kinds[i])), 4)
    return (; kinds = codes, σ, gΦ, gΠ, gDx, gDy)
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

# Curvilinear outer-boundary pass: a single BC `kind` applied to every
# `bdry ≠ 0` face, using the PHYSICAL outward normal from the discrete
# metric terms. The field-radiation residual and penalty generalise the
# axis-aligned version with n = (nflux)/JF, a_n = α√(γ^{ij}n_in_j),
# β^n = βⁱn_i, and the boundary weight JF·invdetJ/H_1d[row] (which
# reduces to invjac[dₙ,dₙ]/H_1d on axis-aligned affine faces). Reads the
# physical gradient from `ws.DΦ1/DΦ2`. `kind` is the single outer BC
# code; data via `bc2d.gΦ/gΠ` (boundary-node entries).
function _apply_bc2d_curv!(Φ̇::AbstractArray{T,3}, Π̇::AbstractArray{T,3},
                           Φ::AbstractArray{T,3}, Π::AbstractArray{T,3},
                           coef, ws, metric; geom::MeshGeometry{2, T, N},
                           ops::SBPOps{N, T}, bc2d) where {N, T}
    backend = get_backend(Φ̇)
    if !(backend isa KernelAbstractions.CPU)
        kind = bc2d.kinds[1]
        kind == BC_EXCISION && return nothing
        # Substitute the field arrays as dummies where no data is given
        # (the corresponding kind-branch in the kernel never reads them).
        gΦ  = bc2d.gΦ  === nothing ? Φ : bc2d.gΦ
        gΠ  = bc2d.gΠ  === nothing ? Φ : bc2d.gΠ
        gDx = bc2d.gDx === nothing ? Φ : bc2d.gDx
        gDy = bc2d.gDy === nothing ? Φ : bc2d.gDy
        _bc2d_curv_kernel!(backend, (N, N))(
            Φ̇, Π̇, Φ, Π, ws.DΦ1, ws.DΦ2,
            coef.alpha, coef.sqrtγ, coef.gu11, coef.gu12, coef.gu22,
            coef.b1, coef.b2, geom.jac, geom.detjac, geom.handedness,
            geom.conn.bdry, ops, gΦ, gΠ, gDx, gDy,
            bc2d.gΦ === nothing, bc2d.gΠ === nothing,
            bc2d.gDx === nothing, bc2d.gDy === nothing,
            Int32(kind), T(bc2d.σ), Val(N); ndrange = (N, N, geom.Ne))
        return nothing
    end
    DΦ1, DΦ2 = ws.DΦ1, ws.DΦ2
    H1 = ops.H; σ = T(bc2d.σ); conn = geom.conn
    kind = bc2d.kinds[1]               # single outer BC kind
    @inbounds for m in 1:geom.Ne, f in 1:4
        conn.bdry[f, m] == 0 && continue
        kind == BC_EXCISION && continue
        a_idx  = (f + 1) ÷ 2                       # normal reference axis
        axis_p = a_idx == 1 ? 2 : 1                # tangent axis
        sgn_f  = isodd(f) ? -one(T) : one(T)
        sgn_c  = a_idx == 1 ? one(T) : -one(T)
        row = isodd(f) ? 1 : N
        for p in 1:N
            i = f ≤ 2 ? row : p
            j = f ≤ 2 ? p   : row
            # Outward unit normal + surface element from the analytic
            # Jacobian columns, exactly as the curvilinear Laplacian's
            # face SAT (`_face_sat_compute_2d!`): 90° rotation of the
            # face tangent, oriented by sgn_f·sgn_c·handedness.
            sgn_out = sgn_f * sgn_c * T(geom.handedness[m])
            tpx = geom.jac[1, axis_p, i, j, m]
            tpy = geom.jac[2, axis_p, i, j, m]
            nfx =  sgn_out * tpy
            nfy = -sgn_out * tpx
            JF  = sqrt(nfx*nfx + nfy*nfy)
            nx  = nfx / JF; ny = nfy / JF
            αv = coef.alpha[i,j,m]; sγ = coef.sqrtγ[i,j,m]
            g11 = coef.gu11[i,j,m]; g12 = coef.gu12[i,j,m]; g22 = coef.gu22[i,j,m]
            a   = αv / sγ
            a_n = αv * sqrt(g11*nx*nx + 2*g12*nx*ny + g22*ny*ny)
            βn  = coef.b1[i,j,m]*nx + coef.b2[i,j,m]*ny
            # SAT lift = surface element / volume mass-per-reference-face
            #          = JF / (H_1d[row]·detjac).
            wt  = JF / (H1[row, row] * geom.detjac[i,j,m])
            if kind == BC_FULL_DIRICHLET
                τ = σ * (abs(a_n - βn) + abs(a_n + βn)) * wt
                gΦv = bc2d.gΦ === nothing ? zero(T) : bc2d.gΦ[i,j,m]
                gΠv = bc2d.gΠ === nothing ? zero(T) : bc2d.gΠ[i,j,m]
                Φ̇[i,j,m] += -τ * (Φ[i,j,m] - gΦv)
                Π̇[i,j,m] += -τ * (Π[i,j,m] - gΠv)
            else
                q = nx*DΦ1[i,j,m] + ny*DΦ2[i,j,m]      # outward normal deriv
                r = Π[i,j,m] + ((βn + a_n) / a) * q
                # Target field-radiation residual: 0 for Sommerfeld
                # (absorbing); for Dirichlet, the residual of the
                # supplied exact-solution data (Π and ∇Φ at the
                # boundary node), formed with the same physical normal.
                g = if kind == BC_SOMMERFELD
                    zero(T)
                else
                    gΠv = bc2d.gΠ  === nothing ? zero(T) : bc2d.gΠ[i,j,m]
                    gDx = bc2d.gDx === nothing ? zero(T) : bc2d.gDx[i,j,m]
                    gDy = bc2d.gDy === nothing ? zero(T) : bc2d.gDy[i,j,m]
                    gΠv + ((βn + a_n) / a) * (nx*gDx + ny*gDy)
                end
                s_in = a_n + βn
                Π̇[i,j,m] += -σ * s_in * wt * (r - g)
            end
        end
    end
    return nothing
end

using KernelAbstractions: @kernel, @index, @Const

# GPU form of `_apply_bc2d_curv!`. Parallelised over NODES (i, j, e):
# each workitem owns its own output node and loops over the ≤ 2 boundary
# faces it lies on, accumulating into local increments and writing once
# — so a corner node touched by two faces gets both contributions with
# no cross-workitem race (per-face parallelisation would race there).
# `kind` is the single outer BC code (Int32); `noΦ/noΠ/noDx/noDy` flag a
# missing data array (its argument is then a dummy and never read). The
# arithmetic mirrors the CPU body line-for-line.
@kernel function _bc2d_curv_kernel!(Fdot, Pdot, @Const(F), @Const(P),
                                    @Const(DΦ1), @Const(DΦ2),
                                    @Const(alpha), @Const(sqrtγ),
                                    @Const(gu11), @Const(gu12), @Const(gu22),
                                    @Const(b1), @Const(b2), @Const(jac),
                                    @Const(detjac), @Const(handed),
                                    @Const(bdry), ops, @Const(gΦ), @Const(gΠ),
                                    @Const(gDx), @Const(gDy), noΦ, noΠ, noDx,
                                    noDy, kind, σ, ::Val{N}) where {N}
    i, j, m = @index(Global, NTuple)
    T = eltype(Fdot)
    dF = zero(T); dP = zero(T)
    @inbounds for f in 1:4
        bdry[f, m] == 0 && continue
        a_idx = (f + 1) ÷ 2
        row = isodd(f) ? 1 : N
        on = a_idx == 1 ? (i == row) : (j == row)
        on || continue
        axis_p = a_idx == 1 ? 2 : 1
        sgn_f = isodd(f) ? -one(T) : one(T)
        sgn_c = a_idx == 1 ? one(T) : -one(T)
        sgn_out = sgn_f * sgn_c * T(handed[m])
        tpx = jac[1, axis_p, i, j, m]; tpy = jac[2, axis_p, i, j, m]
        nfx =  sgn_out * tpy; nfy = -sgn_out * tpx
        JF = sqrt(nfx*nfx + nfy*nfy)
        nx = nfx / JF; ny = nfy / JF
        αv = alpha[i,j,m]; sγ = sqrtγ[i,j,m]
        g11 = gu11[i,j,m]; g12 = gu12[i,j,m]; g22 = gu22[i,j,m]
        a = αv / sγ
        a_n = αv * sqrt(g11*nx*nx + 2*g12*nx*ny + g22*ny*ny)
        βn = b1[i,j,m]*nx + b2[i,j,m]*ny
        wt = JF / (ops.H[row, row] * detjac[i,j,m])
        if kind == Int32(BC_FULL_DIRICHLET)
            τ = σ * (abs(a_n - βn) + abs(a_n + βn)) * wt
            gΦv = noΦ ? zero(T) : gΦ[i,j,m]
            gΠv = noΠ ? zero(T) : gΠ[i,j,m]
            dF += -τ * (F[i,j,m] - gΦv)
            dP += -τ * (P[i,j,m] - gΠv)
        else
            q = nx*DΦ1[i,j,m] + ny*DΦ2[i,j,m]
            r = P[i,j,m] + ((βn + a_n) / a) * q
            g = if kind == Int32(BC_SOMMERFELD)
                zero(T)
            else
                gΠv = noΠ  ? zero(T) : gΠ[i,j,m]
                gx  = noDx ? zero(T) : gDx[i,j,m]
                gy  = noDy ? zero(T) : gDy[i,j,m]
                gΠv + ((βn + a_n) / a) * (nx*gx + ny*gy)
            end
            s_in = a_n + βn
            dP += -σ * s_in * wt * (r - g)
        end
    end
    @inbounds Fdot[i,j,m] += dF
    @inbounds Pdot[i,j,m] += dP
end
