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

# Axis-aligned affine outer-boundary pass: per-face BC kind, axis-aligned
# outward normal (diagonal invjac). Reads ∂_dΦ from ws.DΦ1/DΦ2 (populated
# by the RHS). A single KA kernel (parallelised over output nodes, so a
# corner node touched by two faces accumulates both race-free) runs on
# both CPU and GPU.
function _apply_bc2d!(Φ̇::AbstractArray{T,3}, Π̇::AbstractArray{T,3},
                      Φ::AbstractArray{T,3}, Π::AbstractArray{T,3},
                      coef, ws; geom::MeshGeometry{2, T, N},
                      ops::SBPOps{N, T}, bc2d) where {N, T}
    backend = get_backend(Φ̇)
    gΦ = bc2d.gΦ === nothing ? Φ : bc2d.gΦ
    gΠ = bc2d.gΠ === nothing ? Φ : bc2d.gΠ
    k = bc2d.kinds
    _bc2d_affine_kernel!(backend, (N, N))(
        Φ̇, Π̇, Φ, Π, ws.DΦ1, ws.DΦ2, coef.alpha, coef.sqrtγ,
        coef.gu11, coef.gu22, coef.b1, coef.b2, geom.invjac,
        geom.conn.bdry, ops, gΦ, gΠ, bc2d.gΦ === nothing, bc2d.gΠ === nothing,
        Int32(k[1]), Int32(k[2]), Int32(k[3]), Int32(k[4]),
        T(bc2d.σ), Val(N); ndrange = (N, N, geom.Ne))
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
    kind = bc2d.kinds[1]               # single outer BC kind
    kind == BC_EXCISION && return nothing
    # Substitute the field arrays as dummies where no data is given (the
    # corresponding kind-branch in the kernel never reads them).
    gΦ  = bc2d.gΦ  === nothing ? Φ : bc2d.gΦ
    gΠ  = bc2d.gΠ  === nothing ? Φ : bc2d.gΠ
    gDx = bc2d.gDx === nothing ? Φ : bc2d.gDx
    gDy = bc2d.gDy === nothing ? Φ : bc2d.gDy
    # One KA kernel (per output node) on both CPU and GPU.
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

# GPU/CPU form of the axis-aligned affine boundary pass `_apply_bc2d!`.
# Parallelised over output NODES; each workitem loops over the ≤ 2
# boundary faces it lies on and accumulates locally, writing once (no
# corner-node race). Per-face BC code k1..k4 (faces −x,+x,−y,+y); axis-
# aligned normal sign n̂ and the diagonal invjac give the weight. Mirrors
# the previous CPU loop line-for-line. `noΦ/noΠ` flag missing data arrays.
@kernel function _bc2d_affine_kernel!(Fdot, Pdot, @Const(F), @Const(P),
                                      @Const(DΦ1), @Const(DΦ2), @Const(alpha),
                                      @Const(sqrtγ), @Const(gu11), @Const(gu22),
                                      @Const(b1), @Const(b2), @Const(invjac),
                                      @Const(bdry), ops, @Const(gΦ), @Const(gΠ),
                                      noΦ, noΠ, k1, k2, k3, k4, σ,
                                      ::Val{N}) where {N}
    i, j, m = @index(Global, NTuple)
    T = eltype(Fdot)
    dF = zero(T); dP = zero(T)
    @inbounds for f in 1:4
        bdry[f, m] == 0 && continue
        kind = f == 1 ? k1 : f == 2 ? k2 : f == 3 ? k3 : k4
        kind == Int32(BC_EXCISION) && continue
        dn = (f + 1) ÷ 2
        fn = isodd(f) ? 1 : N
        on = dn == 1 ? (i == fn) : (j == fn)
        on || continue
        n̂ = isodd(f) ? -one(T) : one(T)
        α = alpha[i,j,m]; sγ = sqrtγ[i,j,m]
        gunn = dn == 1 ? gu11[i,j,m] : gu22[i,j,m]
        βax  = dn == 1 ? b1[i,j,m]   : b2[i,j,m]
        ij   = dn == 1 ? invjac[1,1,i,j,m] : invjac[2,2,i,j,m]
        a   = α / sγ
        a_n = α * sqrt(gunn)
        βn  = n̂ * βax
        wt  = ij / ops.H[fn, fn]
        if kind == Int32(BC_FULL_DIRICHLET)
            τ = σ * (abs(a_n - βax) + abs(a_n + βax)) * wt
            gΦv = noΦ ? zero(T) : gΦ[i,j,m]
            gΠv = noΠ ? zero(T) : gΠ[i,j,m]
            dF += -τ * (F[i,j,m] - gΦv)
            dP += -τ * (P[i,j,m] - gΠv)
        else
            DΦn = dn == 1 ? DΦ1[i,j,m] : DΦ2[i,j,m]
            q   = n̂ * DΦn
            r   = P[i,j,m] + ((βn + a_n) / a) * q
            g   = kind == Int32(BC_SOMMERFELD) ? zero(T) :
                  (noΦ ? zero(T) : gΦ[i,j,m])
            s_in = a_n + βn
            dP += -σ * s_in * wt * (r - g)
        end
    end
    @inbounds Fdot[i,j,m] += dF
    @inbounds Pdot[i,j,m] += dP
end
