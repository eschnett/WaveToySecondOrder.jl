# Scalar wave equation on a prescribed 4-metric, fully second-order.
#
#   □_g φ ≡ g^{αβ} ∂_α ∂_β φ − Γ^γ ∂_γ φ = 0,    Γ^γ ≡ g^{αβ} Γ^γ_{αβ}
#
# Treated as a single covariant identity (no 3+1 split). The evolved
# state is (φ, φ̇) where φ̇ is the time derivative of φ stored
# alongside φ — exactly the second-order-ODE pair pattern used by
# `gh_rhs_element!`. The kernel isolates the only unknown second
# derivative `∂_tt φ`:
#
#   ∂_tt φ = (1 / g^{tt}) [ Γ^γ ∂_γ φ − 2 g^{ti} ∂_t∂_i φ − g^{ij} ∂_i∂_j φ ]
#
# with `∂_t φ = φ̇` (state) and `∂_t∂_i φ = ∂_i φ̇`.
#
# The 4-metric is *given* pointwise: callers supply pre-evaluated
# `g^{αβ}` (10 unique entries) and contracted `Γ^γ` (4 entries) at
# every GLL node via `eval_curved_background!`. The kernel itself is
# metric-agnostic.

# Component ordering for the 10 unique entries of a symmetric 4-tensor,
# matching the layout used elsewhere in this repo:
#
#   1 = tt   2 = tx   3 = ty   4 = tz
#            5 = xx   6 = xy   7 = xz
#                     8 = yy   9 = yz
#                             10 = zz

"""
    wave_curved_rhs_element!(φ̈, φ, φ̇, φ_face, φ̇_face, ginv, Γc, ops, h)

Per-element kernel for `□_g φ = 0` on a single Cartesian SBP element.
Mirrors the four-stage structure of `gh_rhs_element!`.

Arguments:
- `φ̈, φ, φ̇ :: AbstractArray{T,3}` of size `(N, N, N)`.
- `φ_face, φ̇_face :: NTuple{6, Matrix{T}}` — neighbour face traces of
  `φ` and `φ̇`, indexed `(−x, +x, −y, +y, −z, +z)`.
- `ginv :: AbstractArray{T,4}` of size `(10, N, N, N)` — 10 unique
  entries of `g^{αβ}` per node, packed as
  `(tt, tx, ty, tz, xx, xy, xz, yy, yz, zz)`.
- `Γc :: AbstractArray{T,4}` of size `(4, N, N, N)` — contracted
  Christoffels `(Γ^t, Γ^x, Γ^y, Γ^z)` per node.
- `ops`, `h` — SBP operators and physical element edge length.
"""
function wave_curved_rhs_element!(φ̈::AbstractArray{T, 3},
                                    φ::AbstractArray{T, 3},
                                    φ̇::AbstractArray{T, 3},
                                    φ_face ::NTuple{6, <:AbstractArray{T, 2}},
                                    φ̇_face::NTuple{6, <:AbstractArray{T, 2}},
                                    ginv::AbstractArray{T, 4},
                                    Γc  ::AbstractArray{T, 4},
                                    ops, h::T) where {T}
    @assert size(φ) == size(φ̇) == size(φ̈)
    N = size(φ, 1)
    @assert size(φ, 1) == size(φ, 2) == size(φ, 3) == N
    @assert size(ginv) == (10, N, N, N)
    @assert size(Γc)   == ( 4, N, N, N)

    inv_h = one(T) / h
    H_inv = SVector{N, T}(ntuple(i -> one(T) / ops.H[i, i], Val(N)))
    G     = SMatrix{N, N, T}(ops.G)

    # Stage 1: ∂_i φ and ∂_i φ̇ via SBP G along each spatial axis.
    du  = Array{T, 4}(undef, 3, N, N, N)
    du̇  = Array{T, 4}(undef, 3, N, N, N)
    for α in 1:3
        _scalar_grad!(@view(du[α, :, :, :]), φ, G, inv_h, Val(α), Val(N))
        _scalar_grad!(@view(du̇[α, :, :, :]), φ̇, G, inv_h, Val(α), Val(N))
    end

    # Stage 2: centred-flux SAT on the perpendicular component of each
    # face, applied to ∂_α φ and ∂_α φ̇ separately.
    @inbounds for f in 1:6
        α        = (f + 1) >> 1
        f_sign   = T(isodd(f) ? -1 : +1)
        face_row = isodd(f) ? 1 : N
        if α == 1
            _scalar_grad_sat!(@view(du[1,  :, :, :]), φ,  φ_face[f],
                                H_inv, inv_h, Val(1), f_sign, face_row, Val(N))
            _scalar_grad_sat!(@view(du̇[1, :, :, :]), φ̇, φ̇_face[f],
                                H_inv, inv_h, Val(1), f_sign, face_row, Val(N))
        elseif α == 2
            _scalar_grad_sat!(@view(du[2,  :, :, :]), φ,  φ_face[f],
                                H_inv, inv_h, Val(2), f_sign, face_row, Val(N))
            _scalar_grad_sat!(@view(du̇[2, :, :, :]), φ̇, φ̇_face[f],
                                H_inv, inv_h, Val(2), f_sign, face_row, Val(N))
        else
            _scalar_grad_sat!(@view(du[3,  :, :, :]), φ,  φ_face[f],
                                H_inv, inv_h, Val(3), f_sign, face_row, Val(N))
            _scalar_grad_sat!(@view(du̇[3, :, :, :]), φ̇, φ̇_face[f],
                                H_inv, inv_h, Val(3), f_sign, face_row, Val(N))
        end
    end

    # Stage 3: second spatial derivatives via a second SBP G sweep on
    # the SAT-corrected ∂_α φ. Six unique symmetric pairs:
    #   1 = xx   2 = xy   3 = xz
    #            4 = yy   5 = yz
    #                     6 = zz
    ddu = Array{T, 4}(undef, 6, N, N, N)
    _scalar_grad!(@view(ddu[1, :, :, :]), @view(du[1, :, :, :]),
                   G, inv_h, Val(1), Val(N))   # ∂_x (∂_x φ)
    _scalar_grad!(@view(ddu[2, :, :, :]), @view(du[1, :, :, :]),
                   G, inv_h, Val(2), Val(N))   # ∂_y (∂_x φ) = ∂_xy φ
    _scalar_grad!(@view(ddu[3, :, :, :]), @view(du[1, :, :, :]),
                   G, inv_h, Val(3), Val(N))   # ∂_z (∂_x φ) = ∂_xz φ
    _scalar_grad!(@view(ddu[4, :, :, :]), @view(du[2, :, :, :]),
                   G, inv_h, Val(2), Val(N))   # ∂_y (∂_y φ)
    _scalar_grad!(@view(ddu[5, :, :, :]), @view(du[2, :, :, :]),
                   G, inv_h, Val(3), Val(N))   # ∂_z (∂_y φ) = ∂_yz φ
    _scalar_grad!(@view(ddu[6, :, :, :]), @view(du[3, :, :, :]),
                   G, inv_h, Val(3), Val(N))   # ∂_z (∂_z φ)

    # Stage 4: pointwise wave-equation solve at every node.
    @inbounds for k in 1:N, j in 1:N, i in 1:N
        gtt = ginv[ 1, i, j, k]
        gtx = ginv[ 2, i, j, k]; gty = ginv[ 3, i, j, k]; gtz = ginv[ 4, i, j, k]
        gxx = ginv[ 5, i, j, k]; gxy = ginv[ 6, i, j, k]; gxz = ginv[ 7, i, j, k]
        gyy = ginv[ 8, i, j, k]; gyz = ginv[ 9, i, j, k]; gzz = ginv[10, i, j, k]

        Γt = Γc[1, i, j, k]; Γx = Γc[2, i, j, k]
        Γy = Γc[3, i, j, k]; Γz = Γc[4, i, j, k]

        φ̇_val = φ̇[i, j, k]
        dφx, dφy, dφz = du[1,i,j,k], du[2,i,j,k], du[3,i,j,k]
        dφ̇x, dφ̇y, dφ̇z = du̇[1,i,j,k], du̇[2,i,j,k], du̇[3,i,j,k]
        ddφ_xx = ddu[1,i,j,k]; ddφ_xy = ddu[2,i,j,k]; ddφ_xz = ddu[3,i,j,k]
        ddφ_yy = ddu[4,i,j,k]; ddφ_yz = ddu[5,i,j,k]; ddφ_zz = ddu[6,i,j,k]

        Γ_dot_dφ   = Γt * φ̇_val + Γx * dφx + Γy * dφy + Γz * dφz
        gti_dot    = gtx * dφ̇x + gty * dφ̇y + gtz * dφ̇z
        gij_dot_dd = gxx * ddφ_xx + gyy * ddφ_yy + gzz * ddφ_zz +
                     2*gxy * ddφ_xy + 2*gxz * ddφ_xz + 2*gyz * ddφ_yz

        φ̈[i, j, k] = (Γ_dot_dφ - 2 * gti_dot - gij_dot_dd) / gtt
    end

    return φ̈
end

"""
    wave_curved_rhs_mesh!(φ̈, φ, φ̇, mesh, ginv_all, Γc_all, ops, h)

Multi-element driver for the curved-background scalar wave RHS. Walks
every element, extracts face traces of `φ` and `φ̇` from neighbours via
`mesh.conn`, slices the per-element metric arrays, and calls
`wave_curved_rhs_element!`.

Restriction (matches `wave_lap_strong_conservative_mesh!`): uniform-cube
mesh with axis-aligned elements of equal edge length `h` and
`orientation ≡ 0`. The uniform-hex builders from `HexMeshes` satisfy
this; periodic boundaries work transparently because periodic seams
appear in `mesh.conn` as ordinary interior faces.
"""
function wave_curved_rhs_mesh!(φ̈::AbstractArray{T, 4},
                                  φ::AbstractArray{T, 4},
                                  φ̇::AbstractArray{T, 4},
                                  mesh,
                                  ginv_all::AbstractArray{T, 5},
                                  Γc_all  ::AbstractArray{T, 5},
                                  ops, h::T) where {T}
    N  = size(φ, 1)
    Ne = size(φ, 4)
    @assert size(φ) == size(φ̇) == size(φ̈) == (N, N, N, Ne)
    @assert size(ginv_all) == (10, N, N, N, Ne)
    @assert size(Γc_all)   == ( 4, N, N, N, Ne)
    @assert all(mesh.conn.orientation .== 0)

    φ_face  = ntuple(_ -> Matrix{T}(undef, N, N), Val(6))
    φ̇_face = ntuple(_ -> Matrix{T}(undef, N, N), Val(6))
    zero_face = zeros(T, N, N)

    for e in 1:Ne
        for f in 1:6
            e_nbr = Int(mesh.conn.neighbour[f, e])
            if e_nbr == 0
                copyto!(φ_face[f],  zero_face)
                copyto!(φ̇_face[f], zero_face)
            else
                f_nbr = Int(mesh.conn.neighbour_face[f, e])
                _extract_face!(φ_face[f],  view(φ,  :, :, :, e_nbr), f_nbr, N)
                _extract_face!(φ̇_face[f], view(φ̇, :, :, :, e_nbr), f_nbr, N)
            end
        end
        wave_curved_rhs_element!(view(φ̈, :, :, :, e),
                                   view(φ,  :, :, :, e),
                                   view(φ̇, :, :, :, e),
                                   φ_face, φ̇_face,
                                   view(ginv_all, :, :, :, :, e),
                                   view(Γc_all,   :, :, :, :, e),
                                   ops, h)
    end
    return φ̈
end

"""
    wave_curved_rhs_conservative_element!(φ̈, φ, φ̇, φ_face, φ̇_face,
                                            ginv, Γc, ops, h; τ = (N-1)²)

Conservative-SAT variant of `wave_curved_rhs_element!`. Same `□_g φ = 0`
on a prescribed 4-metric, but with the Mattsson–Nordström SAT pair from
`wave_lap_strong_conservative_element!` substituted for the centred-flux
SAT used in the original kernel.

Stages 1–2 compute the spatial gradients and Hessian via SBP `G` with
*no* SAT (free derivative inside each element). Stage 3 performs the
pointwise wave-equation solve using the raw gradients. Stage 4 then
adds the per-face conservative SAT pair (Neumann symmetrising +
Dirichlet jump penalty) as a correction on the assembled `φ̈`. The
discretisation reduces on Minkowski to `wave_lap_strong_conservative_element!`
byte-for-byte.

`φ̇_face` is accepted for API parity with the centred-flux variant
but is unused: the conservative scheme couples elements only through
`φ` (Dirichlet jump penalty), not `φ̇`.
"""
function wave_curved_rhs_conservative_element!(φ̈::AbstractArray{T, 3},
                                                  φ::AbstractArray{T, 3},
                                                  φ̇::AbstractArray{T, 3},
                                                  φ_face ::NTuple{6, <:AbstractArray{T, 2}},
                                                  φ̇_face::NTuple{6, <:AbstractArray{T, 2}},
                                                  ginv::AbstractArray{T, 4},
                                                  Γc  ::AbstractArray{T, 4},
                                                  ops, h::T;
                                                  τ::T = T((size(φ, 1) - 1)^2)) where {T}
    @assert size(φ) == size(φ̇) == size(φ̈)
    N = size(φ, 1)
    @assert size(φ, 1) == size(φ, 2) == size(φ, 3) == N
    @assert size(ginv) == (10, N, N, N)
    @assert size(Γc)   == ( 4, N, N, N)

    inv_h = one(T) / h
    H_inv = SVector{N, T}(ntuple(i -> one(T) / ops.H[i, i], Val(N)))
    G     = SMatrix{N, N, T}(ops.G)

    # Stage 1: ∂_i φ and ∂_i φ̇ via SBP G along each spatial axis. NO SAT
    # — the boundary correction lives in stage 4 as a value-and-gradient
    # penalty applied directly to the assembled `φ̈`.
    du  = Array{T, 4}(undef, 3, N, N, N)
    du̇  = Array{T, 4}(undef, 3, N, N, N)
    for α in 1:3
        _scalar_grad!(@view(du[α, :, :, :]), φ, G, inv_h, Val(α), Val(N))
        _scalar_grad!(@view(du̇[α, :, :, :]), φ̇, G, inv_h, Val(α), Val(N))
    end

    # Stage 2: second SBP G sweep for the six unique spatial Hessian
    # components. No SAT.
    ddu = Array{T, 4}(undef, 6, N, N, N)
    _scalar_grad!(@view(ddu[1, :, :, :]), @view(du[1, :, :, :]),
                   G, inv_h, Val(1), Val(N))   # ∂_xx φ
    _scalar_grad!(@view(ddu[2, :, :, :]), @view(du[1, :, :, :]),
                   G, inv_h, Val(2), Val(N))   # ∂_xy φ
    _scalar_grad!(@view(ddu[3, :, :, :]), @view(du[1, :, :, :]),
                   G, inv_h, Val(3), Val(N))   # ∂_xz φ
    _scalar_grad!(@view(ddu[4, :, :, :]), @view(du[2, :, :, :]),
                   G, inv_h, Val(2), Val(N))   # ∂_yy φ
    _scalar_grad!(@view(ddu[5, :, :, :]), @view(du[2, :, :, :]),
                   G, inv_h, Val(3), Val(N))   # ∂_yz φ
    _scalar_grad!(@view(ddu[6, :, :, :]), @view(du[3, :, :, :]),
                   G, inv_h, Val(3), Val(N))   # ∂_zz φ

    # Stage 3: pointwise wave-equation solve with the raw (no-SAT)
    # Hessian and gradients.
    @inbounds for k in 1:N, j in 1:N, i in 1:N
        gtt = ginv[ 1, i, j, k]
        gtx = ginv[ 2, i, j, k]; gty = ginv[ 3, i, j, k]; gtz = ginv[ 4, i, j, k]
        gxx = ginv[ 5, i, j, k]; gxy = ginv[ 6, i, j, k]; gxz = ginv[ 7, i, j, k]
        gyy = ginv[ 8, i, j, k]; gyz = ginv[ 9, i, j, k]; gzz = ginv[10, i, j, k]

        Γt = Γc[1, i, j, k]; Γx = Γc[2, i, j, k]
        Γy = Γc[3, i, j, k]; Γz = Γc[4, i, j, k]

        φ̇_val = φ̇[i, j, k]
        dφx, dφy, dφz = du[1,i,j,k], du[2,i,j,k], du[3,i,j,k]
        dφ̇x, dφ̇y, dφ̇z = du̇[1,i,j,k], du̇[2,i,j,k], du̇[3,i,j,k]
        ddφ_xx = ddu[1,i,j,k]; ddφ_xy = ddu[2,i,j,k]; ddφ_xz = ddu[3,i,j,k]
        ddφ_yy = ddu[4,i,j,k]; ddφ_yz = ddu[5,i,j,k]; ddφ_zz = ddu[6,i,j,k]

        Γ_dot_dφ   = Γt * φ̇_val + Γx * dφx + Γy * dφy + Γz * dφz
        gti_dot    = gtx * dφ̇x + gty * dφ̇y + gtz * dφ̇z
        gij_dot_dd = gxx * ddφ_xx + gyy * ddφ_yy + gzz * ddφ_zz +
                     2*gxy * ddφ_xy + 2*gxz * ddφ_xz + 2*gyz * ddφ_yz

        φ̈[i, j, k] = (Γ_dot_dφ - 2 * gti_dot - gij_dot_dd) / gtt
    end

    # Stage 4: per-face Mattsson–Nordström SAT pair, applied as a
    # correction to the assembled `φ̈`. Byte-identical to the SAT in
    # `wave_lap_strong_conservative_element!`.
    τ_over_h = τ * inv_h
    @inbounds for f in 1:6
        α        = (f + 1) >> 1
        f_sign   = T(isodd(f) ? -1 : +1)
        face_row = isodd(f) ? 1 : N
        coef_neu = -f_sign * inv_h * H_inv[face_row]
        coef_dir = -τ_over_h *        H_inv[face_row]
        uf       = φ_face[f]
        for q in 1:N, p in 1:N
            i, j, k = (α == 1 ? (face_row, p, q) :
                       α == 2 ? (p, face_row, q) :
                                (p, q, face_row))
            du_α   = (α == 1 ? du[1, i, j, k] :
                      α == 2 ? du[2, i, j, k] :
                               du[3, i, j, k])
            φ̈[i, j, k] += coef_neu * du_α
            φ̈[i, j, k] += coef_dir * (φ[i, j, k] - uf[p, q])
        end
    end

    return φ̈
end

"""
    wave_curved_rhs_conservative_mesh!(φ̈, φ, φ̇, mesh, ginv_all, Γc_all, ops, h; τ = (N-1)²)

Multi-element driver for the conservative-SAT curved-background RHS.
Identical loop body to `wave_curved_rhs_mesh!` — walks elements,
extracts face traces from neighbours via `mesh.conn`, calls the
per-element kernel. The `φ̇` face traces are extracted (uniformity
with the centred-flux driver) but unused by the conservative kernel.
"""
function wave_curved_rhs_conservative_mesh!(φ̈::AbstractArray{T, 4},
                                                φ::AbstractArray{T, 4},
                                                φ̇::AbstractArray{T, 4},
                                                mesh,
                                                ginv_all::AbstractArray{T, 5},
                                                Γc_all  ::AbstractArray{T, 5},
                                                ops, h::T;
                                                τ::T = T((size(φ, 1) - 1)^2)) where {T}
    N  = size(φ, 1)
    Ne = size(φ, 4)
    @assert size(φ) == size(φ̇) == size(φ̈) == (N, N, N, Ne)
    @assert size(ginv_all) == (10, N, N, N, Ne)
    @assert size(Γc_all)   == ( 4, N, N, N, Ne)
    @assert all(mesh.conn.orientation .== 0)

    φ_face  = ntuple(_ -> Matrix{T}(undef, N, N), Val(6))
    φ̇_face = ntuple(_ -> Matrix{T}(undef, N, N), Val(6))
    zero_face = zeros(T, N, N)

    for e in 1:Ne
        for f in 1:6
            e_nbr = Int(mesh.conn.neighbour[f, e])
            if e_nbr == 0
                copyto!(φ_face[f],  zero_face)
                copyto!(φ̇_face[f], zero_face)
            else
                f_nbr = Int(mesh.conn.neighbour_face[f, e])
                _extract_face!(φ_face[f],  view(φ,  :, :, :, e_nbr), f_nbr, N)
                _extract_face!(φ̇_face[f], view(φ̇, :, :, :, e_nbr), f_nbr, N)
            end
        end
        wave_curved_rhs_conservative_element!(view(φ̈, :, :, :, e),
                                                view(φ,  :, :, :, e),
                                                view(φ̇, :, :, :, e),
                                                φ_face, φ̇_face,
                                                view(ginv_all, :, :, :, :, e),
                                                view(Γc_all,   :, :, :, :, e),
                                                ops, h; τ = τ)
    end
    return φ̈
end

"""
    eval_curved_background!(ginv_all, Γc_all, mesh, xs, metric_obj, t, h)

Pre-evaluate the inverse 4-metric and contracted Christoffel symbols
at every GLL node of every mesh element from a `SpacetimeMetrics`
metric object at time `t`. `xs` is the vector of reference GLL nodes
on `[0, 1]` (typically `make_element(T, N).xs`). Mutates `ginv_all`
(shape `(10, N, N, N, Ne)`) and `Γc_all` (shape `(4, N, N, N, Ne)`).
"""
function eval_curved_background!(ginv_all::AbstractArray{T, 5},
                                   Γc_all  ::AbstractArray{T, 5},
                                   mesh, xs::AbstractVector{T},
                                   metric_obj, t::T, h::T) where {T}
    N  = size(ginv_all, 2)
    Ne = size(ginv_all, 5)
    @assert length(xs) == N
    @assert size(ginv_all) == (10, N, N, N, Ne)
    @assert size(Γc_all)   == ( 4, N, N, N, Ne)

    @inbounds for e in 1:Ne
        v_lo = mesh.vertex_idx[1, e]
        x0 = mesh.vertex_coords[1, v_lo]
        y0 = mesh.vertex_coords[2, v_lo]
        z0 = mesh.vertex_coords[3, v_lo]
        for k in 1:N, j in 1:N, i in 1:N
            p = SVector{4, T}(t,
                              x0 + h * xs[i],
                              y0 + h * xs[j],
                              z0 + h * xs[k])
            g4    = SpacetimeMetrics.metric(metric_obj, p)
            ginv4 = inv(g4)
            Γ     = SpacetimeMetrics.ChristoffelSymbols(metric_obj, p)
            # Contracted Christoffel Γ^γ = g^{αβ} Γ^γ_{αβ}.
            for γ in 1:4
                s = zero(T)
                for α in 1:4, β in 1:4
                    s += ginv4[α, β] * Γ[γ, α, β]
                end
                Γc_all[γ, i, j, k, e] = s
            end
            # Pack ginv4 into the 10-component layout.
            ginv_all[ 1, i, j, k, e] = ginv4[1, 1]   # tt
            ginv_all[ 2, i, j, k, e] = ginv4[1, 2]   # tx
            ginv_all[ 3, i, j, k, e] = ginv4[1, 3]   # ty
            ginv_all[ 4, i, j, k, e] = ginv4[1, 4]   # tz
            ginv_all[ 5, i, j, k, e] = ginv4[2, 2]   # xx
            ginv_all[ 6, i, j, k, e] = ginv4[2, 3]   # xy
            ginv_all[ 7, i, j, k, e] = ginv4[2, 4]   # xz
            ginv_all[ 8, i, j, k, e] = ginv4[3, 3]   # yy
            ginv_all[ 9, i, j, k, e] = ginv4[3, 4]   # yz
            ginv_all[10, i, j, k, e] = ginv4[4, 4]   # zz
        end
    end
    return (ginv_all, Γc_all)
end
