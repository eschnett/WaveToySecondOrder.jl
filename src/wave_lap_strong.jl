# Strong-form single-element scalar Laplacian.
#
# Alternative to `HexSBPSAT.apply_laplacian!` (SIPG/weak form):
# applies the SBP first-derivative operator `G` twice with a centred-
# flux SAT on the *first* derivative only. Operates on a single
# Cartesian element `[0, h]³` with Dirichlet face data supplied as
# six `(N, N)` traces.
#
# Computes `(L u)[i, j, k] = (∂²_x u + ∂²_y u + ∂²_z u)[i, j, k]` at
# every collocation node.
#
# Two variants live here:
#   * `wave_lap_strong_element!` — centred-flux SAT on the first
#     derivative only. Spectrum sits on the negative real axis (so
#     leapfrog is stable) but `H_phys · L` is *not* symmetric and
#     the discrete energy is not conserved.
#   * `wave_lap_strong_conservative_element!` — Mattsson–Nordström
#     SAT pair (Neumann-symmetrising correction + Dirichlet penalty)
#     applied to the wide-stencil `D D u`. Yields a discrete L with
#     `H_phys · L` symmetric and `-H_phys · L` PSD; semidiscrete
#     energy is exactly conserved.
# Both share the same call signature except for the conservative one
# taking a dimensionless penalty parameter `τ` (default `(N-1)²`).

# SBP `G` along reference axis `α ∈ {1, 2, 3}` on a scalar field,
# scaled by `inv_h`.
@inline function _scalar_grad!(du::AbstractArray{T, 3}, u::AbstractArray{T, 3},
                                G::AbstractMatrix{T}, inv_h::T,
                                ::Val{α}, ::Val{N}) where {α, N, T}
    @inbounds for k in 1:N, j in 1:N, i in 1:N
        s = zero(T)
        for p in 1:N
            uval = (α == 1 ? u[p, j, k] :
                    α == 2 ? u[i, p, k] :
                             u[i, j, p])
            gval = (α == 1 ? G[i, p] :
                    α == 2 ? G[j, p] :
                             G[k, p])
            s += gval * uval
        end
        du[i, j, k] = s * inv_h
    end
    return du
end

# Centred-flux SAT correction on `∂_α u` at face `f` (axis `α`
# perpendicular). `f_sign = -1` for the lower face, `+1` for the upper.
@inline function _scalar_grad_sat!(du_α::AbstractArray{T, 3},
                                     u::AbstractArray{T, 3},
                                     u_face::AbstractArray{T, 2},
                                     H_inv::SVector{N, T}, inv_h::T,
                                     ::Val{α}, f_sign::T, face_row::Int,
                                     ::Val{N}) where {α, N, T}
    coef = f_sign * (one(T) / 2) * H_inv[face_row] * inv_h
    @inbounds for q in 1:N, p in 1:N
        i, j, k = (α == 1 ? (face_row, p, q) :
                   α == 2 ? (p, face_row, q) :
                            (p, q, face_row))
        du_α[i, j, k] += coef * (u_face[p, q] - u[i, j, k])
    end
    return du_α
end

"""
    wave_lap_strong_element!(Lu, u, u_face, ops, h)

Scalar Laplacian `∇² u` on a single Cartesian SBP element of edge
length `h`, computed in strong form via two SBP `G`-sweeps with a
centred-flux SAT on the first.

`u, Lu` are `(N, N, N)` arrays. `u_face` is a 6-tuple of `(N, N)`
face traces in `(−x, +x, −y, +y, −z, +z)` order, indexed by the
face's two volume tangent axes (axis-1 face: `(j, k)`; axis-2 face:
`(i, k)`; axis-3 face: `(i, j)`).
"""
function wave_lap_strong_element!(Lu::AbstractArray{T, 3},
                                    u::AbstractArray{T, 3},
                                    u_face::NTuple{6, <:AbstractArray{T, 2}},
                                    ops, h::T) where {T}
    @assert size(u) == size(Lu)
    N = size(u, 1)
    @assert size(u, 1) == size(u, 2) == size(u, 3) == N

    inv_h = one(T) / h
    H_inv = SVector{N, T}(ntuple(i -> one(T) / ops.H[i, i], Val(N)))
    G     = SMatrix{N, N, T}(ops.G)

    # Stage 1: ∂_α u via SBP G along each axis.
    du = Array{T, 4}(undef, 3, N, N, N)
    for α in 1:3
        _scalar_grad!(@view(du[α, :, :, :]), u, G, inv_h, Val(α), Val(N))
    end

    # Stage 2: centred-flux SAT at the 6 faces. Each face affects only
    # the perpendicular component of `∂_α u`.
    @inbounds for f in 1:6
        α        = (f + 1) >> 1
        f_sign   = T(isodd(f) ? -1 : +1)
        face_row = isodd(f) ? 1 : N
        if α == 1
            _scalar_grad_sat!(@view(du[1, :, :, :]), u, u_face[f],
                                H_inv, inv_h, Val(1), f_sign, face_row, Val(N))
        elseif α == 2
            _scalar_grad_sat!(@view(du[2, :, :, :]), u, u_face[f],
                                H_inv, inv_h, Val(2), f_sign, face_row, Val(N))
        else
            _scalar_grad_sat!(@view(du[3, :, :, :]), u, u_face[f],
                                H_inv, inv_h, Val(3), f_sign, face_row, Val(N))
        end
    end

    # Stage 3: ∂_α ∂_α u via SBP G along the same axis. Sum the three
    # diagonal contractions into `Lu` — the Laplacian needs only the
    # diagonal of the Hessian, so no cross terms required.
    @inbounds for k in 1:N, j in 1:N, i in 1:N
        s = zero(T)
        for p in 1:N
            s += G[i, p] * du[1, p, j, k]      # ∂_xx
        end
        for p in 1:N
            s += G[j, p] * du[2, i, p, k]      # ∂_yy
        end
        for p in 1:N
            s += G[k, p] * du[3, i, j, p]      # ∂_zz
        end
        Lu[i, j, k] = s * inv_h
    end

    return Lu
end

# ────────────────────────────────────────────────────────────────────────
# Conservative variant. Same wide-stencil `D_α D_α` core, but instead of
# a centred-flux SAT inside the first derivative we apply two SAT terms
# *outside* the second derivative, per face:
#
#   (a) Neumann-symmetrising:   −f_sign · (1/h) · (1/H[face_row]) · (D_α u)|_face
#       This cancels the spurious boundary term that arises from
#       `H · D_α D_α = −D_αᵀ H D_α + B_α D_α`, i.e. the `B_α D_α`
#       (boundary normal-derivative) piece. After cancellation,
#       `H · L = −∑_α D_αᵀ H D_α − ∑_f τ · (face mass) · E_f`, which is
#       symmetric.
#
#   (b) Dirichlet penalty:      −(τ/h) · (1/H[face_row]) · (u|_face − g)
#       Pure penalty for the BC. `τ` dimensionless; larger τ → tighter
#       BC enforcement at the cost of stiffness. For SBP-GLL we
#       default to `τ = (N − 1)²` which puts the penalty term on the
#       same scale as the SBP stiffness.

@inline function _wave_lap_inner!(Lu::AbstractArray{T, 3},
                                    u::AbstractArray{T, 3},
                                    du::AbstractArray{T, 4},
                                    G::AbstractMatrix{T}, inv_h::T,
                                    ::Val{N}) where {N, T}
    # Second SBP G sweep: sum of diagonal Hessian components into Lu.
    @inbounds for k in 1:N, j in 1:N, i in 1:N
        s = zero(T)
        for p in 1:N
            s += G[i, p] * du[1, p, j, k]
        end
        for p in 1:N
            s += G[j, p] * du[2, i, p, k]
        end
        for p in 1:N
            s += G[k, p] * du[3, i, j, p]
        end
        Lu[i, j, k] = s * inv_h
    end
    return Lu
end

"""
    wave_lap_strong_conservative_element!(Lu, u, u_face, ops, h; τ = (N-1)^2)

H_phys-symmetric, energy-conserving strong-form scalar Laplacian on a
single Cartesian SBP element of edge length `h`. Mattsson–Nordström
SAT pair (Neumann-symmetrising + Dirichlet penalty) is applied to the
wide-stencil `D_α D_α` core. `τ` is a dimensionless penalty parameter;
the physical penalty entering `H_phys · L` scales as `τ / h`.

Same array conventions as `wave_lap_strong_element!`.
"""
function wave_lap_strong_conservative_element!(Lu::AbstractArray{T, 3},
                                                 u::AbstractArray{T, 3},
                                                 u_face::NTuple{6, <:AbstractArray{T, 2}},
                                                 ops, h::T;
                                                 τ::T = T((size(u, 1) - 1)^2)) where {T}
    @assert size(u) == size(Lu)
    N = size(u, 1)
    @assert size(u, 1) == size(u, 2) == size(u, 3) == N

    inv_h = one(T) / h
    H_inv = SVector{N, T}(ntuple(i -> one(T) / ops.H[i, i], Val(N)))
    G     = SMatrix{N, N, T}(ops.G)

    # Stage 1: ∂_α u via SBP G along each axis (no SAT — the BC is
    # imposed weakly on Lu directly, not on the gradient).
    du = Array{T, 4}(undef, 3, N, N, N)
    for α in 1:3
        _scalar_grad!(@view(du[α, :, :, :]), u, G, inv_h, Val(α), Val(N))
    end

    # Stage 2: wide-stencil `Lu = ∑_α D_α D_α u`.
    _wave_lap_inner!(Lu, u, du, G, inv_h, Val(N))

    # Stage 3: per-face SAT pair.
    τ_over_h = τ * inv_h
    @inbounds for f in 1:6
        α        = (f + 1) >> 1
        f_sign   = T(isodd(f) ? -1 : +1)
        face_row = isodd(f) ? 1 : N
        coef_neu = -f_sign * inv_h * H_inv[face_row]
        coef_dir = -τ_over_h *        H_inv[face_row]
        uf       = u_face[f]
        for q in 1:N, p in 1:N
            i, j, k = (α == 1 ? (face_row, p, q) :
                       α == 2 ? (p, face_row, q) :
                                (p, q, face_row))
            du_α   = (α == 1 ? du[1, i, j, k] :
                      α == 2 ? du[2, i, j, k] :
                               du[3, i, j, k])
            Lu[i, j, k] += coef_neu * du_α
            Lu[i, j, k] += coef_dir * (u[i, j, k] - uf[p, q])
        end
    end

    return Lu
end

# ────────────────────────────────────────────────────────────────────────
# Multi-element driver. Walks the mesh, extracts each element's six
# face traces from its neighbours via `mesh.conn`, and calls the
# per-element kernel. For inter-element faces the neighbour trace
# substitutes for the Dirichlet data `g_face` — the SAT then becomes a
# jump penalty `τ · (u_self − u_neighbour)`, which is exactly the
# energy-conservative inter-element glue we want.
#
# Outer-boundary faces (`neighbour == 0`) use `outer_face = 0` by
# default, i.e. homogeneous Dirichlet. Caller may supply a per-element
# / per-face `outer_bdry` array later if needed.
#
# Restriction (for now): uniform-cube mesh with axis-aligned elements
# of equal edge length `h` and orientation 0 everywhere. The driver
# asserts this so we don't silently produce garbage on multi-patch /
# curvilinear meshes.

@inline function _extract_face!(face::AbstractMatrix{T},
                                  u_nbr::AbstractArray{T, 3},
                                  f_nbr::Integer, N::Integer) where {T}
    if f_nbr == 1
        @inbounds for q in 1:N, p in 1:N; face[p, q] = u_nbr[1, p, q]; end
    elseif f_nbr == 2
        @inbounds for q in 1:N, p in 1:N; face[p, q] = u_nbr[N, p, q]; end
    elseif f_nbr == 3
        @inbounds for q in 1:N, p in 1:N; face[p, q] = u_nbr[p, 1, q]; end
    elseif f_nbr == 4
        @inbounds for q in 1:N, p in 1:N; face[p, q] = u_nbr[p, N, q]; end
    elseif f_nbr == 5
        @inbounds for q in 1:N, p in 1:N; face[p, q] = u_nbr[p, q, 1]; end
    else
        @inbounds for q in 1:N, p in 1:N; face[p, q] = u_nbr[p, q, N]; end
    end
    return face
end

"""
    wave_lap_strong_conservative_mesh!(Lu, u, mesh, ops, h; τ = (N-1)^2)

Apply the H_phys-symmetric strong-form scalar Laplacian element by
element to the full mesh state. `u` and `Lu` are `(N, N, N, Ne)`
arrays. Face data at every element face is read from `mesh.conn`:
interior faces (including periodic seams) get the neighbour's face
trace (energy-conservative inter-element glue), outer-boundary faces
get homogeneous Dirichlet.

Currently restricted to axis-aligned uniform-cube meshes (single
patch, `orientation ≡ 0`, equal edge length `h` on every element).
The uniform-hex builders from HexMeshes satisfy this.
"""
function wave_lap_strong_conservative_mesh!(Lu::AbstractArray{T, 4},
                                              u::AbstractArray{T, 4},
                                              mesh, ops, h::T;
                                              τ::T = T((size(u, 1) - 1)^2)) where {T}
    N  = size(u, 1)
    Ne = size(u, 4)
    @assert size(u) == size(Lu) == (N, N, N, Ne)
    @assert all(mesh.conn.orientation .== 0)  # uniform-cube path

    u_face    = ntuple(_ -> Matrix{T}(undef, N, N), Val(6))
    zero_face = zeros(T, N, N)

    for e in 1:Ne
        for f in 1:6
            e_nbr = Int(mesh.conn.neighbour[f, e])
            if e_nbr == 0
                # Outer boundary: homogeneous Dirichlet for now.
                copyto!(u_face[f], zero_face)
            else
                f_nbr = Int(mesh.conn.neighbour_face[f, e])
                _extract_face!(u_face[f], view(u, :, :, :, e_nbr), f_nbr, N)
            end
        end
        wave_lap_strong_conservative_element!(view(Lu, :, :, :, e),
                                                view(u,  :, :, :, e),
                                                u_face, ops, h; τ = τ)
    end
    return Lu
end
