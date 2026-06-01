# Curvilinear strong-form scalar wave RHS on a `MeshGeometry`.
#
#   ü = ∇²u  +  per-face SAT  +  per-face Sommerfeld at outer boundary
#
# Replaces the uniform-cube-only `wave_lap_strong_conservative_mesh!` with a
# kernel that consumes per-node `J⁻¹` from `HexSBPSAT.MeshGeometry`,
# supports inter-patch `orientation ≠ 0`, and applies a Bayliss–Turkel
# Sommerfeld SAT at outer faces tagged `7` (see `wave.jl:556–622` for
# the formula). The state is `(u, u̇)`; the output is `ü`.

# Reference D̂_α u via SBP G on a scalar field, returning the *reference*
# derivative (no `1/h` factor — the physical gradient is built later
# via `J⁻¹`). Same body as `_scalar_grad!` from `wave_lap_strong.jl`
# but with `inv_h = 1`; reused by passing `1.0` for `inv_h`.

@inline function _ref_grad!(du::AbstractArray{T, 3}, u::AbstractArray{T, 3},
                             G::AbstractMatrix{T},
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
        du[i, j, k] = s
    end
    return du
end

# Orientation-aware face extraction. For `ori == 0` falls back to the
# direct-copy form used by `_extract_face!` in `wave_lap_strong.jl`.
# For `ori ≠ 0`, the neighbour's face `(p, q)` is a D₄ permutation of
# self's, computed by `_neigh_pq` from HexMeshes (exposed via
# HexSBPSAT). The neighbour value to read is `u_nbr[face f_nbr]` at
# the permuted indices.
@inline function _extract_face_oriented!(face::AbstractMatrix{T},
                                           u_nbr::AbstractArray{T, 3},
                                           f_nbr::Integer,
                                           ori::Integer,
                                           N::Integer) where {T}
    if ori == 0
        # Fast path — direct copy.
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
    else
        @inbounds for q in 1:N, p in 1:N
            pn, qn = HexSBPSAT._neigh_pq(Int8(ori), Int32(p), Int32(q), Int32(N))
            face[p, q] = (f_nbr == 1 ? u_nbr[1, pn, qn] :
                          f_nbr == 2 ? u_nbr[N, pn, qn] :
                          f_nbr == 3 ? u_nbr[pn, 1, qn] :
                          f_nbr == 4 ? u_nbr[pn, N, qn] :
                          f_nbr == 5 ? u_nbr[pn, qn, 1] :
                                       u_nbr[pn, qn, N])
        end
    end
    return face
end

"""
    wave_strong_rhs_element!(ü, u, u̇, u_face, u̇_face,
                              invjac, detjac, Hphys, handedness,
                              ops, bdry; τ, sommerfeld_R)

Per-element kernel. Curvilinear `∇²u` via the wide-stencil chain rule
(reference SBP `G` twice with per-node `J⁻¹` between), plus the
Mattsson–Nordström SAT pair at interior faces (`bdry[f] == 0`) and
the Bayliss–Turkel Sommerfeld SAT at outer faces (`bdry[f] == 7`).

* `ü, u, u̇ :: AbstractArray{T,3}` of shape `(N, N, N)`.
* `u_face, u̇_face :: NTuple{6, Matrix{T}}` — neighbour face traces
  (already orientation-resolved by the caller).
* `invjac :: AbstractArray{T,5}` of shape `(3, 3, N, N, N)` — `J⁻¹`
  per node.
* `detjac, Hphys :: AbstractArray{T,3}` of shape `(N, N, N)`.
* `handedness :: Int8` — sign of `det J`.
* `bdry :: NTuple{6, Int8}` — `mesh.conn.bdry[:, e]`. Tag `0` =
  interior face; tag `7` = Sommerfeld outer.
"""
function wave_strong_rhs_element!(ü::AbstractArray{T, 3},
                                    u::AbstractArray{T, 3},
                                    u̇::AbstractArray{T, 3},
                                    u_face ::NTuple{6, <:AbstractArray{T, 2}},
                                    u̇_face::NTuple{6, <:AbstractArray{T, 2}},
                                    invjac::AbstractArray{T, 5},
                                    dinvjac::AbstractArray{T, 6},
                                    detjac::AbstractArray{T, 3},
                                    Hphys ::AbstractArray{T, 3},
                                    handedness::Int8,
                                    ops, bdry::NTuple{6, Int8};
                                    τ::T = T((size(u, 1) - 1)^2),
                                    sommerfeld_R::T = T(Inf)) where {T}
    @assert size(u) == size(u̇) == size(ü)
    N = size(u, 1)
    @assert size(invjac)  == (3, 3, N, N, N)
    @assert size(dinvjac) == (3, 3, 3, N, N, N)
    @assert size(detjac) == (N, N, N)
    @assert size(Hphys)  == (N, N, N)

    H_ref     = SVector{N, T}(ntuple(i -> T(ops.H[i, i]), Val(N)))
    G         = SMatrix{N, N, T}(ops.G)
    inv_R     = isfinite(sommerfeld_R) ? one(T) / sommerfeld_R : zero(T)

    # Stage 1: reference D̂_c u for c = 1, 2, 3 (no `1/h` — curvilinear).
    du_ref = Array{T, 4}(undef, 3, N, N, N)
    for c in 1:3
        _ref_grad!(@view(du_ref[c, :, :, :]), u, G, Val(c), Val(N))
    end

    # Stage 2: reference Hessian D̂_d D̂_c u for the 6 unique (c, d) pairs
    # with c ≤ d. Tensor-product SBP `D̂` operators commute when c ≠ d,
    # so the off-diagonal pair is well-defined.
    ddu_ref = Array{T, 5}(undef, 3, 3, N, N, N)   # only the c ≤ d entries populated
    _ref_grad!(@view(ddu_ref[1, 1, :, :, :]), @view(du_ref[1, :, :, :]), G, Val(1), Val(N))   # ∂̃_x ∂̃_x
    _ref_grad!(@view(ddu_ref[1, 2, :, :, :]), @view(du_ref[1, :, :, :]), G, Val(2), Val(N))   # ∂̃_y ∂̃_x
    _ref_grad!(@view(ddu_ref[1, 3, :, :, :]), @view(du_ref[1, :, :, :]), G, Val(3), Val(N))   # ∂̃_z ∂̃_x
    _ref_grad!(@view(ddu_ref[2, 2, :, :, :]), @view(du_ref[2, :, :, :]), G, Val(2), Val(N))   # ∂̃_y ∂̃_y
    _ref_grad!(@view(ddu_ref[2, 3, :, :, :]), @view(du_ref[2, :, :, :]), G, Val(3), Val(N))   # ∂̃_z ∂̃_y
    _ref_grad!(@view(ddu_ref[3, 3, :, :, :]), @view(du_ref[3, :, :, :]), G, Val(3), Val(N))   # ∂̃_z ∂̃_z

    # Stage 3: physical gradient ∂_x_a u = ∑_α (J⁻¹)[α, a] · D̂_α u (needed by SAT).
    # MeshGeometry convention: invjac[α, β] = ∂ξ_α/∂x_β — first index is the
    # *reference* axis, second is the *physical* axis. So the reference
    # axis is the *inner* loop, summed.
    du_phys = Array{T, 4}(undef, 3, N, N, N)
    @inbounds for k in 1:N, j in 1:N, i in 1:N
        for a in 1:3              # a is the physical axis of ∇u_phys
            s = zero(T)
            for α in 1:3          # α is the reference axis (sum index)
                s += invjac[α, a, i, j, k] * du_ref[α, i, j, k]
            end
            du_phys[a, i, j, k] = s
        end
    end

    # Stage 4: pointwise ∇²u via the full strong-form chain rule:
    #   ∇²u = ∑_{αβ} G^{αβ} · D̂_α D̂_β u  +  ∑_β W^β · D̂_β u
    # with
    #   G^{αβ}(x) = ∑_i (J⁻¹)[α, i] · (J⁻¹)[β, i]                 (symmetric)
    #   W^β(x)    = ∑_α ∑_i (J⁻¹)[α, i] · D̂_α(J⁻¹)[β, i]
    # Both α, β iterate over *reference* axes; i over *physical*.
    @inbounds for k in 1:N, j in 1:N, i in 1:N
        s = zero(T)
        # Principal-metric term, sum over symmetric pairs α ≤ β with
        # the off-diagonal doubling factor.
        for α in 1:3
            Gαα = zero(T)
            for ii in 1:3
                Gαα += invjac[α, ii, i, j, k] * invjac[α, ii, i, j, k]
            end
            s += Gαα * ddu_ref[α, α, i, j, k]
            for β in (α + 1):3
                Gαβ = zero(T)
                for ii in 1:3
                    Gαβ += invjac[α, ii, i, j, k] * invjac[β, ii, i, j, k]
                end
                # ddu_ref stores only c ≤ d entries: (α, β) with α < β.
                s += T(2) * Gαβ * ddu_ref[α, β, i, j, k]
            end
        end
        # Metric-divergence term.
        for β in 1:3
            Wβ = zero(T)
            for α in 1:3, ii in 1:3
                # D̂_α (invjac[β, ii]) = dinvjac[β, ii, α].
                Wβ += invjac[α, ii, i, j, k] * dinvjac[β, ii, α, i, j, k]
            end
            s += Wβ * du_ref[β, i, j, k]
        end
        ü[i, j, k] = s
    end

    # Stage 5: per-face SAT.
    #   tag == 0    → interior seam, Mattsson–Nordström vs neighbour trace
    #   tag 1..6    → Dirichlet outer face, Mattsson–Nordström vs u_face[f]
    #                  (homogeneous when u_face[f] is pre-filled with zeros)
    #   tag == 7    → Sommerfeld outer face, Bayliss–Turkel BGT-0 / BGT-1
    @inbounds for f in 1:6
        α        = (f + 1) >> 1
        f_sign   = T(isodd(f) ? -1 : +1)
        face_row = isodd(f) ? 1 : N
        tag      = bdry[f]
        is_sommerfeld = (tag == Int8(7))
        is_dirichlet  = (Int8(1) ≤ tag ≤ Int8(6))
        # Interior and Dirichlet share the same SAT branch — the only
        # difference is what `u_face[f]` carries (neighbour trace vs
        # boundary data).
        is_interior_or_dirichlet = (tag == Int8(0)) || is_dirichlet

        for q in 1:N, p in 1:N
            i, j, k = (α == 1 ? (face_row, p, q) :
                       α == 2 ? (p, face_row, q) :
                                (p, q, face_row))

            # Physical outward normal along axis α: row α of J⁻¹ —
            # this is `∇ξ_α` in physical space. At the +α face
            # (ξ_α = max, `f_sign = +1`) the outward direction is
            # `+∇ξ_α`; at the −α face it's `−∇ξ_α`. The mapping is
            # `f_sign · invjac[α, :]` for *both* right- and
            # left-handed elements — `invjac` already encodes the
            # parameterisation's orientation, so we should NOT
            # multiply by `handedness` here. (For left-handed
            # elements `invjac[α, :]` points in the opposite physical
            # direction, but the +α reference face is also on the
            # opposite physical side, so the product is outward.)
            n_ux = invjac[α, 1, i, j, k]
            n_uy = invjac[α, 2, i, j, k]
            n_uz = invjac[α, 3, i, j, k]
            n_norm = sqrt(n_ux*n_ux + n_uy*n_uy + n_uz*n_uz)
            sgn    = f_sign
            nx     = sgn * n_ux / n_norm
            ny     = sgn * n_uy / n_norm
            nz     = sgn * n_uz / n_norm

            # n · ∇u at this face node, using the physical gradient.
            ∂nu = nx * du_phys[1, i, j, k] +
                  ny * du_phys[2, i, j, k] +
                  nz * du_phys[3, i, j, k]

            Hp = Hphys[i, j, k]

            if is_interior_or_dirichlet
                # Mattsson–Nordström: Neumann symmetrising + Dirichlet
                # jump penalty. Same formula for both interior seams
                # and Dirichlet outer faces; only `u_face[f]` differs
                # (neighbour trace vs boundary data).
                # Face mass weight wF = ‖J⁻¹_α‖ · |det J| · H_ref[p] H_ref[q];
                # SAT coefficient is wF / Hp = n_norm / H_ref[face_row]
                # (the tangential H's cancel).
                wF_over_Hp = n_norm / H_ref[face_row]
                # Mirrors the uniform-cube case:
                #   coef_neu = -f_sign · (1/h) · H_inv[face_row]   →   -wF_over_Hp · f_sign
                # (the f_sign·handedness already appears in `n_norm`'s direction via `sgn`
                # in `∂nu`, so the sign here is absorbed there.)
                #   coef_dir = -(τ/h) · H_inv[face_row]            →   -τ · wF_over_Hp
                coef_neu = -wF_over_Hp
                coef_dir = -τ * wF_over_Hp
                ü[i, j, k] += coef_neu * ∂nu
                ü[i, j, k] += coef_dir * (u[i, j, k] - u_face[f][p, q])
            elseif is_sommerfeld
                # Bayliss–Turkel BGT-1 SAT (BGT-0 if inv_R = 0):
                #   ü -= wF · (u̇ + n·∇u + u/R) / H_phys
                wF = n_norm * detjac[i, j, k] * H_ref[p] * H_ref[q]
                ü[i, j, k] -= wF * (u̇[i, j, k] + ∂nu + inv_R * u[i, j, k]) / Hp
            end
        end
    end

    return ü
end

"""
    wave_strong_rhs_mesh!(ü, u, u̇, mesh, geom, ops;
                            τ = (N-1)², sommerfeld_R = Inf)

Multi-element driver. Walks every element, extracts each element's six
face traces from its neighbours via `mesh.conn` (with orientation
handling), slices the per-element `MeshGeometry` views, calls
`wave_strong_rhs_element!`.

Outer-boundary face tag semantics:
- `bdry == 0`  — interior seam, Mattsson–Nordström vs the neighbour
  trace (already extracted into `u_face[f]`).
- `bdry == 7`  — Sommerfeld outer face, Bayliss–Turkel BGT-0 / BGT-1.
- `bdry 1..6`  — Dirichlet outer face. The Mattsson–Nordström branch
  fires against the value in `u_face[f]` — the mesh driver pre-fills
  this with zeros at outer faces (`mesh.conn.neighbour == 0`), so the
  default is homogeneous Dirichlet. Inhomogeneous Dirichlet would
  need a per-face boundary-data argument (not implemented yet).
"""
function wave_strong_rhs_mesh!(ü::AbstractArray{T, 4},
                                  u::AbstractArray{T, 4},
                                  u̇::AbstractArray{T, 4},
                                  mesh,
                                  geom,
                                  dinvjac::AbstractArray{T, 7},
                                  ops;
                                  τ::T = T((size(u, 1) - 1)^2),
                                  sommerfeld_R::T = T(Inf)) where {T}
    N  = size(u, 1)
    Ne = size(u, 4)
    @assert size(u) == size(u̇) == size(ü) == (N, N, N, Ne)
    @assert size(dinvjac) == (3, 3, 3, N, N, N, Ne)

    u_face  = ntuple(_ -> Matrix{T}(undef, N, N), Val(6))
    u̇_face  = ntuple(_ -> Matrix{T}(undef, N, N), Val(6))
    zero_face = zeros(T, N, N)

    for e in 1:Ne
        for f in 1:6
            e_nbr = Int(mesh.conn.neighbour[f, e])
            if e_nbr == 0
                copyto!(u_face[f],  zero_face)
                copyto!(u̇_face[f], zero_face)
            else
                f_nbr = Int(mesh.conn.neighbour_face[f, e])
                ori   = Int(mesh.conn.orientation[f, e])
                _extract_face_oriented!(u_face[f],  view(u,  :, :, :, e_nbr), f_nbr, ori, N)
                _extract_face_oriented!(u̇_face[f], view(u̇, :, :, :, e_nbr), f_nbr, ori, N)
            end
        end
        bdry_e = ntuple(f -> Int8(mesh.conn.bdry[f, e]), Val(6))
        wave_strong_rhs_element!(view(ü, :, :, :, e),
                                   view(u,  :, :, :, e),
                                   view(u̇, :, :, :, e),
                                   u_face, u̇_face,
                                   view(geom.invjac,  :, :, :, :, :, e),
                                   view(dinvjac,      :, :, :, :, :, :, e),
                                   view(geom.detjac, :, :, :, e),
                                   view(geom.Hphys,  :, :, :, e),
                                   geom.handedness[e],
                                   ops, bdry_e;
                                   τ = τ, sommerfeld_R = sommerfeld_R)
    end
    return ü
end

"""
    wave_strong_rhs_mesh!(ü, u, u̇, mesh, geom, ops; …)

Convenience overload that computes `dinvjac` internally on every call.
Fine for one-off calls and small meshes; for performance-sensitive
loops, pre-compute via `make_metric_derivs(geom, ops)` and use the
`dinvjac`-explicit signature.
"""
function wave_strong_rhs_mesh!(ü::AbstractArray{T, 4},
                                  u::AbstractArray{T, 4},
                                  u̇::AbstractArray{T, 4},
                                  mesh, geom, ops;
                                  τ::T = T((size(u, 1) - 1)^2),
                                  sommerfeld_R::T = T(Inf)) where {T}
    dinvjac = make_metric_derivs(geom, ops)
    return wave_strong_rhs_mesh!(ü, u, u̇, mesh, geom, dinvjac, ops;
                                  τ = τ, sommerfeld_R = sommerfeld_R)
end

"""
    make_metric_derivs(geom, ops) -> Array{T, 7}

Pre-compute the metric-derivative array
`dinvjac[a, c, d, i, j, k, e] = D̂_d (J⁻¹)_a^c` at every GLL node, by
applying the reference SBP `G` operator (no `1/h` scaling) to each
`(a, c)` entry of `geom.invjac` along each reference axis `d`. The
result is consumed by the chain-rule branch in
`wave_strong_rhs_element!`.

For a fixed mesh this is computed once at setup and reused across
every time step. Memory cost is `(3·3·3 · N³ · Ne · sizeof(T))` —
about 100 KB on the canonical 2×2×2 / N=4 testbed.
"""
function make_metric_derivs(geom, ops)
    invjac = geom.invjac
    T  = eltype(invjac)
    Ne = geom.Ne
    N  = size(invjac, 3)
    G  = SMatrix{N, N, T}(ops.G)
    dinvjac = Array{T, 7}(undef, 3, 3, 3, N, N, N, Ne)
    for e in 1:Ne, a in 1:3, c in 1:3
        for d in 1:3
            _ref_grad!(@view(dinvjac[a, c, d, :, :, :, e]),
                        @view(invjac[a, c, :, :, :, e]),
                        G, Val(d), Val(N))
        end
    end
    return dinvjac
end
