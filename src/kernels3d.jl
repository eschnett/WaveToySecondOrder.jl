# 3D kernel: curvilinear-aware SBP-SAT Laplacian on a `MeshGeometry`.
#
# Element shape enters only through the per-node Jacobian carried by the
# `MeshGeometry`. The kernel applies the same machinery to axis-aligned
# cubes (where the Jacobian is a constant diagonal) and to curved hex
# elements (where the Jacobian varies per node and may be dense).
#
# Per-element structure
# ---------------------
# For each element `e`, the kernel computes `(L u)|_e` where `L` is the
# discrete physical Laplacian:
#
#   1. **Reference-axis gradients.** Compute `∂_a u[i,j,k]` for `a = 1,2,3`
#      via three 1D `G`-sweeps along each reference axis.
#
#   2. **Weak stiffness flux.** At each node, form
#         W_a[i,j,k] = H_ref[i] H_ref[j] H_ref[k] · |det J| ·
#                        Σ_b (J⁻¹ J⁻ᵀ)_{ab} · ∂_b u.
#      The contravariant metric `J⁻¹ J⁻ᵀ` couples reference axes
#      whenever `J` is non-diagonal (curvilinear elements).
#
#   3. **Weak volume term.** Apply `-Gᵀ` along each reference axis to
#      `W_a` and sum into `(H_phys · L u)`. This is the volume part of
#      the SBP-DG weak Laplacian; for axis-aligned cubes it reduces to
#      `(1/h²) · (−H⁻¹ Gᵀ H G) u` along each axis.
#
#   4. **Per-face SBP-SAT terms** (six faces per element). At every face
#      quadrature node, the kernel:
#         (a) builds the two tangent vectors as columns of `J` and takes
#             their cross product → physical outward normal `n_phys` and
#             surface element `|J_F|`;
#         (b) forms `μ = J⁻¹ · n_phys` (per-face-node 3-vector);
#         (c) evaluates the physical normal gradient
#                ∇u · n_phys = Σ_c μ_c · ∂_c u
#             on both the self and (when interior) neighbour sides;
#         (d) adds, with face quadrature weight `w_F = |J_F| · H_face`,
#                + w_F · (∇u · n_phys)_self            ← boundary lift
#                − (1/2) · w_F · Δ(∇u · n_phys)        ← adjoint consistency
#                − τ · w_F · Δu                        ← penalty
#             to the face node, and the symmetric SIPG lift
#                + α · w_F · Δu  ·  Σ_c μ_c · Gᵀ_c
#             distributed via three reference-axis `Gᵀ` lifts. For
#             axis-aligned cubes only `μ_a` (the orthogonal axis) is
#             non-zero, so this collapses to the existing single-axis
#             lift used in earlier code.
#
#   5. **Mass division.** Divide the accumulated `H_phys · L u` by the
#      per-node physical mass `H_phys = H_ref · |det J|` to recover
#      `L u` in physical units.

using Polyester: @batch
using StaticArrays: SVector

################################################################################
# Initialisation (unchanged from previous version)

function initialize3d!(u::AbstractArray{T,3}, u̇::AbstractArray{T,3},
                       x::AbstractArray{T,3}, y::AbstractArray{T,3}, z::AbstractArray{T,3},
                       t; A, kx, ky, kz, ω) where {T}
    @. u  =  A   * sin(kx*x) * sin(ky*y) * sin(kz*z) * cos(ω*t)
    @. u̇ = -A*ω * sin(kx*x) * sin(ky*y) * sin(kz*z) * sin(ω*t)
    return u, u̇
end

function initialize3d!(u::AbstractArray{T,4}, u̇::AbstractArray{T,4},
                       coords::AbstractArray{T,5}, t;
                       A, kx, ky, kz, ω) where {T}
    @assert size(u̇) == size(u)
    Ne = size(u, 4)
    @assert size(coords, 5) == Ne
    for e in 1:Ne
        initialize3d!(view(u,  :, :, :, e),
                      view(u̇, :, :, :, e),
                      view(coords, 1, :, :, :, e),
                      view(coords, 2, :, :, :, e),
                      view(coords, 3, :, :, :, e), t;
                      A, kx, ky, kz, ω)
    end
    return u, u̇
end

################################################################################
# Face axis bookkeeping
#
# Face index convention (matches `HexMesh`):
#   1 = −x    2 = +x    3 = −y    4 = +y    5 = −z    6 = +z
#
# For face `f`, the orthogonal reference axis is `a = (f+1) ÷ 2`. The two
# in-face tangent reference axes `(axis_p, axis_q)` are picked so that the
# face quadrature node index `(p, q)` maps naturally to the volume nodes:
#
#   a = 1 → (axis_p, axis_q) = (2, 3),  volume node = (face_row, p, q)
#   a = 2 → (axis_p, axis_q) = (1, 3),  volume node = (p, face_row, q)
#   a = 3 → (axis_p, axis_q) = (1, 2),  volume node = (p, q, face_row)
#
# `face_row = 1` on a −face, `N` on a +face. The natural cross product
# `t_axis_p × t_axis_q` equals `+ê_a` for `a ∈ {1,3}` and `−ê_a` for
# `a = 2`; the kernel multiplies by `cross_sign · face_sign` so the
# resulting vector points outward from the element.

@inline _face_axis_idx(::Val{f}) where {f} = (f + 1) ÷ 2
@inline _face_row(::Val{f}, ::Val{N}) where {f, N} = isodd(f) ? 1 : N
@inline _face_sign(::Val{f}, ::Type{T}) where {f, T} = isodd(f) ? -one(T) : one(T)
@inline _cross_sign(::Val{a}, ::Type{T}) where {a, T} = a == 2 ? -one(T) : one(T)
@inline _tangent_axes(::Val{1}) = (2, 3)
@inline _tangent_axes(::Val{2}) = (1, 3)
@inline _tangent_axes(::Val{3}) = (1, 2)

# Volume index of face node `(p, q)` for face axis `a` and `face_row`.
@inline _face_volume_idx(::Val{1}, face_row, p, q) = (face_row, p, q)
@inline _face_volume_idx(::Val{2}, face_row, p, q) = (p, face_row, q)
@inline _face_volume_idx(::Val{3}, face_row, p, q) = (p, q, face_row)

# Runtime variant used for the neighbour lookup (the neighbour's face
# axis is known only at runtime from `mesh.neighbour_face`).
@inline function _face_volume_idx(a::Integer, face_row, p, q)
    if a == 1
        return (face_row, p, q)
    elseif a == 2
        return (p, face_row, q)
    else
        return (p, q, face_row)
    end
end

# `_neigh_pq` is defined in `mesh.jl` (alongside `_compute_face_orientation`).

################################################################################
# Per-face SAT contribution

# Adds, for one face `f` of element `e`, the four SBP-SAT terms
# (boundary lift, adjoint consistency, penalty, symmetric SIPG lift)
# to `üe = H_phys · L u`. `Du` holds this element's reference gradients
# already populated by the volume pass. Neighbour reference gradients
# at the shared face are recomputed on the fly.
@inline function _add_face_sat!(::Val{f}, üe::AbstractArray{T,3},
                                 ue::AbstractArray{T,3},
                                 u_global::AbstractArray{T,4},
                                 Du::AbstractArray{T,4},
                                 e::Int,
                                 geom::MeshGeometry{T, N}, mesh::HexMesh{T},
                                 bdry_values::NTuple{6, T},
                                 ops::SBPOps{N, T}, τ,
                                 H_1d::SVector{N, T}) where {f, N, T}
    a_idx   = _face_axis_idx(Val(f))
    face_r  = _face_row(Val(f), Val(N))
    sgn_f   = _face_sign(Val(f), T)
    axis_p, axis_q = _tangent_axes(Val(a_idx))

    G   = ops.G
    nbr = mesh.neighbour[f, e]
    tag = mesh.bdry[f, e]
    α   = nbr == 0 ? one(T) : one(T) / 2

    # Neighbour-side face index, axis, row, and (p, q) orientation.
    # Pulled at runtime — different shared faces can pair self's face with
    # any of the neighbour's six faces and with any of the eight D₄
    # orientations.
    nbr_face = Int(mesh.neighbour_face[f, e])
    nbr_a    = (nbr_face + 1) ÷ 2
    nbr_row  = isodd(nbr_face) ? 1 : N
    nbr_o    = mesh.orientation[f, e]

    @inbounds for q in 1:N, p in 1:N
        i, j, k = _face_volume_idx(Val(a_idx), face_r, p, q)

        # Self-side J at this face node.
        J11 = geom.jac[1,1,i,j,k,e]; J12 = geom.jac[1,2,i,j,k,e]; J13 = geom.jac[1,3,i,j,k,e]
        J21 = geom.jac[2,1,i,j,k,e]; J22 = geom.jac[2,2,i,j,k,e]; J23 = geom.jac[2,3,i,j,k,e]
        J31 = geom.jac[3,1,i,j,k,e]; J32 = geom.jac[3,2,i,j,k,e]; J33 = geom.jac[3,3,i,j,k,e]

        # Tangent vectors (columns axis_p and axis_q of J).
        tp_x = axis_p == 1 ? J11 : axis_p == 2 ? J12 : J13
        tp_y = axis_p == 1 ? J21 : axis_p == 2 ? J22 : J23
        tp_z = axis_p == 1 ? J31 : axis_p == 2 ? J32 : J33
        tq_x = axis_q == 1 ? J11 : axis_q == 2 ? J12 : J13
        tq_y = axis_q == 1 ? J21 : axis_q == 2 ? J22 : J23
        tq_z = axis_q == 1 ? J31 : axis_q == 2 ? J32 : J33

        # Natural cross product of the tangent vectors (in physical space).
        cx = tp_y * tq_z - tp_z * tq_y
        cy = tp_z * tq_x - tp_x * tq_z
        cz = tp_x * tq_y - tp_y * tq_x

        # Determine the outward sign from the actual J column along axis a:
        # the outward normal at face row 1 (resp. N) is anti-parallel (resp.
        # parallel) to `col_a = ∂x/∂ξ_a` independent of element handedness.
        col_a_x = a_idx == 1 ? J11 : a_idx == 2 ? J12 : J13
        col_a_y = a_idx == 1 ? J21 : a_idx == 2 ? J22 : J23
        col_a_z = a_idx == 1 ? J31 : a_idx == 2 ? J32 : J33
        dot_ca  = cx * col_a_x + cy * col_a_y + cz * col_a_z   # = ε_{p,q,a} · det J
        sgn_out = sgn_f * (dot_ca ≥ 0 ? one(T) : -one(T))

        nx_u = sgn_out * cx
        ny_u = sgn_out * cy
        nz_u = sgn_out * cz

        JF = sqrt(nx_u*nx_u + ny_u*ny_u + nz_u*nz_u)
        nx = nx_u / JF; ny = ny_u / JF; nz = nz_u / JF
        # SIPG penalty needs an extra `1/h_F` (the local face mesh size).
        # `sqrt(|J_F|)` reduces to the element edge length on axis-aligned
        # cubes and is the natural per-face-node size on curvilinear elements.
        hF = sqrt(JF)

        # Self-side J⁻¹ and μ = J⁻¹ · n_phys.
        Ji11 = geom.invjac[1,1,i,j,k,e]; Ji12 = geom.invjac[1,2,i,j,k,e]; Ji13 = geom.invjac[1,3,i,j,k,e]
        Ji21 = geom.invjac[2,1,i,j,k,e]; Ji22 = geom.invjac[2,2,i,j,k,e]; Ji23 = geom.invjac[2,3,i,j,k,e]
        Ji31 = geom.invjac[3,1,i,j,k,e]; Ji32 = geom.invjac[3,2,i,j,k,e]; Ji33 = geom.invjac[3,3,i,j,k,e]
        μ1 = Ji11*nx + Ji12*ny + Ji13*nz
        μ2 = Ji21*nx + Ji22*ny + Ji23*nz
        μ3 = Ji31*nx + Ji32*ny + Ji33*nz

        wF = JF * H_1d[p] * H_1d[q]

        # Self-side u and physical normal gradient.
        u_self  = ue[i, j, k]
        d1s = Du[1, i, j, k]; d2s = Du[2, i, j, k]; d3s = Du[3, i, j, k]
        Gn_self = μ1*d1s + μ2*d2s + μ3*d3s

        # Neighbour-side or boundary.
        u_neigh::T  = zero(T)
        Gn_neigh::T = zero(T)
        if nbr == 0
            u_neigh  = bdry_values[tag]
            Gn_neigh = Gn_self                  # mirror gradient ⇒ ΔGn = 0
        else
            # Map self's face-local (p, q) → neighbour's, then index into
            # the neighbour's volume using its own face axis & row.
            pn, qn = _neigh_pq(nbr_o, p, q, N)
            in_, jn_, kn_ = _face_volume_idx(nbr_a, nbr_row, pn, qn)
            u_neigh = u_global[in_, jn_, kn_, nbr]

            d1n = zero(T); d2n = zero(T); d3n = zero(T)
            for l in 1:N
                d1n += G[in_, l] * u_global[l,   jn_, kn_, nbr]
                d2n += G[jn_, l] * u_global[in_, l,   kn_, nbr]
                d3n += G[kn_, l] * u_global[in_, jn_, l,   nbr]
            end

            Jin11 = geom.invjac[1,1,in_,jn_,kn_,nbr]; Jin12 = geom.invjac[1,2,in_,jn_,kn_,nbr]; Jin13 = geom.invjac[1,3,in_,jn_,kn_,nbr]
            Jin21 = geom.invjac[2,1,in_,jn_,kn_,nbr]; Jin22 = geom.invjac[2,2,in_,jn_,kn_,nbr]; Jin23 = geom.invjac[2,3,in_,jn_,kn_,nbr]
            Jin31 = geom.invjac[3,1,in_,jn_,kn_,nbr]; Jin32 = geom.invjac[3,2,in_,jn_,kn_,nbr]; Jin33 = geom.invjac[3,3,in_,jn_,kn_,nbr]
            μn1 = Jin11*nx + Jin12*ny + Jin13*nz
            μn2 = Jin21*nx + Jin22*ny + Jin23*nz
            μn3 = Jin31*nx + Jin32*ny + Jin33*nz
            Gn_neigh = μn1*d1n + μn2*d2n + μn3*d3n
        end

        Δu  = u_self  - u_neigh
        ΔGn = Gn_self - Gn_neigh

        # Boundary lift (restores strong form at face node).
        üe[i, j, k] += wF * Gn_self

        # Adjoint consistency at face node.
        üe[i, j, k] -= (T(1)/2) * wF * ΔGn

        # Penalty at face node (σ/h_F · w_F · Δu).
        üe[i, j, k] -= τ * wF * Δu / hF

        # Symmetric SIPG lift via Gᵀ_c · μ_c (three reference-axis lifts).
        ν = α * wF * Δu
        for l in 1:N
            üe[l, j, k] += G[i, l] * μ1 * ν
        end
        for l in 1:N
            üe[i, l, k] += G[j, l] * μ2 * ν
        end
        for l in 1:N
            üe[i, j, l] += G[k, l] * μ3 * ν
        end
    end
    return üe
end

################################################################################
# Global RHS

"""
    rhs3d!(ü, u, u̇, bdry_values; geom, ops, τ)

Curvilinear-aware 3D RHS over the mesh carried by `geom :: MeshGeometry`.
State arrays are 4-D, shape `(N, N, N, Ne)`, with element ordering
matching `geom.mesh`. The 6-tuple `bdry_values` is indexed by the
boundary-condition tag stored on each outer face (`mesh.bdry[f, e]`).

The kernel assumes a **diagonal** 1D mass matrix (`ops.H` is the GLL
quadrature). For dense-`H` branches (e.g. the Rational/Vandermonde
operator construction) the weak-form assembly would need a different
implementation.

# Choosing τ (SIPG penalty constant)

`τ` must be large enough to make the discrete Laplacian negative
semi-definite. Empirical rules of thumb (N = 5 GLL nodes per element):

* **Axis-aligned cubical mesh**: `τ ≈ 1.5·(N−1)² = 24` is sufficient.
* **Curvilinear / multi-patch mesh** (e.g. `make_inflated_cube_mesh`):
  the SIPG threshold rises significantly because the outer-patch
  elements are anisotropic and skewed. Use `τ ≈ 8·(N−1)² = 128` (for
  N=5). The threshold at which the operator first becomes NSD is around
  `4·(N−1)² ≈ 64`, but staying close to that edge can leave a handful
  of slowly-growing modes which compound over long integrations; the
  extra margin keeps the spectrum cleanly negative. Increasing `τ`
  further is harmless apart from a tighter CFL — use
  [`recommended_dt`](@ref) to size `dt` automatically.

If the operator's spectrum has positive eigenvalues, evolution will
slowly grow even with a symplectic integrator; you can verify
coercivity for any (geom, ops, τ) tuple via [`discrete_laplacian`](@ref)
on a small representative mesh.
"""
function rhs3d!(ü::AbstractArray{T,4}, u::AbstractArray{T,4}, u̇::AbstractArray{T,4},
                bdry_values::NTuple{6, T};
                geom::MeshGeometry{T, N}, ops::SBPOps{N, T}, τ) where {N, T}
    @assert size(ü) == size(u̇) == size(u)
    @assert size(u, 1) == size(u, 2) == size(u, 3) == N
    @assert size(u, 4) == geom.mesh.Ne

    mesh = geom.mesh
    G    = ops.G
    H_1d = SVector{N, T}(ntuple(i -> ops.H[i, i], Val(N)))

    # Per-thread scratch buffers. The per-element loop is embarrassingly
    # parallel (each iteration only writes to `view(ü, :, :, :, e)` and
    # reads global `u`), so we split it with `Polyester.@batch`. Polyester
    # uses a persistent worker pool (no per-call task allocation, no
    # scheduler overhead), giving near-zero parallel-launch cost relative
    # to `Threads.@threads`. Per-iteration `threadid()` is stable because
    # `@batch` does not migrate work between threads. Buffers are sized to
    # `maxthreadid()` since Julia's thread ids span the default and
    # interactive thread pools.
    Du_buf = [Array{T, 4}(undef, 3, N, N, N) for _ in 1:Threads.maxthreadid()]
    W_buf  = [Array{T, 4}(undef, 3, N, N, N) for _ in 1:Threads.maxthreadid()]

    @inbounds @batch for e in 1:mesh.Ne
        tid = Threads.threadid()
        Du  = Du_buf[tid]
        W   = W_buf[tid]

        ue = view(u,  :, :, :, e)
        üe = view(ü, :, :, :, e)

        # 1. Reference-axis gradients.
        for k in 1:N, j in 1:N, i in 1:N
            s1 = zero(T); s2 = zero(T); s3 = zero(T)
            for l in 1:N
                s1 += G[i, l] * ue[l, j, k]
                s2 += G[j, l] * ue[i, l, k]
                s3 += G[k, l] * ue[i, j, l]
            end
            Du[1, i, j, k] = s1
            Du[2, i, j, k] = s2
            Du[3, i, j, k] = s3
        end

        # 2. Weak stiffness flux.
        for k in 1:N, j in 1:N, i in 1:N
            Ji11 = geom.invjac[1,1,i,j,k,e]; Ji12 = geom.invjac[1,2,i,j,k,e]; Ji13 = geom.invjac[1,3,i,j,k,e]
            Ji21 = geom.invjac[2,1,i,j,k,e]; Ji22 = geom.invjac[2,2,i,j,k,e]; Ji23 = geom.invjac[2,3,i,j,k,e]
            Ji31 = geom.invjac[3,1,i,j,k,e]; Ji32 = geom.invjac[3,2,i,j,k,e]; Ji33 = geom.invjac[3,3,i,j,k,e]
            dJ = geom.detjac[i, j, k, e]
            wt = H_1d[i] * H_1d[j] * H_1d[k] * dJ

            g11 = Ji11*Ji11 + Ji12*Ji12 + Ji13*Ji13
            g12 = Ji11*Ji21 + Ji12*Ji22 + Ji13*Ji23
            g13 = Ji11*Ji31 + Ji12*Ji32 + Ji13*Ji33
            g22 = Ji21*Ji21 + Ji22*Ji22 + Ji23*Ji23
            g23 = Ji21*Ji31 + Ji22*Ji32 + Ji23*Ji33
            g33 = Ji31*Ji31 + Ji32*Ji32 + Ji33*Ji33

            d1 = Du[1, i, j, k]; d2 = Du[2, i, j, k]; d3 = Du[3, i, j, k]
            W[1, i, j, k] = wt * (g11*d1 + g12*d2 + g13*d3)
            W[2, i, j, k] = wt * (g12*d1 + g22*d2 + g23*d3)
            W[3, i, j, k] = wt * (g13*d1 + g23*d2 + g33*d3)
        end

        # 3. Volume divergence: H_phys · L u = -Σ_a Gᵀ_a W_a.
        for k in 1:N, j in 1:N, i in 1:N
            s = zero(T)
            for l in 1:N
                s += G[l, i] * W[1, l, j, k]
                s += G[l, j] * W[2, i, l, k]
                s += G[l, k] * W[3, i, j, l]
            end
            üe[i, j, k] = -s
        end

        # 4. Per-face SBP-SAT contributions.
        _add_face_sat!(Val(1), üe, ue, u, Du, e, geom, mesh, bdry_values, ops, τ, H_1d)
        _add_face_sat!(Val(2), üe, ue, u, Du, e, geom, mesh, bdry_values, ops, τ, H_1d)
        _add_face_sat!(Val(3), üe, ue, u, Du, e, geom, mesh, bdry_values, ops, τ, H_1d)
        _add_face_sat!(Val(4), üe, ue, u, Du, e, geom, mesh, bdry_values, ops, τ, H_1d)
        _add_face_sat!(Val(5), üe, ue, u, Du, e, geom, mesh, bdry_values, ops, τ, H_1d)
        _add_face_sat!(Val(6), üe, ue, u, Du, e, geom, mesh, bdry_values, ops, τ, H_1d)

        # 5. Divide by H_phys per node.
        for k in 1:N, j in 1:N, i in 1:N
            Hp = H_1d[i] * H_1d[j] * H_1d[k] * geom.detjac[i, j, k, e]
            üe[i, j, k] /= Hp
        end
    end

    return ü
end

################################################################################
# Diagnostics

using SparseArrays: SparseMatrixCSC, sparse

"""
    discrete_laplacian(geom, ops, τ;
                       bdry_values = ntuple(_->0, 6),
                       drop_tol    = 0) → SparseMatrixCSC

Assemble the curvilinear discrete Laplacian `L_h` as an explicit sparse
matrix by repeatedly calling `rhs3d!` on each canonical basis vector.
The result `L_h` is the matrix that maps the linearly-indexed global
state vector `vec(u)` to `vec(rhs3d!(u))` (with `u̇ = 0` and zero
boundary data, so only the linear part of the homogeneous Laplacian
is captured).

Intended for analysis — eigenvalues, symmetry, condition number — not
for production timestepping. Cost is `O(N⁶ · Ne²)` element operations,
i.e. one `rhs3d!` call per degree of freedom.

The per-node physical mass `H_phys` (which makes `H_phys · L_h` the
symmetric "stiffness" matrix) is available as `vec(geom.Hphys_diag())`
or recoverable directly from `ops.H` and `geom.detjac`.

# Keyword arguments

* `bdry_values` — passed straight through to `rhs3d!`; defaults to zero
  Dirichlet on every outer-face tag.
* `drop_tol` — entries with `abs(v) ≤ drop_tol` are not stored. Default
  `0` keeps every numerically computed value.
"""
function discrete_laplacian(geom::MeshGeometry{T, N}, ops::SBPOps{N, T}, τ;
                            bdry_values::NTuple{6, T} = ntuple(_ -> zero(T), Val(6)),
                            drop_tol = zero(T)) where {N, T}
    Ne   = geom.mesh.Ne
    ndof = N^3 * Ne
    u  = zeros(T, N, N, N, Ne)
    u̇ = zeros(T, N, N, N, Ne)
    ü = similar(u)

    I_idx = Int[];  J_idx = Int[];  V = T[]
    sizehint!(I_idx, 64 * ndof);  sizehint!(J_idx, 64 * ndof);  sizehint!(V, 64 * ndof)

    u_flat = vec(u)
    ü_flat = vec(ü)
    @inbounds for col in 1:ndof
        u_flat[col] = one(T)
        rhs3d!(ü, u, u̇, bdry_values; geom, ops, τ)
        u_flat[col] = zero(T)
        for row in 1:ndof
            v = ü_flat[row]
            if abs(v) > drop_tol
                push!(I_idx, row);  push!(J_idx, col);  push!(V, v)
            end
        end
    end
    return sparse(I_idx, J_idx, V, ndof, ndof)
end

"""
    spectral_radius_estimate(geom, ops, τ;
                             iters = 60, tol = 1e-4) → T

Estimate `|λ_max|`, the magnitude of the largest eigenvalue of the
curvilinear discrete Laplacian, via power iteration. For the negative
semi-definite wave-equation operator this equals `|λ_min| = ω_max²`,
the square of the highest-frequency spatial mode. Used by
[`recommended_dt`](@ref) to size an explicit-integration timestep.

Power iteration on a random vector converges geometrically in
`|λ_2 / λ_1|`; `iters = 60` is enough for a few-digit estimate in
practice. Cost is `≈ iters` calls to `rhs3d!`.
"""
function spectral_radius_estimate(geom::MeshGeometry{T, N}, ops::SBPOps{N, T}, τ;
                                   iters::Int = 60, tol = T(1e-4)) where {N, T}
    Ne   = geom.mesh.Ne
    bdry = ntuple(_ -> zero(T), Val(6))
    x  = randn(T, N, N, N, Ne)
    y  = similar(x)
    u̇ = zeros(T, N, N, N, Ne)

    nx = sqrt(sum(abs2, x))
    nx == 0 && return zero(T)
    x ./= nx

    λ_prev = zero(T)
    @inbounds for k in 1:iters
        rhs3d!(y, x, u̇, bdry; geom, ops, τ)
        # Rayleigh quotient ⟨x, L x⟩ — `x` is already normalised.
        λ = zero(T)
        @simd for I in eachindex(x)
            λ += x[I] * y[I]
        end
        ny = sqrt(sum(abs2, y))
        ny == 0 && return abs(λ)
        if k > 5 && abs(λ - λ_prev) ≤ tol * abs(λ)
            return abs(λ)
        end
        λ_prev = λ
        x .= y ./ ny
    end
    return abs(λ_prev)
end

"""
    recommended_dt(geom, ops, τ; cfl_safety = 0.9) → T

Suggest a stable timestep for an explicit symplectic integrator
(`KahanLi8` and friends) on the wave equation `ü = L u`. Returns

    cfl_safety · 2 / sqrt(|λ_max|)

where `|λ_max|` is estimated by [`spectral_radius_estimate`](@ref).
The bare condition `dt · ω_max < 2` is the Störmer–Verlet stability
limit; `cfl_safety = 0.9` keeps a margin away from the edge.

Typical usage at the top of a driver:

```julia
geom = make_geometry(mesh, elem)
ops  = make_operators(elem)
τ    = 100.0
dt   = recommended_dt(geom, ops, τ)
```
"""
function recommended_dt(geom::MeshGeometry{T, N}, ops::SBPOps{N, T}, τ;
                         cfl_safety = T(0.9)) where {N, T}
    λ = spectral_radius_estimate(geom, ops, τ)
    λ == 0 && return T(Inf)
    return cfl_safety * T(2) / sqrt(λ)
end

"""
    discrete_inner_product(u, v, geom, ops) → T

Discrete physical inner product `⟨u, v⟩_{H_phys} = Σ H_phys · u · v`,
where `H_phys[i, j, k, e] = H_ref[i] H_ref[j] H_ref[k] · |det J|` is the
per-node mass induced by the GLL quadrature on the curvilinear element
map. This is the natural inner product on the SBP-DG discretisation —
it recovers the continuous `L²(Ω)` inner product in the high-resolution
limit and makes the discrete Laplacian (operator from `rhs3d!`) symmetric
in this inner product on a coercive mesh.

`u` and `v` are 4-D arrays of shape `(N, N, N, Ne)`.
"""
function discrete_inner_product(u::AbstractArray{T, 4}, v::AbstractArray{T, 4},
                                 geom::MeshGeometry{T, N},
                                 ops::SBPOps{N, T}) where {N, T}
    @assert size(u) == size(v) == (N, N, N, geom.mesh.Ne)
    H_1d = SVector{N, T}(ntuple(i -> ops.H[i, i], Val(N)))
    s = zero(T)
    @inbounds for e in 1:geom.mesh.Ne, k in 1:N, j in 1:N, i in 1:N
        Hp = H_1d[i] * H_1d[j] * H_1d[k] * geom.detjac[i, j, k, e]
        s += Hp * u[i, j, k, e] * v[i, j, k, e]
    end
    return s
end

"""
    discrete_l2_norm(u, geom, ops) → T

Physical-mass-weighted L² norm `sqrt(⟨u, u⟩_{H_phys})`. See
[`discrete_inner_product`](@ref).
"""
discrete_l2_norm(u::AbstractArray{T, 4}, geom::MeshGeometry{T, N},
                  ops::SBPOps{N, T}) where {N, T} =
    sqrt(discrete_inner_product(u, u, geom, ops))

"""
    physical_mass_diagonal(geom, ops) → Vector{T}

Diagonal of the global mass matrix `H_phys`, in the same linear node
ordering as `vec(u)`. Useful in conjunction with [`discrete_laplacian`](@ref)
to form `H_phys · L_h` or to solve the generalized eigenproblem
`L_h · v = λ · v` against the physical inner product.
"""
function physical_mass_diagonal(geom::MeshGeometry{T, N}, ops::SBPOps{N, T}) where {N, T}
    Ne = geom.mesh.Ne
    H_1d = SVector{N, T}(ntuple(i -> ops.H[i, i], Val(N)))
    out = Vector{T}(undef, N^3 * Ne)
    @inbounds for e in 1:Ne, k in 1:N, j in 1:N, i in 1:N
        idx = i + (j-1)*N + (k-1)*N^2 + (e-1)*N^3
        out[idx] = H_1d[i] * H_1d[j] * H_1d[k] * geom.detjac[i, j, k, e]
    end
    return out
end
