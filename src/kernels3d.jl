# 3D kernels: per-element Laplacian + SAT on a tensor-product GLL block,
# and the corresponding global RHS over a `HexMesh` of conforming hexes.
#
# Tensor-product GLL element: per-element data has shape (N, N, N). The 3D
# Laplacian is the sum of three 1D Laplacians (one per axis); each 1D
# contribution carries its own SAT at the two faces orthogonal to its axis.
# The 1D operators (`G`, `L`, `Hinv`, `HinvG_L`, `HinvG_R`) and the SAT
# increment (`_sat_increment`) are reused unchanged from `kernels1d.jl`.
#
# Global state arrays are `Array{T, 4}` of shape `(N, N, N, Ne)` — one
# (N, N, N) block per element, ordered as a 1-D list of elements. Face
# connectivity comes from the `HexMesh` rather than from the array layout,
# which lets the same kernel handle structured cubical, unstructured, and
# (eventually) multi-block / refined meshes with no further changes.

################################################################################
# Initialisation

# Per-element: evaluate the analytic separable eigenmode on a 3D node block,
# given the physical (x, y, z) coordinate of each node.
function initialize3d!(u::AbstractArray{T,3}, u̇::AbstractArray{T,3},
                       x::AbstractArray{T,3}, y::AbstractArray{T,3}, z::AbstractArray{T,3},
                       t; A, kx, ky, kz, ω) where {T}
    @. u  =  A   * sin(kx*x) * sin(ky*y) * sin(kz*z) * cos(ω*t)
    @. u̇ = -A*ω * sin(kx*x) * sin(ky*y) * sin(kz*z) * sin(ω*t)
    return u, u̇
end

# Global: walk the 1-D element list, dispatch to the per-element method
# with the appropriate slice of the (3, N, N, N, Ne) coordinate array.
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
# Per-element face data (one struct per axis, lives on the stack)

# Per-axis face-pair data passed to `add_axis_laplacian3d!`: neighbour value
# and boundary-gradient slices at the two faces orthogonal to a single axis,
# plus the two per-face SIPG/Nitsche consistency weights. `FaceData` is
# fully `isbits` (all fields are SMatrix or T), so it is stack-allocated on
# CPU and lives in registers / shared memory on GPU — no heap traffic.
struct FaceData{N, T, NN}
    u_minus  :: SMatrix{N, N, T, NN}
    u_plus   :: SMatrix{N, N, T, NN}
    Gu_minus :: SMatrix{N, N, T, NN}
    Gu_plus  :: SMatrix{N, N, T, NN}
    α_minus  :: T
    α_plus   :: T
end

################################################################################
# Axis abstractions

# Fiber view: at face-position (p, q) along the two passive axes, return
# the 1D slice running along the active axis D. Axis dispatch is at compile
# time via `Val{D}`.
@inline _fiber_view(::Val{1}, A, p, q) = view(A, :, p, q)
@inline _fiber_view(::Val{2}, A, p, q) = view(A, p, :, q)
@inline _fiber_view(::Val{3}, A, p, q) = view(A, p, q, :)

# Scalar node access into the 4D state array (N, N, N, Ne): pick the
# active-axis position `i` and the two face-position coordinates `(p, q)`
# for element `e`. Axis dispatch via `Val{D}` collapses at compile time,
# so callers do direct 4D indexing without any `SubArray` allocation.
@inline _node(::Val{1}, u::AbstractArray{<:Any,4}, i, p, q, e) =
    @inbounds u[i, p, q, e]
@inline _node(::Val{2}, u::AbstractArray{<:Any,4}, i, p, q, e) =
    @inbounds u[p, i, q, e]
@inline _node(::Val{3}, u::AbstractArray{<:Any,4}, i, p, q, e) =
    @inbounds u[p, q, i, e]

# Uniform-value N×N face matrix: every entry equals `b`. Built via
# `ntuple(_, Val(N*N))` for full unrolling and stack allocation.
@inline _uniform_face(::Val{N}, ::Type{T}, b) where {N, T} =
    SMatrix{N, N, T}(ntuple(_ -> T(b), Val(N*N)))

# Read the N×N face slice of element `e` at the `row`-th node along axis
# D, returning an `SMatrix{N,N,T}`.
@inline function _face_smatrix(::Val{D}, ::Val{N},
                               u::AbstractArray{T,4}, row::Integer,
                               e) where {D, N, T}
    out = MMatrix{N,N,T}(undef)
    @inbounds for q in 1:N, p in 1:N
        out[p, q] = _node(Val(D), u, row, p, q, e)
    end
    return SMatrix(out)
end

# Compute the N×N matrix of (∂u/∂ξ_D) at the `row`-th face of element `e`
# (two row-of-G dot products per face point). Returns `SMatrix{N,N,T}`.
# No `SubArray`s, no heap.
@inline function _face_gradient(::Val{D}, ::Val{N},
                                u::AbstractArray{T,4}, row::Integer,
                                e, ops::SBPOps{N,T}) where {D, N, T}
    G = ops.G
    out = MMatrix{N,N,T}(undef)
    @inbounds for q in 1:N, p in 1:N
        s = zero(T)
        for i in 1:N
            s += G[row, i] * _node(Val(D), u, i, p, q, e)
        end
        out[p, q] = s
    end
    return SMatrix(out)
end

################################################################################
# Per-element 3D Laplacian + SAT

# Apply the 1D Laplacian + SIPG-SAT along axis `D` of a 3D element block,
# *accumulating* into `Lu`. Loads each fiber as an `SVector{N}`, computes
# `ops.L * fiber + _sat_increment(...)` statically (no heap), writes back.
function add_axis_laplacian3d!(::Val{D},
                                Lu::AbstractArray{T,3}, u::AbstractArray{T,3},
                                face::FaceData{N,T};
                                ops::SBPOps{N,T}, τ) where {D, T, N}
    @inbounds for q in 1:N, p in 1:N
        u_f = SVector{N}(_fiber_view(Val(D), u, p, q))

        ΔuL  = u_f[1] - face.u_minus[p, q]
        ΔuR  = u_f[N] - face.u_plus[p, q]
        GuL  = dot(ops.G[1, :], u_f)
        GuR  = dot(ops.G[N, :], u_f)
        ΔGuL = GuL - face.Gu_minus[p, q]
        ΔGuR = GuR - face.Gu_plus[p, q]

        inc = ops.L * u_f +
              _sat_increment(ΔuL, ΔuR, ΔGuL, ΔGuR,
                             face.α_minus, face.α_plus, ops, τ)

        Lu_f = _fiber_view(Val(D), Lu, p, q)
        for i in 1:N
            Lu_f[i] += inc[i]
        end
    end
    return Lu
end

# Apply the 3D Laplacian + SIPG-SAT to one element. Face data is grouped
# into one `FaceData` per axis (six SMatrix slices + two α scalars each).
function apply_laplacian3d!(Lu::AbstractArray{T,3}, u::AbstractArray{T,3},
                            facex::FaceData{N,T},
                            facey::FaceData{N,T},
                            facez::FaceData{N,T};
                            ops::SBPOps{N,T}, τ) where {N, T}
    fill!(Lu, zero(T))
    add_axis_laplacian3d!(Val(1), Lu, u, facex; ops, τ)
    add_axis_laplacian3d!(Val(2), Lu, u, facey; ops, τ)
    add_axis_laplacian3d!(Val(3), Lu, u, facez; ops, τ)
    return Lu
end

################################################################################
# 3D global RHS

# Map a mesh face index (1..6) to (active-axis Val, own-face row, neighbour
# row). Face indices follow the `HexMesh` convention:
#
#     1 = −x   2 = +x   3 = −y   4 = +y   5 = −z   6 = +z
#
# Own row at the −face is 1, at the +face is N; the neighbour's matching
# face is the opposite row.
@inline _face_axis(::Val{1}) = Val(1)
@inline _face_axis(::Val{2}) = Val(1)
@inline _face_axis(::Val{3}) = Val(2)
@inline _face_axis(::Val{4}) = Val(2)
@inline _face_axis(::Val{5}) = Val(3)
@inline _face_axis(::Val{6}) = Val(3)
@inline _face_self_row(::Val{f},  ::Val{N}) where {f, N} = isodd(f) ? 1 : N
@inline _face_neigh_row(::Val{f}, ::Val{N}) where {f, N} = isodd(f) ? N : 1

# Build the `FaceData` for one axis (a pair of opposite mesh faces `fm`,
# `fp` with `fp = fm + 1`) of one element.
@inline function _axis_face_data(::Val{fm}, ::Val{fp}, ::Val{N},
                                 u::AbstractArray{T,4}, e,
                                 mesh::HexMesh{T},
                                 bdry_values::NTuple{6, T},
                                 ops::SBPOps{N,T}) where {fm, fp, N, T}
    half = T(1) / 2
    axisV = _face_axis(Val(fm))

    # − face (mesh face index `fm`)
    nm = mesh.neighbour[fm, e]
    if nm == 0
        u_m  = _uniform_face(Val(N), T, bdry_values[mesh.bdry[fm, e]])
        Gu_m = _face_gradient(axisV, Val(N), u,
                              _face_self_row(Val(fm), Val(N)), e, ops)
        α_m  = one(T)
    else
        nrow = _face_neigh_row(Val(fm), Val(N))
        u_m  = _face_smatrix(axisV, Val(N), u, nrow, nm)
        Gu_m = _face_gradient(axisV, Val(N), u, nrow, nm, ops)
        α_m  = half
    end

    # + face (mesh face index `fp`)
    np = mesh.neighbour[fp, e]
    if np == 0
        u_p  = _uniform_face(Val(N), T, bdry_values[mesh.bdry[fp, e]])
        Gu_p = _face_gradient(axisV, Val(N), u,
                              _face_self_row(Val(fp), Val(N)), e, ops)
        α_p  = one(T)
    else
        nrow = _face_neigh_row(Val(fp), Val(N))
        u_p  = _face_smatrix(axisV, Val(N), u, nrow, np)
        Gu_p = _face_gradient(axisV, Val(N), u, nrow, np, ops)
        α_p  = half
    end
    return FaceData(u_m, u_p, Gu_m, Gu_p, α_m, α_p)
end

"""
    rhs3d!(ü, u, u̇, bdry_values; mesh, ops, τ)

3D RHS over the `HexMesh` `mesh`. State arrays are 4D, shape (N, N, N, Ne),
with element ordering matching `mesh`. The 6-tuple `bdry_values` is indexed
by the boundary tag stored on each outer face (`mesh.bdry[f, e]`), so
`bdry_values[k]` is the uniform Dirichlet value at every outer face
carrying tag `k`.

Element-local: each iteration touches only `u[:,:,:, e]` and the boundary
slices of its (up to six) immediate neighbours, looked up through
`mesh.neighbour[f, e]`. Outer faces (`neighbour == 0`) use the boundary
scalar plus the mirror-gradient convention.

The Laplacian is scaled per element by `1/h_e²` where `h_e` is taken from
the (axis-aligned, uniform) hex's vertex extent. For non-uniform or curved
meshes this scaling will eventually be replaced by a per-node Jacobian /
inverse-metric.
"""
function rhs3d!(ü::AbstractArray{T,4}, u::AbstractArray{T,4}, u̇::AbstractArray{T,4},
                bdry_values::NTuple{6, T};
                mesh::HexMesh{T}, ops::SBPOps{N,T}, τ) where {N, T}
    @assert size(ü) == size(u̇) == size(u)
    @assert size(u, 1) == size(u, 2) == size(u, 3) == N
    @assert size(u, 4) == mesh.Ne

    @inbounds for e in 1:mesh.Ne
        ue = view(u,  :, :, :, e)
        üe = view(ü, :, :, :, e)

        facex = _axis_face_data(Val(1), Val(2), Val(N), u, e, mesh, bdry_values, ops)
        facey = _axis_face_data(Val(3), Val(4), Val(N), u, e, mesh, bdry_values, ops)
        facez = _axis_face_data(Val(5), Val(6), Val(N), u, e, mesh, bdry_values, ops)

        apply_laplacian3d!(üe, ue, facex, facey, facez; ops, τ)

        # Per-element 1/h² scaling. For axis-aligned uniform hexes the
        # extent along any axis works; for curved or non-uniform hexes,
        # replace with a per-node inverse-metric scaling. Corner 1 is the
        # (−x, −y, −z) vertex; corner 2 is (+x, −y, −z), so corners 1→2
        # span one edge of the element along x.
        v1 = mesh.vertex_idx[1, e]
        v2 = mesh.vertex_idx[2, e]
        h_e = mesh.vertex_coords[1, v2] - mesh.vertex_coords[1, v1]
        inv_h2 = T(1) / (h_e * h_e)
        for k in 1:N, j in 1:N, i in 1:N
            üe[i, j, k] *= inv_h2
        end
    end

    return ü
end
