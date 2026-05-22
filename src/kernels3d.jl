# 3D kernels: per-element Laplacian + SAT on a tensor-product GLL block,
# and the corresponding global RHS over a cuboid mesh.
#
# Tensor-product GLL element: per-element data has shape (N, N, N). The 3D
# Laplacian is the sum of three 1D Laplacians (one per axis); each 1D
# contribution carries its own SAT at the two faces orthogonal to its axis.
# The 1D operators (`G`, `L`, `Hinv`, `HinvG_L`, `HinvG_R`) and the SAT
# increment (`_sat_increment`) are reused unchanged from `kernels1d.jl`.

################################################################################
# Initialisation

# Initialise a 3D scalar field on a tensor-product GLL element to the
# separable eigenmode
#     u(x,y,z,t) = A · sin(kx·x) · sin(ky·y) · sin(kz·z) · cos(ω·t)
# which satisfies u_tt = ∇²u for ω² = kx² + ky² + kz².
function initialize3d!(u::AbstractArray{T,3}, u̇::AbstractArray{T,3},
                       x::AbstractVector, y::AbstractVector, z::AbstractVector,
                       t; A, kx, ky, kz, ω) where {T}
    sx = reshape(sin.(kx .* x), :, 1, 1)
    sy = reshape(sin.(ky .* y), 1, :, 1)
    sz = reshape(sin.(kz .* z), 1, 1, :)
    @. u  =  A   * sx * sy * sz * cos(ω*t)
    @. u̇ = -A*ω * sx * sy * sz * sin(ω*t)
    return u, u̇
end

# Per-element initialise on a 6D state array `u[i, j, k, mx, my, mz]`.
function initialize3d!(u::AbstractArray{T,6}, u̇::AbstractArray{T,6},
                       x::AbstractMatrix, y::AbstractMatrix, z::AbstractMatrix, t;
                       A, kx, ky, kz, ω) where {T}
    Mx, My, Mz = size(u, 4), size(u, 5), size(u, 6)
    @assert size(u̇) == size(u)
    @assert size(x, 2) == Mx
    @assert size(y, 2) == My
    @assert size(z, 2) == Mz
    for mz in 1:Mz, my in 1:My, mx in 1:Mx
        initialize3d!(view(u,  :, :, :, mx, my, mz),
                      view(u̇, :, :, :, mx, my, mz),
                      view(x, :, mx), view(y, :, my), view(z, :, mz), t;
                      A, kx, ky, kz, ω)
    end
    return u, u̇
end

################################################################################
# Axis abstractions

# Fiber view: at face-position (p, q) along the two passive axes, return
# the 1D slice running along the active axis D. Axis dispatch is at compile
# time via `Val{D}`.
@inline _fiber_view(::Val{1}, A, p, q) = view(A, :, p, q)
@inline _fiber_view(::Val{2}, A, p, q) = view(A, p, :, q)
@inline _fiber_view(::Val{3}, A, p, q) = view(A, p, q, :)

# Scalar node access into the 6D state array: pick the active-axis position
# `i` and the two face-position coordinates `(p, q)` for element `(mx,my,mz)`.
# Axis dispatch via `Val{D}` collapses at compile time, so callers do direct
# 6D indexing without any intermediate `SubArray` allocation.
@inline _node(::Val{1}, u::AbstractArray{<:Any,6}, i, p, q, mx, my, mz) =
    @inbounds u[i, p, q, mx, my, mz]
@inline _node(::Val{2}, u::AbstractArray{<:Any,6}, i, p, q, mx, my, mz) =
    @inbounds u[p, i, q, mx, my, mz]
@inline _node(::Val{3}, u::AbstractArray{<:Any,6}, i, p, q, mx, my, mz) =
    @inbounds u[p, q, i, mx, my, mz]

# Read the N×N face slice of element `(mx, my, mz)` at the `row`-th node
# along axis D and return as a stack-allocated `SMatrix{N,N,T}`.
@inline function _face_smatrix(::Val{D}, ::Val{N},
                               u::AbstractArray{T,6}, row::Integer,
                               mx, my, mz) where {D, N, T}
    out = MMatrix{N,N,T}(undef)
    @inbounds for q in 1:N, p in 1:N
        out[p, q] = _node(Val(D), u, row, p, q, mx, my, mz)
    end
    return SMatrix(out)
end

# Compute the N×N matrix of (∂u/∂ξ_D) at the `row`-th face of element
# `(mx, my, mz)` and return as `SMatrix`. Two row-of-G dot products per
# face point. No `SubArray`s, no heap.
@inline function _face_gradient(::Val{D}, ::Val{N},
                                u::AbstractArray{T,6}, row::Integer,
                                mx, my, mz, ops::SBPOps{N,T}) where {D, N, T}
    G = ops.G
    out = MMatrix{N,N,T}(undef)
    @inbounds for q in 1:N, p in 1:N
        s = zero(T)
        for i in 1:N
            s += G[row, i] * _node(Val(D), u, i, p, q, mx, my, mz)
        end
        out[p, q] = s
    end
    return SMatrix(out)
end

################################################################################
# Per-element face data

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
# Per-element 3D Laplacian + SAT

# Apply the 1D Laplacian + SIPG-SAT along axis `D` of a 3D element block,
# *accumulating* into `Lu`. The two faces orthogonal to D each carry an
# N×N matrix of neighbour values and an N×N matrix of neighbour boundary
# gradients; the per-face α weights are scalars (no spatial variation
# along a single face).
#
# Per (p, q) on the face, this is exactly the 1D operation `apply_laplacian!`
# applied to one fiber, plus accumulation instead of overwrite.
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

# Uniform-value N×N face matrix: an `SMatrix{N,N,T}` with every entry equal
# to the scalar `b`. Built via `ntuple(_, Val(N*N))` so it is fully unrolled
# and stack-allocated.
@inline _uniform_face(::Val{N}, ::Type{T}, b) where {N, T} =
    SMatrix{N, N, T}(ntuple(_ -> T(b), Val(N*N)))

# 3D RHS over a cuboid mesh of elements. The six scalar arguments `bxL`,
# `bxR`, `byL`, `byR`, `bzL`, `bzR` set the (uniform) Dirichlet value on
# each of the six outer cuboid faces (= 0 reproduces the previous
# homogeneous behaviour). Element-local: each `(mx, my, mz)` iteration
# touches only `u[:,:,:, mx, my, mz]` and its six immediate neighbours'
# boundary slices.
function rhs3d!(ü::AbstractArray{T,6}, u::AbstractArray{T,6}, u̇::AbstractArray{T,6},
                bxL, bxR, byL, byR, bzL, bzR;
                dom, ops::SBPOps{N,T}, τ) where {N, T}
    Mx, My, Mz = size(u, 4), size(u, 5), size(u, 6)
    @assert size(ü) == size(u̇) == size(u)
    half = one(T) / 2

    @inbounds for mz in 1:Mz, my in 1:My, mx in 1:Mx
        ue = view(u,  :, :, :, mx, my, mz)
        üe = view(ü, :, :, :, mx, my, mz)

        # Build each face's value and gradient as `SMatrix` locals — fully
        # stack-allocated. Outer faces use a constant N×N matrix filled with
        # the boundary scalar (mirror gradient for ΔGu = 0).

        # --- −x face ---
        if mx == 1
            u_xm  = _uniform_face(Val(N), T, bxL)
            Gu_xm = _face_gradient(Val(1), Val(N), u, 1, mx, my, mz, ops)
            αx_m  = one(T)
        else
            u_xm  = _face_smatrix(Val(1), Val(N), u, N, mx-1, my, mz)
            Gu_xm = _face_gradient(Val(1), Val(N), u, N, mx-1, my, mz, ops)
            αx_m  = half
        end
        # --- +x face ---
        if mx == Mx
            u_xp  = _uniform_face(Val(N), T, bxR)
            Gu_xp = _face_gradient(Val(1), Val(N), u, N, mx, my, mz, ops)
            αx_p  = one(T)
        else
            u_xp  = _face_smatrix(Val(1), Val(N), u, 1, mx+1, my, mz)
            Gu_xp = _face_gradient(Val(1), Val(N), u, 1, mx+1, my, mz, ops)
            αx_p  = half
        end
        # --- −y face ---
        if my == 1
            u_ym  = _uniform_face(Val(N), T, byL)
            Gu_ym = _face_gradient(Val(2), Val(N), u, 1, mx, my, mz, ops)
            αy_m  = one(T)
        else
            u_ym  = _face_smatrix(Val(2), Val(N), u, N, mx, my-1, mz)
            Gu_ym = _face_gradient(Val(2), Val(N), u, N, mx, my-1, mz, ops)
            αy_m  = half
        end
        # --- +y face ---
        if my == My
            u_yp  = _uniform_face(Val(N), T, byR)
            Gu_yp = _face_gradient(Val(2), Val(N), u, N, mx, my, mz, ops)
            αy_p  = one(T)
        else
            u_yp  = _face_smatrix(Val(2), Val(N), u, 1, mx, my+1, mz)
            Gu_yp = _face_gradient(Val(2), Val(N), u, 1, mx, my+1, mz, ops)
            αy_p  = half
        end
        # --- −z face ---
        if mz == 1
            u_zm  = _uniform_face(Val(N), T, bzL)
            Gu_zm = _face_gradient(Val(3), Val(N), u, 1, mx, my, mz, ops)
            αz_m  = one(T)
        else
            u_zm  = _face_smatrix(Val(3), Val(N), u, N, mx, my, mz-1)
            Gu_zm = _face_gradient(Val(3), Val(N), u, N, mx, my, mz-1, ops)
            αz_m  = half
        end
        # --- +z face ---
        if mz == Mz
            u_zp  = _uniform_face(Val(N), T, bzR)
            Gu_zp = _face_gradient(Val(3), Val(N), u, N, mx, my, mz, ops)
            αz_p  = one(T)
        else
            u_zp  = _face_smatrix(Val(3), Val(N), u, 1, mx, my, mz+1)
            Gu_zp = _face_gradient(Val(3), Val(N), u, 1, mx, my, mz+1, ops)
            αz_p  = half
        end

        facex = FaceData(u_xm, u_xp, Gu_xm, Gu_xp, αx_m, αx_p)
        facey = FaceData(u_ym, u_yp, Gu_ym, Gu_yp, αy_m, αy_p)
        facez = FaceData(u_zm, u_zp, Gu_zm, Gu_zp, αz_m, αz_p)
        apply_laplacian3d!(üe, ue, facex, facey, facez; ops, τ)
    end

    ü .*= inv(dom.h^2)
    return ü
end
