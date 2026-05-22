module WaveToySecondOrder

using FastGaussQuadrature
using LinearAlgebra
using OrdinaryDiffEqSymplecticRK
using PolynomialBases: LobattoLegendre
using StaticArrays

# Operator container with N as a compile-time parameter. The eight fields
# (B, G, H, Hinv, HinvG_L, HinvG_R, D, L) match the previous NamedTuple
# layout exactly, so callers can keep using `ops.G`, `ops.L`, etc. The
# only change is that field types are now static (SMatrix / SVector) —
# value-time `N` is propagated through the type system, which is the
# prerequisite for the inner kernels to operate on `SVector` fibers and
# generate fully-unrolled `SMatrix` arithmetic.
struct SBPOps{N, T, NN}
    B       :: SMatrix{N, N, T, NN}
    G       :: SMatrix{N, N, T, NN}
    H       :: SMatrix{N, N, T, NN}
    Hinv    :: SMatrix{N, N, T, NN}
    HinvG_L :: SVector{N, T}
    HinvG_R :: SVector{N, T}
    D       :: SMatrix{N, N, T, NN}
    L       :: SMatrix{N, N, T, NN}
end

################################################################################

function make_element(::Type{T}, N::Int) where {T}
    ns = 0:N-1
    x0 = T(0//1)
    x1 = T(1//1)

    if T <: Rational
        # Equispaced "vertex-centred" points keep the operator construction
        # exact in rational arithmetic.
        h = (x1-x0) / (N-1)
        xs = x0 .+ h * ns
    else
        # Gauss-Lobatto-Legendre collocation points (better-conditioned SBP
        # operators than equispaced). `gausslobatto` returns nodes on [-1,1];
        # map them linearly to [x0, x1].
        ξ, _ = gausslobatto(N)
        xs = T.((ξ .+ 1) ./ 2) .* (x1 - x0) .+ x0
        # GLL points cluster near the endpoints; the minimum spacing sets
        # the effective CFL length scale.
        h = minimum(diff(xs))
    end

    return (;N,ns,x0,x1,h,xs)
end

function make_domain(::Type{T}, N::Int, x0, x1) where {T}
    ns = 0:N-1

    # Calculate operators with rational numbers for improved accuracy
    x0 = T(x0)
    x1 = T(x1)
    # N elements, "cell-centred"
    h = (x1-x0) / N
    xs = x0 .+ h * ns

    return (;N,ns,x0,x1,h,xs)
end

# Top-level entry: dispatch to a `Val{N}`-parameterised implementation so
# the returned `SBPOps` has fully-concrete static types. The dispatch site
# is type-unstable but `make_operators` is called once per simulation, so
# the cost is irrelevant.
make_operators(dom) = _make_operators(Val(dom.N), dom)

function _make_operators(::Val{N}, dom) where {N}
    _, ns, x0, x1, h, xs = dom
    T = eltype(xs)

    # Boundary operator (same in both branches)
    B = Diagonal([ifelse(n==ns[begin], -T(1), ifelse(n==ns[end], +T(1), zero(T)))
                  for n in ns])

    if T <: Rational
        # Polynomial-exactness construction. Exact in rational arithmetic.
        # Numerically catastrophic in floating point for N ≳ 13 because the
        # Vandermonde-style `lhs` matrix has condition number ∼ 4ᴺ.
        G = let
            lhs = [x^p for x in xs, p in 0:N-1]
            rhs = [p==0 ? 0 : p * x^(p-1) for x in xs, p in 0:N-1]
            G = rhs / lhs
            @assert G * lhs == rhs
            @assert all(G * xs.^0 .== 0)
            @assert all(G * xs.^p == p * xs.^(p-1) for p in 1:N-1)
            G
        end
        H = let
            u = [x^p for x in xs, p in 0:N-1]
            v = [x^q for x in xs, q in 0:N-1]
            r = [1//(p+q+1) * (x1^(p+q+1) - x0^(p+q+1)) for p in 0:N-1, q in 0:N-1]
            H = (u' \ r) / v
            @assert issymmetric(H)
            @assert all(>(0), eigvals(H))
            @assert all(u[:,k]' * H * v[:,l] == r[k,l] for k in 1:N, l in 1:N)
            H
        end
        D = H \ (B - (H*G)')
        @assert D == G
        @assert all(D * xs.^0 .== 0)
        @assert all(D * xs.^p == p * xs.^(p-1) for p in 1:N-1)
    else
        # GLL spectral-collocation operators via PolynomialBases.jl —
        # numerically stable for any N. The basis is defined on [−1, 1];
        # we map to [x0, x1] via the Jacobian dx/dξ = (x1−x0)/2.
        basis = LobattoLegendre(N - 1, T)
        jac   = (x1 - x0) / T(2)
        G = basis.D ./ jac                # ∂/∂x = (1/jac) · ∂/∂ξ
        H = Diagonal(basis.weights .* jac) # ∫·dx = jac · ∫·dξ
        D = G                              # diagonal-norm GLL collocation: D = G
    end

    # Laplacian (without boundary conditions)
    L = D * G

    Hinv = inv(H)

    # Precompute Hinv · (first/last rows of G), used by the SAT consistency
    # term `Hinv · Gᵀ · B · …` which only touches the first and last columns
    # of `Gᵀ` (i.e., the first and last rows of `G`).
    HinvG_L = Hinv * G[1, :]
    HinvG_R = Hinv * G[N, :]

    # Build the concretely-typed static container. `SMatrix{N,N,T}(A)` and
    # `SVector{N,T}(v)` copy values into stack-allocated static storage; the
    # Matrix/Diagonal sources are discarded after construction.
    return SBPOps{N, T, N*N}(
        SMatrix{N, N, T}(B),
        SMatrix{N, N, T}(G),
        SMatrix{N, N, T}(H),
        SMatrix{N, N, T}(Hinv),
        SVector{N, T}(HinvG_L),
        SVector{N, T}(HinvG_R),
        SMatrix{N, N, T}(D),
        SMatrix{N, N, T}(L),
    )
end



# SIPG-style SAT penalty. Exchanges both u and Gu across the face.
#
#     L_SAT u = L u + H⁻¹ [ (1/2) Gᵀ B Δu  −  (1/2) B ΔGu  −  τ |B| Δu ]
#
# `gL`, `gR`    : neighbour's u  (or boundary data) at the left/right face.
# `gGL`, `gGR`  : neighbour's Gu (or `Gu_local` for outer mirror) at the face.
#
# At interior interfaces (sharing both u and Gu), the summed contribution to
# `vᵀ H L_SAT u` is the symmetric SIPG bilinear form
#     −(Gv,Gu)_H + {Gu}[v] + {Gv}[u] − τ [u][v] ,
# and Hᴸ_SAT is symmetric at the interior — eliminating the kink-mode null
# space of the value-only coupling.
# Per-face consistency weight: 1 at outer Dirichlet faces (full Nitsche),
# 1/2 at interior interfaces (SIPG averages between two elements).
# SAT increment as a pure function returning a stack-allocated SVector.
# Combines the four scalar coefficients with the precomputed `HinvG_L`,
# `HinvG_R` and the first/last columns of `Hinv` to form the per-fiber
# correction `H⁻¹·[(½)·GᵀB·Δu − (½)·B·ΔGu − τ·|B|·Δu]`.
@inline function _sat_increment(ΔuL, ΔuR, ΔGuL, ΔGuR, αL, αR,
                                ops::SBPOps{N,T}, τ) where {N, T}
    cL = -αL * ΔuL
    cR =  αR * ΔuR
    b1 =  T(1//2) * ΔGuL - τ * ΔuL
    bN = -T(1//2) * ΔGuR - τ * ΔuR
    return cL * ops.HinvG_L + cR * ops.HinvG_R +
           b1 * ops.Hinv[:, 1] + bN * ops.Hinv[:, N]
end

# Mutating wrapper, kept for callers that still want to accumulate into a
# regular `AbstractVector` (e.g. when writing back to a column of a global
# array).
function add_dirichlet_penalties!(Lu::AbstractVector,
                                  ΔuL, ΔuR, ΔGuL, ΔGuR, αL, αR;
                                  ops::SBPOps{N}, τ) where {N}
    inc = _sat_increment(ΔuL, ΔuR, ΔGuL, ΔGuR, αL, αR, ops, τ)
    @inbounds for i in 1:N
        Lu[i] += inc[i]
    end
    return Lu
end

# Pure-functional 1D Laplacian + SAT: loads `u` into an SVector and returns
# the result as an SVector, never touching the heap.
@inline function _apply_laplacian(u_s::SVector{N,T},
                                  gL, gR, gGL, gGR, αL, αR,
                                  ops::SBPOps{N,T}, τ) where {N, T}
    ΔuL  = u_s[1] - gL
    ΔuR  = u_s[N] - gR
    GuL  = dot(ops.G[1, :], u_s)
    GuR  = dot(ops.G[N, :], u_s)
    ΔGuL = GuL - gGL
    ΔGuR = GuR - gGR
    return ops.L * u_s + _sat_increment(ΔuL, ΔuR, ΔGuL, ΔGuR, αL, αR, ops, τ)
end

# Convenience wrapper: caller supplies `u` and the neighbour data; we load
# to SVector, compute statically, write back into `Lu`.
function apply_laplacian!(Lu::AbstractVector, u::AbstractVector,
                          gL, gR, gGL, gGR, αL, αR;
                          ops::SBPOps{N}, τ) where {N}
    result = _apply_laplacian(SVector{N}(u), gL, gR, gGL, gGR, αL, αR, ops, τ)
    @inbounds for i in 1:N
        Lu[i] = result[i]
    end
    return Lu
end



# Assemble the global L_SAT for `M` elements coupled DG-style. Outer faces use
# zero Dirichlet data and the mirror convention `gGu = Gu_local` (so ΔGu = 0
# at the outer boundary). Interior interfaces exchange both u and Gu.
function build_global_laplacian(M::Integer; ops, τ)
    G, L = ops.G, ops.L
    N = size(L, 1)

    T = eltype(L)
    n = N * M
    A = zeros(T, n, n)
    for j in 1:n
        e = zeros(T, n)
        e[j] = one(T)
        # precompute Gu per element
        Gu_all = [G * e[(i-1)*N+1 : i*N] for i in 1:M]
        for i in 1:M
            rng = (i-1)*N+1 : i*N
            gL  = i == 1 ? zero(T)             : e[first(rng) - 1]
            gR  = i == M ? zero(T)             : e[last(rng)  + 1]
            # outer mirror: ΔGu = 0 by using local value
            gGL = i == 1 ? Gu_all[i][begin]    : Gu_all[i-1][end]
            gGR = i == M ? Gu_all[i][end]      : Gu_all[i+1][begin]
            αL  = i == 1 ? one(T)              : one(T) / 2
            αR  = i == M ? one(T)              : one(T) / 2
            apply_laplacian!(view(A, rng, j), view(e, rng),
                             gL, gR, gGL, gGR, αL, αR; ops, τ=τ)
        end
    end
    return A
end



# One-element functions (1D)

function initialize!(u::AbstractVector, u̇::AbstractVector, x::AbstractVector, t; A,k,ω)
    u .=  A   * sin.(k*x) * cos(ω*t)
    u̇ .= -A*ω * sin.(k*x) * sin(ω*t)
    return u, u̇
end



# One-element functions (3D)
#
# Tensor-product GLL element: per-element data has shape (N, N, N). The 3D
# Laplacian is the sum of three 1D Laplacians (one per axis); each 1D
# contribution carries its own SAT at the two faces orthogonal to its axis.
# The 1D operators (`G`, `L`, `Hinv`, `HinvG_L`, `HinvG_R`) and the scalar
# SAT updater `add_dirichlet_penalties!` are reused unchanged.

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

# Fiber view: at face-position (p, q) along the two passive axes, return
# the 1D slice running along the active axis D. Axis dispatch is at compile
# time via `Val{D}`.
@inline _fiber_view(::Val{1}, A, p, q) = view(A, :, p, q)
@inline _fiber_view(::Val{2}, A, p, q) = view(A, p, :, q)
@inline _fiber_view(::Val{3}, A, p, q) = view(A, p, q, :)

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
                                uminus::SMatrix{N,N,T}, uplus::SMatrix{N,N,T},
                                Guminus::SMatrix{N,N,T}, Guplus::SMatrix{N,N,T},
                                αminus, αplus;
                                ops::SBPOps{N,T}, τ) where {D, T, N}
    @inbounds for q in 1:N, p in 1:N
        u_f = SVector{N}(_fiber_view(Val(D), u, p, q))

        ΔuL  = u_f[1] - uminus[p, q]
        ΔuR  = u_f[N] - uplus[p, q]
        GuL  = dot(ops.G[1, :], u_f)
        GuR  = dot(ops.G[N, :], u_f)
        ΔGuL = GuL - Guminus[p, q]
        ΔGuR = GuR - Guplus[p, q]

        inc = ops.L * u_f +
              _sat_increment(ΔuL, ΔuR, ΔGuL, ΔGuR, αminus, αplus, ops, τ)

        Lu_f = _fiber_view(Val(D), Lu, p, q)
        for i in 1:N
            Lu_f[i] += inc[i]
        end
    end
    return Lu
end

# Apply the 3D Laplacian + SIPG-SAT to one element. Six faces' worth of
# data are passed as 18 positional args (six per axis: neighbour value
# slice, neighbour gradient slice, and two α weights). This avoids the
# heap-allocated NamedTuple-of-views pattern.
function apply_laplacian3d!(Lu::AbstractArray{T,3}, u::AbstractArray{T,3},
                            ux_m::SMatrix{N,N,T}, ux_p::SMatrix{N,N,T},
                            Gux_m::SMatrix{N,N,T}, Gux_p::SMatrix{N,N,T}, αx_m, αx_p,
                            uy_m::SMatrix{N,N,T}, uy_p::SMatrix{N,N,T},
                            Guy_m::SMatrix{N,N,T}, Guy_p::SMatrix{N,N,T}, αy_m, αy_p,
                            uz_m::SMatrix{N,N,T}, uz_p::SMatrix{N,N,T},
                            Guz_m::SMatrix{N,N,T}, Guz_p::SMatrix{N,N,T}, αz_m, αz_p;
                            ops::SBPOps{N,T}, τ) where {N, T}
    fill!(Lu, zero(T))
    add_axis_laplacian3d!(Val(1), Lu, u, ux_m, ux_p, Gux_m, Gux_p, αx_m, αx_p; ops, τ)
    add_axis_laplacian3d!(Val(2), Lu, u, uy_m, uy_p, Guy_m, Guy_p, αy_m, αy_p; ops, τ)
    add_axis_laplacian3d!(Val(3), Lu, u, uz_m, uz_p, Guz_m, Guz_p, αz_m, αz_p; ops, τ)
    return Lu
end



# Global functions

function initialize!(u::AbstractMatrix, u̇::AbstractMatrix, x::AbstractMatrix, t; A,k,ω)
    M = size(u,2)
    @assert size(u̇,2) == size(x,2) == M
    for m in 1:M
        initialize!(view(u, : , m), view(u̇, : , m), view(x, : , m), t; A,k,ω)
    end
    return u, u̇
end

function rhs!(ü::AbstractMatrix, u::AbstractMatrix, u̇::AbstractMatrix, bL, bR;
              dom, ops::SBPOps{N,T}, τ) where {N, T}
    M = size(ü, 2)
    @assert size(u,2) == size(u̇,2) == M
    half = one(T) / 2

    # Each iteration loads its own column and the boundary slices of the
    # immediate neighbours into stack-allocated SVectors. All per-element
    # arithmetic is then `SMatrix · SVector` / `SVector` algebra — fully
    # unrolled by the compiler with no heap activity.
    @inbounds for m in 1:M
        u_self = SVector{N}(view(u, :, m))

        GuL_self = dot(ops.G[1, :], u_self)
        GuR_self = dot(ops.G[N, :], u_self)

        # Left face.
        if m == 1
            ΔuL  = u_self[1] - bL
            ΔGuL = zero(T)
            αL   = one(T)
        else
            u_left = SVector{N}(view(u, :, m-1))
            ΔuL  = u_self[1] - u_left[N]
            ΔGuL = GuL_self - dot(ops.G[N, :], u_left)
            αL   = half
        end

        # Right face — symmetric.
        if m == M
            ΔuR  = u_self[N] - bR
            ΔGuR = zero(T)
            αR   = one(T)
        else
            u_right = SVector{N}(view(u, :, m+1))
            ΔuR  = u_self[N] - u_right[1]
            ΔGuR = GuR_self - dot(ops.G[1, :], u_right)
            αR   = half
        end

        result = ops.L * u_self +
                 _sat_increment(ΔuL, ΔuR, ΔGuL, ΔGuR, αL, αR, ops, τ)
        for i in 1:N
            ü[i, m] = result[i]
        end
    end

    ü .*= inv(dom.h^2)
    return ü
end

# Global functions (3D)

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

# Scalar node access into the 6D state array: pick the active-axis position
# `i` and the two face-position coordinates `(p, q)` for element `(mx,my,mz)`.
# Axis dispatch via `Val{D}` collapses at compile time, so the body of
# `_face_gradient!` and friends ends up doing direct 6D array indexing
# without any intermediate `SubArray` allocation.
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

# 3D RHS over a cuboid mesh of elements. Homogeneous Dirichlet on all six
# outer faces. Element-local: each `(mx, my, mz)` iteration touches only
# `u[:,:,:, mx, my, mz]` and its six immediate neighbours' boundary slices.
function rhs3d!(ü::AbstractArray{T,6}, u::AbstractArray{T,6}, u̇::AbstractArray{T,6};
                dom, ops::SBPOps{N,T}, τ) where {N, T}
    Mx, My, Mz = size(u, 4), size(u, 5), size(u, 6)
    @assert size(ü) == size(u̇) == size(u)
    half = one(T) / 2
    Z = zero(SMatrix{N,N,T})   # outer Dirichlet face value (u = 0)

    @inbounds for mz in 1:Mz, my in 1:My, mx in 1:Mx
        ue = view(u,  :, :, :, mx, my, mz)
        üe = view(ü, :, :, :, mx, my, mz)

        # Build each face's value and gradient as `SMatrix` locals — fully
        # stack-allocated. Outer faces use the homogeneous Dirichlet zero
        # matrix `Z` and the mirror (own) gradient.

        # --- −x face ---
        if mx == 1
            u_xm  = Z
            Gu_xm = _face_gradient(Val(1), Val(N), u, 1, mx, my, mz, ops)
            αx_m  = one(T)
        else
            u_xm  = _face_smatrix(Val(1), Val(N), u, N, mx-1, my, mz)
            Gu_xm = _face_gradient(Val(1), Val(N), u, N, mx-1, my, mz, ops)
            αx_m  = half
        end
        # --- +x face ---
        if mx == Mx
            u_xp  = Z
            Gu_xp = _face_gradient(Val(1), Val(N), u, N, mx, my, mz, ops)
            αx_p  = one(T)
        else
            u_xp  = _face_smatrix(Val(1), Val(N), u, 1, mx+1, my, mz)
            Gu_xp = _face_gradient(Val(1), Val(N), u, 1, mx+1, my, mz, ops)
            αx_p  = half
        end
        # --- −y face ---
        if my == 1
            u_ym  = Z
            Gu_ym = _face_gradient(Val(2), Val(N), u, 1, mx, my, mz, ops)
            αy_m  = one(T)
        else
            u_ym  = _face_smatrix(Val(2), Val(N), u, N, mx, my-1, mz)
            Gu_ym = _face_gradient(Val(2), Val(N), u, N, mx, my-1, mz, ops)
            αy_m  = half
        end
        # --- +y face ---
        if my == My
            u_yp  = Z
            Gu_yp = _face_gradient(Val(2), Val(N), u, N, mx, my, mz, ops)
            αy_p  = one(T)
        else
            u_yp  = _face_smatrix(Val(2), Val(N), u, 1, mx, my+1, mz)
            Gu_yp = _face_gradient(Val(2), Val(N), u, 1, mx, my+1, mz, ops)
            αy_p  = half
        end
        # --- −z face ---
        if mz == 1
            u_zm  = Z
            Gu_zm = _face_gradient(Val(3), Val(N), u, 1, mx, my, mz, ops)
            αz_m  = one(T)
        else
            u_zm  = _face_smatrix(Val(3), Val(N), u, N, mx, my, mz-1)
            Gu_zm = _face_gradient(Val(3), Val(N), u, N, mx, my, mz-1, ops)
            αz_m  = half
        end
        # --- +z face ---
        if mz == Mz
            u_zp  = Z
            Gu_zp = _face_gradient(Val(3), Val(N), u, N, mx, my, mz, ops)
            αz_p  = one(T)
        else
            u_zp  = _face_smatrix(Val(3), Val(N), u, 1, mx, my, mz+1)
            Gu_zp = _face_gradient(Val(3), Val(N), u, 1, mx, my, mz+1, ops)
            αz_p  = half
        end

        apply_laplacian3d!(üe, ue,
                           u_xm, u_xp, Gu_xm, Gu_xp, αx_m, αx_p,
                           u_ym, u_yp, Gu_ym, Gu_yp, αy_m, αy_p,
                           u_zm, u_zp, Gu_zm, Gu_zp, αz_m, αz_p;
                           ops, τ)
    end

    ü .*= inv(dom.h^2)
    return ü
end

function evolve3d(x0, x1, M, N)
    elem = make_element(Float64, N)
    ops  = make_operators(elem)
    dom  = make_domain(Float64, M, x0, x1)

    # Per-axis node coordinates: x[i, m] = element-m's i-th GLL node along x.
    # For an isotropic cubic mesh we use the same dom for all three axes.
    x = [x + dom.h * a for a in elem.xs, x in dom.xs]
    y = x
    z = x
    dx = dom.h * elem.h

    u  = Array{Float64, 6}(undef, N, N, N, M, M, M)
    u̇  = similar(u)

    t0, t1 = 0.0, 1.0
    A = 1.0
    kx = ky = kz = 2π
    ω  = sqrt(kx^2 + ky^2 + kz^2)
    initialize3d!(u, u̇, x, y, z, t0; A, kx, ky, kz, ω)

    # τ ≳ 2·(N−1)² is the SIPG threshold in unit-element coords; use a safety
    # factor. The 3D spectral radius is ≈ 3× the 1D one, so dt ≈ (1/√3)·dt_1D.
    τ  = 3//2 * (N-1)^2
    dt = (1//2 * dx) / sqrt(3)

    f!(ü, u̇, u, p, t) = rhs3d!(ü, u, u̇; dom, ops, τ)
    prob = SecondOrderODEProblem(f!, u̇, u, (t0, t1))
    sol  = solve(prob, KahanLi8(); dt)

    # Diagnostic: compare to the analytic eigenmode at t1.
    u_exact = similar(u);  u̇_exact = similar(u)
    initialize3d!(u_exact, u̇_exact, x, y, z, t1; A, kx, ky, kz, ω)
    final = sol(t1)
    n = N^3 * M^3
    u_num = reshape(final[n+1:2n], N, N, N, M, M, M)
    err = maximum(abs, u_num - u_exact)
    @info "evolve3d done" t1 max_u_error=err

    return (; t0, t1, x, y, z, sol)
end

function evolve(x0, x1, M, N)
    elem = make_element(Float64, N)
    ops = make_operators(elem)
    dom = make_domain(Float64, M, 0, 1)
    x = [x + dom.h * a for a in elem.xs, x in dom.xs]
    dx = dom.h * elem.h

    u = similar(x)
    u̇ = similar(x)

    t0 = 0.0
    t1 = 1
    dt = 1//2 * dx

    A = 1
    k = 2π
    ω = sqrt(k^2)
    initialize!(u, u̇, x, t0; A,k,ω)

    bL = 0
    bR = 0
    τ = 3//2 * (N-1)^2       # dimension 1/length, scales with (N-1)^2

    # ü = similar(u)
    # rhs!(ü, u, u̇, bL, bR; dom, ops, τ)

    # Note u, u̇ are switched
    f!(ü, u̇, u, p, t) = rhs!(ü, u, u̇, bL, bR; dom, ops, τ)
    prob = SecondOrderODEProblem(f!, u̇, u, (t0, t1))

    # solver = McAte5()           # 5th order
    solver = KahanLi8()           # 8th order
    sol = solve(prob, solver; dt)

    return (;t0, t1, x, sol)
end

end
