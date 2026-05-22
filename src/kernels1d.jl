# 1D kernels: per-element Laplacian + SAT, global RHS, initialisation, and
# the diagnostic global-Laplacian assembler used by tests.

################################################################################
# Initialisation

function initialize!(u::AbstractVector, u̇::AbstractVector, x::AbstractVector, t;
                     A, k, ω)
    u .=  A   * sin.(k*x) * cos(ω*t)
    u̇ .= -A*ω * sin.(k*x) * sin(ω*t)
    return u, u̇
end

function initialize!(u::AbstractMatrix, u̇::AbstractMatrix, x::AbstractMatrix, t;
                     A, k, ω)
    M = size(u, 2)
    @assert size(u̇, 2) == size(x, 2) == M
    for m in 1:M
        initialize!(view(u, :, m), view(u̇, :, m), view(x, :, m), t; A, k, ω)
    end
    return u, u̇
end

################################################################################
# Per-element 1D Laplacian + SAT

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

################################################################################
# 1D global RHS

# `u`, `u̇`, `ü` are (N, M) matrices: row = local GLL node, column = element.
# Boundary data `bL`, `bR` are scalars (homogeneous-ish outer Dirichlet).
function rhs!(ü::AbstractMatrix, u::AbstractMatrix, u̇::AbstractMatrix, bL, bR;
              dom, ops::SBPOps{N,T}, τ) where {N, T}
    M = size(ü, 2)
    @assert size(u, 2) == size(u̇, 2) == M
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

################################################################################
# Diagnostic: assemble the global L_SAT matrix for `M` elements coupled
# DG-style. Used by the test suite to verify symmetry / null-space /
# spectrum properties; not on any simulation hot path.

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
