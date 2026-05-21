module WaveToySecondOrder

using LinearAlgebra
using OrdinaryDiffEqSymplecticRK

################################################################################

function make_element(::Type{T}, N::Int) where {T}
    ns = 0:N-1

    # Calculate operators with rational numbers for improved accuracy
    x0 = T(0//1)
    x1 = T(1//1)
    # N points, "vertex centred"
    h = (x1-x0) / (N-1)
    xs = x0 .+ h * ns

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

function make_operators(dom)
    N,ns,x0,x1,h,xs = dom

    # Boundary operator
    # Straightforward.
    B = Diagonal([ifelse(n==ns[begin], -1, ifelse(n==ns[end], +1, 0)) for n in ns])

    # Gradient
    # Ensure it is exact for polynomials up to order N-1.
    G = let
        # G x^p = p x^(p-1)
        lhs = [x^p for x in xs, p in 0:N-1]
        rhs = [p==0 ? 0 : p * x^(p-1) for x in xs, p in 0:N-1]
        # G[i,j] * lhs[j,k] = rhs[i,k]
        # Solve linear system
        G = rhs / lhs
        # Test
        if eltype(G) <: Rational
            @assert G * lhs == rhs
            @assert all(G * xs.^0 .== 0)
            @assert all(G * xs.^p == p * xs.^(p-1) for p in 1:N-1)
        end
        G
    end

    # Norm
    # Ensure it is exact for polynomials up to order N-1.
    H = let
        # x^p H x^q = [1/(p+q+1) x^(p+q+1)]₀¹
        u = [x^p for x in xs, p in 0:N-1]
        v = [x^q for x in xs, q in 0:N-1]
        r = [1//(p+q+1) * (x1^(p+q+1) - x0^(p+q+1)) for p in 0:N-1, q in 0:N-1]
        # u[i,k] H[i,j] v[j,l] = r[k,l]
        # u' H v = r
        # H v = u' \ r
        # Solve linear system
        H = (u' \ r) / v
        if eltype(G) <: Rational
            @assert issymmetric(H)
            @assert all(>(0), eigvals(H))
            # Test
            @assert all(u[:,k]' * H * v[:,l] == r[k,l] for k in 1:N, l in 1:N)
        else
            H = (H + H') / 2
        end
        H
    end

    # Divergence, defined via summation by parts
    # H*D + (H*G)' = B
    D = H \ (B - (H*G)')
    # Note: This is just D = G...
    if eltype(G) <: Rational
        @assert D == G
        @assert all(D * xs.^0 .== 0)
        @assert all(D * xs.^p == p * xs.^(p-1) for p in 1:N-1)
    end

    # Laplacian (without boundary conditions)
    # H*L = (B − (H*G)') * G
    L = D * G

    Hinv = inv(H)

    # Precompute Hinv * (first/last rows of G), used by the SAT consistency
    # term `Hinv · Gᵀ · B · …` which only touches the first and last columns
    # of `Gᵀ` (i.e., the first and last rows of `G`).
    HinvG_L = Hinv * G[1, :]
    HinvG_R = Hinv * G[N, :]

    return (;B,G,H,Hinv,HinvG_L,HinvG_R,D,L)
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
# Low-level penalty update: takes already-computed boundary jumps
# `ΔuL, ΔuR, ΔGuL, ΔGuR` and accumulates the SAT contribution into `Lu`.
# Touches `Hinv`, `HinvG_L`, `HinvG_R` only — no `G` or `B` matrix work.
function add_dirichlet_penalties!(
    Lu::AbstractVector,
    ΔuL, ΔuR, ΔGuL, ΔGuR, αL, αR;
    ops, τ,
)
    Hinv, HinvG_L, HinvG_R = ops.Hinv, ops.HinvG_L, ops.HinvG_R
    N = size(Hinv, 1)

    cL = -αL * ΔuL
    cR =  αR * ΔuR
    b1 =  (1//2) * ΔGuL - τ * ΔuL
    bN = -(1//2) * ΔGuR - τ * ΔuR

    @inbounds for i in 1:N
        Lu[i] += cL * HinvG_L[i] + cR * HinvG_R[i] +
                 b1 * Hinv[i, 1] + bN * Hinv[i, N]
    end
    return Lu
end

# Convenience wrapper: caller supplies `u` and the neighbour data; we
# compute boundary jumps locally (one `G`-row dot product per face).
function apply_laplacian!(Lu::AbstractVector, u::AbstractVector,
                          gL, gR, gGL, gGR, αL, αR; ops, τ)
    G = ops.G
    N = length(u)

    mul!(Lu, ops.L, u)

    ΔuL = u[begin] - gL
    ΔuR = u[end]   - gR

    GuL = zero(eltype(u))
    GuR = zero(eltype(u))
    @inbounds for j in 1:N
        GuL += G[1, j] * u[j]
        GuR += G[N, j] * u[j]
    end
    ΔGuL = GuL - gGL
    ΔGuR = GuR - gGR

    add_dirichlet_penalties!(Lu, ΔuL, ΔuR, ΔGuL, ΔGuR, αL, αR; ops, τ)
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



# One-element functions

function initialize!(u::AbstractVector, u̇::AbstractVector, x::AbstractVector, t; A,k,ω)
    u .=  A   * sin.(k*x) * cos(ω*t)
    u̇ .= -A*ω * sin.(k*x) * sin(ω*t)
    return u, u̇
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

function rhs!(ü::AbstractMatrix, u::AbstractMatrix, u̇::AbstractMatrix, bL, bR; dom, ops, τ)
    M = size(ü, 2)
    N = size(u, 1)
    @assert size(u,2) == size(u̇,2) == M
    G, L = ops.G, ops.L

    T    = eltype(u)
    half = one(T) / 2

    # Per-element work. Each iteration touches only its own column of u and
    # the boundary slices of the immediate-neighbour columns (a one-node-wide
    # halo of values plus a `G`-row dot product to recover the neighbour's
    # boundary gradient). No global `G*u` or `L*u` — the body is suitable
    # for parallel (e.g. GPU kernel) execution over `m`.
    @inbounds for m in 1:M
        um = view(u, :, m)
        üm = view(ü, :, m)

        # Bulk Laplacian for this element.
        mul!(üm, L, um)

        # Own boundary gradients — two scalar dot products with rows of `G`.
        GuL_self = zero(T)
        GuR_self = zero(T)
        for j in 1:N
            GuL_self += G[1, j] * um[j]
            GuR_self += G[N, j] * um[j]
        end

        # Left face: at outer m=1 use mirror (ΔGu = 0); else compute the
        # neighbour's right-edge gradient `G[N,:] · u[:,m−1]` locally.
        if m == 1
            ΔuL  = um[1] - bL
            ΔGuL = zero(T)
            αL   = one(T)
        else
            gGL = zero(T)
            for j in 1:N
                gGL += G[N, j] * u[j, m-1]
            end
            ΔuL  = um[1] - u[N, m-1]
            ΔGuL = GuL_self - gGL
            αL   = half
        end

        # Right face — symmetric.
        if m == M
            ΔuR  = um[N] - bR
            ΔGuR = zero(T)
            αR   = one(T)
        else
            gGR = zero(T)
            for j in 1:N
                gGR += G[1, j] * u[j, m+1]
            end
            ΔuR  = um[N] - u[1, m+1]
            ΔGuR = GuR_self - gGR
            αR   = half
        end

        add_dirichlet_penalties!(üm, ΔuL, ΔuR, ΔGuL, ΔGuR, αL, αR; ops, τ)
    end

    ü .*= inv(dom.h^2)
    return ü
end

function evolve(x0, x1, M)
    N = 9
    elem = make_element(Float64, N)
    ops = make_operators(elem)
    dom = make_domain(Float64, M, 0, 1)
    x = [x + dom.h * a for a in elem.xs, x in dom.xs]
    dx = dom.h * elem.h

    u = similar(x)
    u̇ = similar(x)

    t0 = 0.0
    t1 = 1
    dt = 1//20 * dx

    A = 1
    k = 2π
    ω = sqrt(k^2)
    initialize!(u, u̇, x, t0; A,k,ω)

    bL = 0
    bR = 0
    # τ = 24   # for N=5
    # τ = 96   # for N=9
    τ = 3//2 * (N-1)^2       # dimension 1/length, scales with (N-1)^2

    # ü = similar(u)
    # rhs!(ü, u, u̇, bL, bR; dom, ops, τ)

    # Note u, u̇ are switched
    f!(ü, u̇, u, p, t) = rhs!(ü, u, u̇, bL, bR; dom, ops, τ)
    prob = SecondOrderODEProblem(f!, u̇, u, (t0, t1))

    # solver = McAte5()           # 5th order
    solver = KahanLi8()           # 8th order
    sol = solve(prob, solver; dt)

    return (;t0, t1, x, sol)
end

end
