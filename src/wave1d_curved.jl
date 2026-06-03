# 1D scalar wave on a periodic 1+1 background metric with α = 1 and
# space- and time-varying shift β(t, x) and spatial metric γ_xx(t, x).
# State: (Φ, Π) with the *densitised* momentum
#
#     Π := √γ_xx · (∂_t Φ − β · ∂_x Φ).
#
# The covariant wave equation ∂_μ(√|g| g^{μν} ∂_ν Φ) = 0 reduces to
# the flux-conservative first-order system
#
#     ∂_t Φ = β · ∂_x Φ + Π / √γ_xx
#     ∂_t Π = ∂_x ( (1/√γ_xx) · ∂_x Φ + β · Π ).
#
# The single ∂_x of the combined flux automatically supplies both the
# `β · ∂_x Π` advection term and the `(∂_x β) · Π` source term that a
# non-conservative ADM form would have to add by hand. For γ_xx ≡ 1,
# β = const this reduces to the textbook constant-shift wave equation
# in first-order form.
#
# (We use CAPITAL Greek `Φ` and `Π` rather than lowercase `π`, because
# Julia's `π = 3.14…` constant is the lowercase letter — binding it to
# a matrix shadows the constant and produces confusing errors in any
# closure that uses `π` in arithmetic.)
#
# Discretisation: ONE consistent `D` operator everywhere — reference
# SBP-G + centred-flux SAT at every element interface (including the
# periodic wrap), applied at every SBP-G stage. With this exact choice
# of SAT coefficient the assembled operator `H · D` is exactly skew
# (`H·D + (H·D)^T = 0`, verified empirically).
#
# RK4 + skew operator → pure imaginary eigenvalues + RK4 stability disk
# tangent to the imaginary axis. Marginally stable, slow polynomial
# growth on noise. Kreiss-Oliger artificial dissipation patches this:
#     u̇ += ε · h^{2p+1} · D^{2p+2} · u.
# For p=2 (8th-order dissipation): `+ε · h⁵ · D⁶ · u`. Since `D` is
# skew, `D⁶` has eigenvalues `−μ⁶ ≤ 0`, so this term is dissipative.
# Smooth solutions: contribution scales as `ε · h⁵ · k⁶`, which is
# `O(h^{2p−1}) → 0` under refinement; KO doesn't degrade formal
# accuracy.

using StaticArrays

"""
    make_periodic_domain_1d(::Type{T}, M::Int, x0, x1) → NamedTuple

1D periodic mesh: `M` elements covering `[x0, x1]` with the last
element's right face wrapping to the first element's left face.
Returned NamedTuple matches the shape of `make_domain` plus a
`periodic = true` tag.
"""
function make_periodic_domain_1d(::Type{T}, M::Int, x0::Real, x1::Real) where {T}
    @assert M ≥ 1
    h  = (T(x1) - T(x0)) / M
    xs = T[T(x0) + h * (m - 1) for m in 1:M]
    return (; M, x0 = T(x0), x1 = T(x1), h, xs, periodic = true)
end

# Apply the consistent `D` operator (SBP-G + CF-SAT, periodic wrap)
# to a state matrix `u :: (N, M)`, writing into `Du`. The N=size(u,1)
# value is wrapped in `Val` so the inner kernel specialises on it and
# `SMatrix{N,N,T}(ops.G)` becomes a compile-time-sized stack object.
function _apply_D_1d!(Du::AbstractMatrix{T}, u::AbstractMatrix{T};
                       dom, ops) where {T}
    _apply_D_1d_impl!(Du, u, Val(size(u, 1)); dom, ops)
    return Du
end

@inline function _apply_D_1d_impl!(Du::AbstractMatrix{T},
                                     u::AbstractMatrix{T},
                                     ::Val{N}; dom, ops) where {N, T}
    M = size(u, 2)
    @assert dom.M == M
    @assert dom.periodic === true
    inv_h = one(T) / dom.h
    G = SMatrix{N, N, T}(ops.G)
    H_diag = SVector{N, T}(ntuple(i -> T(ops.H[i, i]), Val(N)))
    @inbounds for m in 1:M, i in 1:N
        s = zero(T)
        for p in 1:N
            s += G[i, p] * u[p, m]
        end
        Du[i, m] = s * inv_h
    end
    # Centred-flux SAT. The SBP property `H·G = Q + ½(e_N e_N^T −
    # e_1 e_1^T)` leaves a `+½` boundary term at (N, N) (diagonal of
    # H·G) on each element. For the assembled operator `H·D` to be
    # skew (no diagonal entries), the SAT subtracts that `+½` via
    # `Du[N, m] += -½/H_N · u[N, m] + ½/H_N · u_nbr` (and symmetric at
    # i=1). Combined: a single difference term with coefficient
    # `1/(2 H_face)` per face, scaled by `1/h` for the physical
    # derivative.
    c1 = inv_h / (T(2) * H_diag[1])
    cN = inv_h / (T(2) * H_diag[N])
    @inbounds for m in 1:M
        mL = mod1(m - 1, M)
        mR = mod1(m + 1, M)
        Du[1, m] += c1 * (u[1, m] - u[N, mL])
        Du[N, m] += cN * (u[1, mR] - u[N, m])
    end
    return nothing
end

"""
    wave1d_curved1d_rhs!(Φ̇, Π̇, Φ, Π; dom, ops, β, sqrtγ, inv_sqrtγ,
                          ε_KO = 0.1) → (Φ̇, Π̇)

1D scalar wave RHS on a periodic 1+1 background with `α = 1` and
space- and time-varying shift `β(t, x)` and spatial metric
`γ_{xx}(t, x)`. Densitised-momentum convention
`Π := √γ_{xx} · (∂_t Φ − β · ∂_x Φ)`. See the file-level comment for
the derivation; the kernel evolves

    ∂_t Φ = β · ∂_x Φ + Π / √γ_{xx}
    ∂_t Π = ∂_x ( (1/√γ_{xx}) · ∂_x Φ + β · Π ).

Inputs `β`, `sqrtγ`, `inv_sqrtγ` are `(N, M)` matrices evaluated at
the current RK4 stage time. `ε_KO` is the Kreiss-Oliger coefficient
(`0.1` is the standard NR default; set `0.0` to disable).
"""
function wave1d_curved1d_rhs!(Φ̇::AbstractMatrix{T},
                                Π̇::AbstractMatrix{T},
                                Φ::AbstractMatrix{T},
                                Π::AbstractMatrix{T};
                                dom, ops,
                                β::AbstractMatrix{T},
                                sqrtγ::AbstractMatrix{T},
                                inv_sqrtγ::AbstractMatrix{T},
                                ε_KO::Real = 0.1) where {T}
    N, M = size(Φ)
    @assert size(Π) == size(Φ̇) == size(Π̇) == (N, M)
    @assert size(β) == size(sqrtγ) == size(inv_sqrtγ) == (N, M)
    εT = T(ε_KO)
    h5 = dom.h^5

    DΦ    = Matrix{T}(undef, N, M)
    flux  = Matrix{T}(undef, N, M)
    Dflux = Matrix{T}(undef, N, M)
    _apply_D_1d!(DΦ, Φ; dom, ops)
    @inbounds for idx in eachindex(Φ)
        flux[idx] = inv_sqrtγ[idx] * DΦ[idx] + β[idx] * Π[idx]
    end
    _apply_D_1d!(Dflux, flux; dom, ops)

    # KO chain: ε · h⁵ · D⁶ applied separately to Φ and Π. Ping-pong
    # between two scratch buffers; no per-call allocations beyond the
    # constant set declared above.
    KOΦ_buf::Matrix{T} = Dflux  # placeholder; reset below
    KOΠ_buf::Matrix{T} = Dflux
    if εT != 0
        DΠ = Matrix{T}(undef, N, M)
        s1 = Matrix{T}(undef, N, M)
        s2 = Matrix{T}(undef, N, M)
        # D⁶Φ
        _apply_D_1d!(DΠ, DΦ; dom, ops)   # D²Φ (reuse DΠ as scratch)
        _apply_D_1d!(s1, DΠ; dom, ops)   # D³Φ
        _apply_D_1d!(s2, s1; dom, ops)   # D⁴Φ
        _apply_D_1d!(s1, s2; dom, ops)   # D⁵Φ
        _apply_D_1d!(s2, s1; dom, ops)   # D⁶Φ
        KOΦ_buf = s2
        # D⁶Π
        _apply_D_1d!(DΠ, Π;  dom, ops)   # D¹Π
        _apply_D_1d!(s1, DΠ; dom, ops)   # D²Π
        _apply_D_1d!(DΠ, s1; dom, ops)   # D³Π
        _apply_D_1d!(s1, DΠ; dom, ops)   # D⁴Π
        _apply_D_1d!(DΠ, s1; dom, ops)   # D⁵Π
        _apply_D_1d!(s1, DΠ; dom, ops)   # D⁶Π
        KOΠ_buf = s1
    end

    @inbounds for idx in eachindex(Φ)
        Φ̇[idx] = β[idx] * DΦ[idx] + Π[idx] * inv_sqrtγ[idx]
        Π̇[idx] = Dflux[idx]
        if εT != 0
            Φ̇[idx] += εT * h5 * KOΦ_buf[idx]
            Π̇[idx] += εT * h5 * KOΠ_buf[idx]
        end
    end
    return Φ̇, Π̇
end
