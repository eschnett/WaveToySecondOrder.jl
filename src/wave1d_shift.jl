# 1D scalar wave equation on a 1+1 background metric with a constant shift β.
#
# Metric: `ds² = -dt² + (dx + β dt)²` with `α = 1`, `γ_xx = 1`,
# `β = const`. The wave equation `□Φ = 0` becomes
#
#     ∂_t² u = (1 - β²) ∂_x² u + 2β ∂_x u̇        (constant β)
#
# with `u̇ ≡ ∂_t u`. For variable β(x) we additionally pick up
# `2β β'(x) ∂_x u + β'(x) u̇`; not implemented here.
#
# Characteristic speeds: `ω/k = β ± 1`. For `β < 1` they have
# opposite signs (left and right movers); for `β > 1` both have the
# same sign (the wave is dragged in the +x direction by the shift).
#
# Periodic in `x ∈ [x0, x1]`: the M-th element's right face wraps to
# the 1st element's left face.
#
# Discretisation note: we use SBP-G + centred-flux SAT applied to
# `∂_x u` and `∂_x u̇` separately at every element interface
# (including the periodic wrap), then a second SBP-G on the
# corrected `∂_x u` to get `∂_x² u`. The Mattsson–Nordström
# conservative pair would be the natural choice for the standard
# `u_tt = u_xx` case, but its derivation assumes a positive-definite
# principal symbol — for `β > 1` the coefficient `(1-β²)` of the
# `u_xx` term flips sign and the standard MN dissipation becomes
# anti-dissipative. Centred flux is the simplest symmetric choice
# that survives both regimes. It is marginally stable: high-
# frequency modes can grow slowly, so the test runs for a bounded
# time and asserts finite output rather than strict boundedness.

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

"""
    wave1d_shift_rhs!(ü, u, u̇; dom, ops, β) → ü

In-place RHS of the constant-shift scalar wave equation on a 1D
periodic mesh:

    ü = (1 - β²) ∂_x² u + 2β ∂_x u̇

Both `∂_x u` and `∂_x u̇` are computed with SBP-G + a centred-flux
SAT at every element interface (including the periodic wrap from the
M-th element's right face to the 1st element's left face). The
second `x`-derivative re-applies SBP-G to the corrected `∂_x u`
without further SAT — the centred-flux pair already enforced
continuity.

`u`, `u̇`, `ü` are `(N, M)` matrices. `dom = make_periodic_domain_1d(...)`.
`ops = make_operators(make_element(T, N))`. `β` is `<: Real`.
"""
function wave1d_shift_rhs!(ü::AbstractMatrix{T},
                             u::AbstractMatrix{T},
                             u̇::AbstractMatrix{T};
                             dom, ops, β::Real) where {T}
    N, M = size(u)
    @assert size(u̇) == size(ü) == (N, M)
    @assert dom.M == M
    @assert dom.periodic === true

    inv_h = one(T) / dom.h
    βT    = T(β)
    G     = SMatrix{N, N, T}(ops.G)
    H_inv = SVector{N, T}(ntuple(i -> one(T) / ops.H[i, i], Val(N)))

    du  = Matrix{T}(undef, N, M)   # ∂_x u
    du̇  = Matrix{T}(undef, N, M)   # ∂_x u̇

    # Stage 1 — SBP-G on each element with centred-flux SAT at the
    # left and right element faces (periodic neighbours via `mod1`).
    half_inv_h = T(1//2) * inv_h
    @inbounds for m in 1:M
        mL = mod1(m - 1, M)
        mR = mod1(m + 1, M)
        # Bulk SBP-G.
        for i in 1:N
            s_u  = zero(T); s_u̇ = zero(T)
            for p in 1:N
                s_u  += G[i, p] * u[p, m]
                s_u̇  += G[i, p] * u̇[p, m]
            end
            du[i, m]  = s_u  * inv_h
            du̇[i, m] = s_u̇ * inv_h
        end
        # Centred-flux SAT: at the left face (i = 1, f_sign = -1)
        # neighbour value lives at (i = N, mL); at the right face
        # (i = N, f_sign = +1) neighbour value lives at (i = 1, mR).
        coef1 = half_inv_h * H_inv[1]
        coefN = half_inv_h * H_inv[N]
        du[1, m]  -= coef1 * (u[N, mL]  - u[1, m])
        du̇[1, m] -= coef1 * (u̇[N, mL] - u̇[1, m])
        du[N, m]  += coefN * (u[1, mR]  - u[N, m])
        du̇[N, m] += coefN * (u̇[1, mR] - u̇[N, m])
    end

    # Stage 2 — second SBP-G applied to the corrected `du` to get
    # `∂_x² u`. No SAT.
    one_minus_β² = one(T) - βT * βT
    two_β        = T(2) * βT
    @inbounds for m in 1:M
        for i in 1:N
            ddu = zero(T)
            for p in 1:N
                ddu += G[i, p] * du[p, m]
            end
            ddu *= inv_h
            ü[i, m] = one_minus_β² * ddu + two_β * du̇[i, m]
        end
    end
    return ü
end
