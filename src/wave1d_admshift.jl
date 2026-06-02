# 1D scalar wave on a 1+1 background with shift β, ADM form.
# State: (Φ, Π) with Π := ∂_t Φ − β ∂_x Φ.
#
#     ∂_t Φ = Π + β · D Φ
#     ∂_t Π = D · (D Φ) + β · D Π
#
# (We use CAPITAL Greek `Φ` and `Π` rather than lowercase `π`,
# because Julia's `π = 3.14…` constant is the lowercase letter —
# binding it to a matrix shadows the constant and produces
# confusing errors in any closure that uses `π` in arithmetic.)
#
# Discretisation: ONE consistent `D` operator everywhere — reference
# SBP-G + centred-flux SAT at every element interface (including the
# periodic wrap), applied at every SBP-G stage. The Laplacian is the
# composition `D · D` (each application gets its own SAT pass).
#
# Why "consistent D": the discrete energy identity
#     <Π, D²Φ>_H = −<D Π, D Φ>_H
# only holds when the same `D` appears on both sides. Mixing raw
# bulk SBP-G with CF-SAT'd SBP-G (as the prior failed attempt did)
# breaks this and lets noise grow exponentially. With one consistent
# D the discrete H-norm energy
#     E = ½ <Π, Π>_H + ½ <D Φ, D Φ>_H
# is exactly conserved by the spatial operator for *constant* β:
#     dE/dt = <Π, D²Φ + β D Π>_H + <D Φ, D(Π + β D Φ)>_H
#           = <Π, D²Φ>_H + β <Π, D Π>_H + <D Φ, D Π>_H
#                                       + β <D Φ, D²Φ>_H
#           = -<D Π, D Φ>_H + 0 + <D Φ, D Π>_H + 0
#           = 0                                 (skewness of D in H)
# For variable β(x) the identity picks up an extra source
# `−½ ∫ β'(x) Π² dx` that breaks exact conservation; the operator
# is then no longer skew and the spectrum can develop real parts.
# That regime is the subject of the spectral analysis in
# `bin/admshift_spectrum.jl`.
#
# RK4 + skew operator → pure imaginary eigenvalues + RK4 stability
# disk tangent to the imaginary axis. Marginally stable, slow
# polynomial growth on noise. Kreiss-Oliger artificial dissipation
# patches this:
#     u̇ += ε · h^{2p+1} · D^{2p+2} · u
# For p=2 (8th-order dissipation): `+ε · h⁵ · D⁶ · u`. Since `D` is
# skew, `D⁶` has eigenvalues `−μ⁶ ≤ 0`, so this term is dissipative.
# Smooth solutions: contribution scales as `ε · h⁵ · k⁶` which is
# `O(h^{2p−1}) → 0` under refinement; KO doesn't degrade formal
# accuracy.

using StaticArrays

# Apply the consistent `D` operator (SBP-G + CF-SAT, periodic wrap)
# to a state matrix `u :: (N, M)`. Returns a fresh `(N, M)` matrix
# `Du`. Stateless; allocates a fresh output.
function _apply_D_1d(u::AbstractMatrix{T};
                      dom, ops) where {T}
    N, M = size(u)
    @assert dom.M == M
    @assert dom.periodic === true
    inv_h = one(T) / dom.h
    G = SMatrix{N, N, T}(ops.G)
    H_diag = SVector{N, T}(ntuple(i -> T(ops.H[i, i]), Val(N)))
    Du = Matrix{T}(undef, N, M)
    @inbounds for m in 1:M, i in 1:N
        s = zero(T)
        for p in 1:N
            s += G[i, p] * u[p, m]
        end
        Du[i, m] = s * inv_h
    end
    # Centred-flux SAT: the SBP property `H·G = Q + ½(e_N e_N^T −
    # e_1 e_1^T)` leaves a `+½` boundary term at (N, N) (diagonal
    # of H·G) on each element. For the assembled operator `H·D` to
    # be skew (no diagonal entries), the SAT subtracts that `+½`
    # via `Du[N, m] += -½/H_N · u[N, m] + ½/H_N · u_nbr` (and
    # symmetric at i=1). Combined: a single difference term with
    # coefficient `1/(2 H_face)` per face, scaled by `1/h` for the
    # physical derivative. With this exact factor, `H·D + (H·D)^T = 0`
    # (verified empirically).
    @inbounds for m in 1:M
        mL = mod1(m - 1, M)
        mR = mod1(m + 1, M)
        c1 = inv_h / (T(2) * H_diag[1])
        cN = inv_h / (T(2) * H_diag[N])
        Du[1, m] += c1 * (u[1, m] - u[N, mL])
        Du[N, m] += cN * (u[1, mR] - u[N, m])
    end
    return Du
end

"""
    wave1d_admshift_rhs!(Φ̇, Π̇, Φ, Π; dom, ops, β,
                          ε_KO = 0.1) → (Φ̇, Π̇)

ADM-form scalar-wave RHS on a 1D periodic mesh, with Kreiss-Oliger
artificial dissipation. See file-level comment for the equation,
consistent-D rule, and KO formula.

`β` can be either:
- `<: Real` — constant shift, applied uniformly.
- `AbstractMatrix{T}` of shape `(N, M)` — per-node shift `β(x)`,
  sampled at every GLL collocation node. No new SAT logic; the
  advection terms become pointwise products `β[i,m] · D Φ[i,m]`.

`ε_KO` is the KO coefficient (`0.1` is the standard NR default;
set `0.0` to disable).
"""
function wave1d_admshift_rhs!(Φ̇::AbstractMatrix{T},
                                Π̇::AbstractMatrix{T},
                                Φ::AbstractMatrix{T},
                                Π::AbstractMatrix{T};
                                dom, ops,
                                β::Union{Real, AbstractMatrix},
                                ε_KO::Real = 0.1) where {T}
    N, M = size(Φ)
    @assert size(Π) == size(Φ̇) == size(Π̇) == (N, M)
    if β isa AbstractMatrix
        @assert size(β) == (N, M) "β matrix must have shape (N, M) = ($N, $M)"
    end
    εT  = T(ε_KO)
    h5  = dom.h^5

    # Consistent-D applied to Φ and Π (and the Laplacian = D·D).
    DΦ  = _apply_D_1d(Φ; dom, ops)
    DΠ  = _apply_D_1d(Π; dom, ops)
    D2Φ = _apply_D_1d(DΦ; dom, ops)

    # KO term D⁶: apply D four more times to each of DΦ, DΠ.
    KOΦ = D2Φ; KOΠ = DΠ
    if εT != 0
        DKO = _apply_D_1d(D2Φ; dom, ops)   # D³Φ
        DKO = _apply_D_1d(DKO; dom, ops)   # D⁴Φ
        DKO = _apply_D_1d(DKO; dom, ops)   # D⁵Φ
        KOΦ = _apply_D_1d(DKO; dom, ops)   # D⁶Φ
        DKO = _apply_D_1d(DΠ;  dom, ops)   # D²Π
        DKO = _apply_D_1d(DKO; dom, ops)   # D³Π
        DKO = _apply_D_1d(DKO; dom, ops)   # D⁴Π
        DKO = _apply_D_1d(DKO; dom, ops)   # D⁵Π
        KOΠ = _apply_D_1d(DKO; dom, ops)   # D⁶Π
    end

    if β isa Real
        βT = T(β)
        @inbounds for m in 1:M, i in 1:N
            Φ̇[i, m] = Π[i, m]   + βT * DΦ[i, m]
            Π̇[i, m] = D2Φ[i, m] + βT * DΠ[i, m]
            if εT != 0
                Φ̇[i, m] += εT * h5 * KOΦ[i, m]
                Π̇[i, m] += εT * h5 * KOΠ[i, m]
            end
        end
    else
        β_arr::AbstractMatrix{T} = β
        @inbounds for m in 1:M, i in 1:N
            βij = β_arr[i, m]
            Φ̇[i, m] = Π[i, m]   + βij * DΦ[i, m]
            Π̇[i, m] = D2Φ[i, m] + βij * DΠ[i, m]
            if εT != 0
                Φ̇[i, m] += εT * h5 * KOΦ[i, m]
                Π̇[i, m] += εT * h5 * KOΠ[i, m]
            end
        end
    end
    return Φ̇, Π̇
end
