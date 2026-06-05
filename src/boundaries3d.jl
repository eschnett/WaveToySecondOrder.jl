# Outer boundary conditions for the 3D ADM scalar wave
# (`wave3d_curved_rhs!`), the 3D analog of boundaries2d.jl. Faces are
# classified from the normal characteristic structure (c± = −β^n ± a_n,
# β^n = n̂·β^{dₙ}, a_n = α√γ^{dₙdₙ}) using the 1D helpers. Sommerfeld /
# Dirichlet use the characteristic-free field-radiation SAT; superluminal
# outflow → excision (no term); superluminal inflow → full-state
# Dirichlet. Axis-aligned affine hex meshes (Milestone 1); the
# physical-normal curvilinear pass is Milestone 2.

using HexSBPSAT: MeshGeometry, SBPOps
using KernelAbstractions: get_backend, @kernel, @index, @Const

# 3D face → (normal axis dₙ, outward sign n̂, along-axis node index).
@inline _face_geom3d(f, ::Val{N}) where {N} =
    (((f + 1) ÷ 2), (isodd(f) ? -1 : 1), (isodd(f) ? 1 : N))

"""
    classify_face3d(α, β1, β2, β3, gu11, gu22, gu33, n̂axis, n̂sign;
                    sonic_tol) → Int

Characteristic class (`FACE_*`) of an axis-aligned 3D boundary face with
outward normal `n̂sign·e_{n̂axis}`, from `a_n = α√(γ^{n̂axis n̂axis})` and
`β^{n̂axis}`. Delegates to [`classify_face1d`](@ref).
"""
function classify_face3d(α::T, β1::T, β2::T, β3::T, gu11::T, gu22::T, gu33::T,
                         n̂axis::Int, n̂sign::Int;
                         sonic_tol = eps(T)^(1//4)) where {T}
    gunn = n̂axis == 1 ? gu11 : n̂axis == 2 ? gu22 : gu33
    βax  = n̂axis == 1 ? β1   : n̂axis == 2 ? β2   : β3
    a_n  = α * sqrt(gunn)
    return classify_face1d(a_n, βax, n̂sign; sonic_tol)
end

"""
    make_bc3d(kinds::NTuple{6}; σ = 1, gΦ = nothing, gΠ = nothing,
              gDx = nothing, gDy = nothing, gDz = nothing, excision_tag = 0)

Boundary bundle for [`wave3d_curved_rhs!`] (kwarg `bc3d`). `kinds` gives
the `BC_*` code (or Symbol) for each of the six face directions
(−x,+x,−y,+y,−z,+z). `gΦ` is the field-radiation `:dirichlet` target;
`(gΦ, gΠ)` the `:full_dirichlet` state target. `excision_tag` (default 0)
faces get no SAT (curvilinear pass). `gDx/gDy/gDz` are reserved for the
curvilinear Dirichlet pass (Milestone 2).
"""
function make_bc3d(kinds; σ = 1, gΦ = nothing, gΠ = nothing,
                   gDx = nothing, gDy = nothing, gDz = nothing,
                   excision_tag = 0)
    codes = ntuple(i -> (kinds[i] isa Symbol ? bc1d_kind(kinds[i]) :
                         Int(kinds[i])), 6)
    return (; kinds = codes, σ, gΦ, gΠ, gDx, gDy, gDz,
            excision_tag = Int32(excision_tag))
end

# Axis-aligned affine outer-boundary pass. Reads ∂_dΦ from
# ws.DΦ1/DΦ2/DΦ3. A single KA kernel parallelised over output nodes, so a
# node touched by up to three boundary faces accumulates all
# contributions race-free in one workitem.
function _apply_bc3d!(Φ̇::AbstractArray{T,4}, Π̇::AbstractArray{T,4},
                      Φ::AbstractArray{T,4}, Π::AbstractArray{T,4},
                      coef, ws; geom::MeshGeometry{3, T, N},
                      ops::SBPOps{N, T}, bc3d) where {N, T}
    backend = get_backend(Φ̇)
    gΦ = bc3d.gΦ === nothing ? Φ : bc3d.gΦ
    gΠ = bc3d.gΠ === nothing ? Φ : bc3d.gΠ
    k = bc3d.kinds
    _bc3d_affine_kernel!(backend, (N, N, N))(
        Φ̇, Π̇, Φ, Π, ws.DΦ1, ws.DΦ2, ws.DΦ3, coef.alpha, coef.sqrtγ,
        coef.gu11, coef.gu22, coef.gu33, coef.b1, coef.b2, coef.b3,
        geom.invjac, geom.conn.bdry, ops, gΦ, gΠ,
        bc3d.gΦ === nothing, bc3d.gΠ === nothing,
        Int32(k[1]), Int32(k[2]), Int32(k[3]), Int32(k[4]), Int32(k[5]),
        Int32(k[6]), T(bc3d.σ), Val(N); ndrange = (N, N, N, geom.Ne))
    return nothing
end

@kernel function _bc3d_affine_kernel!(Fdot, Pdot, @Const(F), @Const(P),
                                      @Const(DΦ1), @Const(DΦ2), @Const(DΦ3),
                                      @Const(alpha), @Const(sqrtγ),
                                      @Const(gu11), @Const(gu22), @Const(gu33),
                                      @Const(b1), @Const(b2), @Const(b3),
                                      @Const(invjac), @Const(bdry), ops,
                                      @Const(gΦ), @Const(gΠ), noΦ, noΠ,
                                      k1, k2, k3, k4, k5, k6, σ,
                                      ::Val{N}) where {N}
    i, j, k, m = @index(Global, NTuple)
    T = eltype(Fdot)
    dF = zero(T); dP = zero(T)
    @inbounds for f in 1:6
        bdry[f, m] == 0 && continue
        kind = f == 1 ? k1 : f == 2 ? k2 : f == 3 ? k3 :
               f == 4 ? k4 : f == 5 ? k5 : k6
        kind == Int32(BC_EXCISION) && continue
        dn = (f + 1) ÷ 2
        fn = isodd(f) ? 1 : N
        on = dn == 1 ? (i == fn) : dn == 2 ? (j == fn) : (k == fn)
        on || continue
        n̂ = isodd(f) ? -one(T) : one(T)
        α = alpha[i,j,k,m]; sγ = sqrtγ[i,j,k,m]
        gunn = dn == 1 ? gu11[i,j,k,m] : dn == 2 ? gu22[i,j,k,m] : gu33[i,j,k,m]
        βax  = dn == 1 ? b1[i,j,k,m]   : dn == 2 ? b2[i,j,k,m]   : b3[i,j,k,m]
        ij   = invjac[dn, dn, i, j, k, m]
        a   = α / sγ
        a_n = α * sqrt(gunn)
        βn  = n̂ * βax
        wt  = ij / ops.H[fn, fn]
        if kind == Int32(BC_FULL_DIRICHLET)
            τ = σ * (abs(a_n - βax) + abs(a_n + βax)) * wt
            gΦv = noΦ ? zero(T) : gΦ[i,j,k,m]
            gΠv = noΠ ? zero(T) : gΠ[i,j,k,m]
            dF += -τ * (F[i,j,k,m] - gΦv)
            dP += -τ * (P[i,j,k,m] - gΠv)
        else
            DΦn = dn == 1 ? DΦ1[i,j,k,m] : dn == 2 ? DΦ2[i,j,k,m] : DΦ3[i,j,k,m]
            q   = n̂ * DΦn
            r   = P[i,j,k,m] + ((βn + a_n) / a) * q
            g   = kind == Int32(BC_SOMMERFELD) ? zero(T) :
                  (noΦ ? zero(T) : gΦ[i,j,k,m])
            s_in = a_n + βn
            dP += -σ * abs(s_in) * wt * (r - g)
        end
    end
    @inbounds Fdot[i,j,k,m] += dF
    @inbounds Pdot[i,j,k,m] += dP
end
