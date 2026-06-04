# Boundary conditions for the 1D ADM scalar wave (`wave1d_curved_rhs!`).
#
# Design goal: the package is a testbed for the Einstein equations, so
# the *radiative* boundary conditions deliberately avoid characteristic
# **eigenvector** projection (which is cheap here but expensive and
# gauge-dependent for multi-field GR systems). Only characteristic
# **speeds** (eigenvalues) are used — for inflow/outflow classification
# and for the penalty magnitude — which is unavoidable for any open
# boundary and far simpler than eigenvectors.
#
# Characteristic speeds. The principal system in (∂_xΦ, Π) has flux
# matrix A = [β a; a β] (a = α/√γ) with propagation speeds (dx/dt)
# s_R = a − β (rightward) and s_L = −a − β (leftward). A mode is
# *outgoing* at a face with outward normal n̂ iff s·n̂ > 0. These
# speeds classify a face:
#   * SUBLUMINAL (|β| < a; one mode in, one out):  radiative
#     (Sommerfeld) or Dirichlet data injection — see below.
#   * OUTFLOW (superluminal, both modes leave):    excision — no
#     boundary term; the one-sided `apply_D!` rows are already correct.
#   * INFLOW (superluminal, both modes enter):     full-state
#     Dirichlet — both Φ and Π pinned to data (no characteristics).
#   * SONIC (|β| ≈ a at the face):                 error; a vanishing
#     speed leaves a mode undetermined.
#
# Field-radiation SAT (subluminal faces). Rather than projecting onto
# the eigenvector ∂_xΦ ∓ Π, impose the scalar radiation condition on
# the FIELD,
#     ∂_tΦ + a·n̂·∂_xΦ = (data rate),
# rewritten with the evolution equation ∂_tΦ = β∂_xΦ + aΠ and divided
# by a into the normalised residual
#     r := Π + (n̂ + β/a)·∂_xΦ.
# At β = 0, r is exactly the incoming characteristic, so the penalty
# coincides with the textbook Sommerfeld SAT; for β ≠ 0 it differs by
# O(β)·∂_xΦ — a perturbation of the proven operator, stable for the
# small shift (|β| ≲ 0.1) radiative BCs are intended for. This is the
# NR-standard "apply ∂_t f + ∂_r f = 0 to each field" outer condition,
# and it ports field-by-field to Einstein with no eigendecomposition.
# Sommerfeld drives r → 0 (absorbing); Dirichlet drives r → its value
# on the boundary data (incoming wave injected, outgoing wave free).
#
# Energy. For E = ½∫[(Π/√γ)² + (∂_xΦ)²/γ]√γ dx the boundary terms are
# dE/dt = −¼ Σ_faces Σ_modes (s·n̂)·u² (up to the a/√γ weights):
# outgoing modes drain energy, the radiative penalty controls the
# ingoing injection. The penalties act on Π̇ at the single boundary
# node with strength σ·|s_in|/Hphys_face, |s_in| = a + n̂·β; σ = 1 was
# confirmed by the dense-operator spectrum tests (max Re(λ) ≤ round-off
# for every admissible subluminal configuration; σ = 1/2 is marginally
# unstable because the one-sided bulk operator leaves the full boundary
# flux for the penalty to cancel).
#
# Known limitation (genuine physics, not a SAT defect): with strongly
# space-varying *superluminal* β on an open domain the operator has
# eigenvalues with Re(λ) > 0 approaching the continuum bound
# max|∂_xβ| from below (compression amplification; present already
# with pure excision and no penalties, and absent on periodic meshes
# where modes recirculate through the exactly-skew operator). KO does
# not help — the growing mode is smooth. The tests assert
# max Re(λ) ≤ max|∂_xβ| for this regime.

# Boundary-condition kinds (isbits Ints — GPU-passable).
const BC_DIRICHLET      = 1   # subluminal: ingoing characteristic from data
const BC_SOMMERFELD     = 2   # subluminal: ingoing characteristic = 0
const BC_EXCISION       = 3   # superluminal outflow: no boundary term
const BC_FULL_DIRICHLET = 4   # superluminal inflow: pin Φ and Π to data

const _BC_KIND_FROM_SYMBOL = Dict(
    :dirichlet      => BC_DIRICHLET,
    :sommerfeld     => BC_SOMMERFELD,
    :excision       => BC_EXCISION,
    :full_dirichlet => BC_FULL_DIRICHLET,
)

bc1d_kind(s::Symbol) = get(_BC_KIND_FROM_SYMBOL, s) do
    throw(ArgumentError("unknown 1D boundary condition $s " *
                        "(expected :dirichlet, :sommerfeld, :excision, " *
                        ":full_dirichlet)"))
end

# Characteristic classes of a boundary face.
const FACE_SUBLUMINAL = 1
const FACE_OUTFLOW    = 2   # all characteristics leave the domain
const FACE_INFLOW     = 3   # all characteristics enter the domain
const FACE_SONIC      = 4

_face_class_name(c) = c == FACE_SUBLUMINAL ? "subluminal" :
                      c == FACE_OUTFLOW    ? "superluminal outflow" :
                      c == FACE_INFLOW     ? "superluminal inflow" :
                                             "sonic"

"""
    classify_face1d(a, β, n̂; sonic_tol = eps(typeof(a))^(1//4)) → Int

Characteristic class of a boundary face from the local `a = α/√γ` and
shift `β`, with outward normal `n̂ = ±1`. Returns one of
`FACE_SUBLUMINAL`, `FACE_OUTFLOW`, `FACE_INFLOW`, `FACE_SONIC`
(a characteristic speed within `sonic_tol·a` of zero).
"""
function classify_face1d(a::T, β::T, n̂::Integer;
                         sonic_tol = eps(T)^(1//4)) where {T}
    s_R = a - β            # propagation speed of u_R = ∂_xΦ − Π
    s_L = -a - β           # propagation speed of u_L = ∂_xΦ + Π
    tol = T(sonic_tol) * a
    (abs(s_R) ≤ tol || abs(s_L) ≤ tol) && return FACE_SONIC
    out_R = s_R * n̂ > 0
    out_L = s_L * n̂ > 0
    out_R && out_L   && return FACE_OUTFLOW
    !out_R && !out_L && return FACE_INFLOW
    return FACE_SUBLUMINAL
end

"""
    validate_bc1d(class::Int, kind::Int, face::AbstractString)

Check that boundary-condition `kind` is admissible for a face of
characteristic `class`; throw a descriptive `ArgumentError` otherwise.
The rules: subluminal → Dirichlet or Sommerfeld; superluminal
outflow → excision; superluminal inflow → full-state Dirichlet;
sonic → always an error.
"""
function validate_bc1d(class::Int, kind::Int, face::AbstractString)
    class == FACE_SONIC &&
        throw(ArgumentError("$face boundary face is at a sonic point " *
            "(|β| ≈ α/√γ): a characteristic speed vanishes there and " *
            "no boundary condition is well-posed; move the boundary " *
            "or change the background"))
    ok = (class == FACE_SUBLUMINAL && (kind == BC_DIRICHLET ||
                                       kind == BC_SOMMERFELD)) ||
         (class == FACE_OUTFLOW    && kind == BC_EXCISION) ||
         (class == FACE_INFLOW     && kind == BC_FULL_DIRICHLET)
    ok || throw(ArgumentError("$face boundary face is " *
        "$(_face_class_name(class)); admissible boundary conditions: " *
        (class == FACE_SUBLUMINAL ? ":dirichlet or :sommerfeld" :
         class == FACE_OUTFLOW    ? ":excision (no condition may be imposed " *
                                    "where all characteristics leave)" :
                                    ":full_dirichlet (all characteristics " *
                                    "enter; the full state must be given)")))
    return nothing
end

"""
    make_bc1d(kindL, kindR; g1L = 0, g2L = 0, g1R = 0, g2R = 0, σ = 1/2)

Assemble the scalar boundary-condition bundle consumed by
[`wave1d_curved_rhs!`](@ref) (kwarg `bc1d`). `kindL`/`kindR` are
`BC_*` codes (or Symbols) for the −x / +x faces. Data slots per face:

* `BC_SOMMERFELD`:     radiative (absorbing). `g1` = 0 drives the
                       field-radiation residual `r = Π + (n̂+β/a)∂_xΦ`
                       to zero — no incoming wave. Characteristic-free
                       (no eigenvector projection); valid for small
                       shift (|β| ≲ 0.1).
* `BC_DIRICHLET`:      data injection through the *same* field-
                       radiation operator: `g1` = `r` evaluated on the
                       boundary data (e.g. an exact solution), so the
                       prescribed incoming wave enters while outgoing
                       waves leave. `g2` unused. (Only differs from
                       `BC_SOMMERFELD` by the nonzero target `g1`.)
* `BC_EXCISION`:       no data.
* `BC_FULL_DIRICHLET`: `g1` = Φ data, `g2` = Π data (superluminal
                       inflow — both modes enter, so the full state is
                       pinned; no characteristics).

All entries are plain scalars — assemble a fresh bundle at every
integrator stage time. σ is the penalty strength; σ = 1 (full upwind
weight) makes the subluminal operators exactly non-growing (spectrum
at round-off) at β = 0, while σ = 1/2 is marginally unstable — the
one-sided bulk operator leaves the *full* boundary flux for the
penalty to cancel. Verified by the dense-operator spectrum tests.
"""
make_bc1d(kindL, kindR; g1L = 0, g2L = 0, g1R = 0, g2R = 0, σ = 1) =
    (; kindL = kindL isa Symbol ? bc1d_kind(kindL) : Int(kindL),
       kindR = kindR isa Symbol ? bc1d_kind(kindR) : Int(kindR),
       g1L, g2L, g1R, g2R, σ)

# Per-node boundary penalty, shared by the CPU and GPU paths.
# Returns the (Φ̇, Π̇) increments for one boundary node given the local
# state (Φv, Πv, DΦv), coefficients (av, βv), inverse face mass, the
# outward normal, the BC kind, and the two data scalars.
@inline function _bc1d_increments(kind::Int, Φv::T, Πv::T, DΦv::T,
                                  av::T, βv::T, invHf::T, n̂::Int,
                                  g1::T, g2::T, σ::T) where {T}
    kind == BC_EXCISION && return zero(T), zero(T)
    s_R = av - βv
    s_L = -av - βv
    if kind == BC_FULL_DIRICHLET
        τ = σ * (abs(s_R) + abs(s_L)) * invHf
        return -τ * (Φv - g1), -τ * (Πv - g2)
    end
    # Subluminal faces (Dirichlet / Sommerfeld): field-radiation SAT.
    # Impose the scalar radiation condition ∂_tΦ + a·n̂·∂_xΦ = 0 on the
    # FIELD (via ∂_tΦ = β∂_xΦ + aΠ), not on a characteristic
    # eigenvector — this is the form that ports to multi-field systems
    # (Einstein) where eigenvector projection is expensive. Normalised
    # residual r = Π + (n̂ + β/a)·∂_xΦ; at β = 0 this is exactly the
    # incoming characteristic, and for β ≠ 0 it differs by O(β)·∂_xΦ
    # (stable for |β| ≲ 0.1, the small-shift regime radiative BCs are
    # used in). The penalty drives r → g1, where g1 is the residual's
    # value on the boundary data (0 for Sommerfeld = absorbing). |s_in|
    # = a + n̂·β is the incoming-mode speed (an eigenvalue, not an
    # eigenvector); κ scales the penalty just as in the β = 0 case.
    r = Πv + (n̂ + βv / av) * DΦv
    s_in = av + n̂ * βv
    κ = σ * abs(s_in) * invHf
    return zero(T), -κ * (r - g1)
end

@kernel function _bc1d_kernel!(Φ̇, Π̇, @Const(Φ), @Const(Π), @Const(DΦ),
                               @Const(a), @Const(β), @Const(Hphys),
                               kindL::Int, kindR::Int,
                               g1L::T, g2L::T, g1R::T, g2R::T,
                               σ::T, ::Val{N}) where {T, N}
    side = @index(Global, Linear)
    Ne = size(Φ, 2)
    @inbounds if side == 1
        dΦ̇, dΠ̇ = _bc1d_increments(kindL, Φ[1, 1], Π[1, 1], DΦ[1, 1],
                                  a[1, 1], β[1, 1], inv(Hphys[1, 1]),
                                  -1, g1L, g2L, σ)
        Φ̇[1, 1] += dΦ̇
        Π̇[1, 1] += dΠ̇
    else
        dΦ̇, dΠ̇ = _bc1d_increments(kindR, Φ[N, Ne], Π[N, Ne], DΦ[N, Ne],
                                  a[N, Ne], β[N, Ne], inv(Hphys[N, Ne]),
                                  +1, g1R, g2R, σ)
        Φ̇[N, Ne] += dΦ̇
        Π̇[N, Ne] += dΠ̇
    end
end

# Boundary post-pass: apply the SAT penalties at the two outer faces.
# Runs after the bulk + KO passes of `wave1d_curved_rhs!`; reads the
# already-computed `DΦ` from the workspace. No-op entries use
# BC_EXCISION. Requires a non-periodic mesh (the −x face of element 1
# and the +x face of element Ne tagged as boundary).
function _apply_bc1d!(Φ̇::AbstractMatrix{T}, Π̇::AbstractMatrix{T},
                      Φ::AbstractMatrix{T}, Π::AbstractMatrix{T},
                      DΦ::AbstractMatrix{T},
                      a::AbstractMatrix{T}, β::AbstractMatrix{T};
                      geom::MeshGeometry{1, T, N}, bc1d) where {T, N}
    Ne = geom.Ne
    backend = get_backend(Φ̇)
    kindL, kindR = Int(bc1d.kindL), Int(bc1d.kindR)
    g1L, g2L = T(bc1d.g1L), T(bc1d.g2L)
    g1R, g2R = T(bc1d.g1R), T(bc1d.g2R)
    σ = T(bc1d.σ)
    if backend isa KernelAbstractions.CPU
        @inbounds begin
            dΦ̇, dΠ̇ = _bc1d_increments(kindL, Φ[1, 1], Π[1, 1], DΦ[1, 1],
                                      a[1, 1], β[1, 1],
                                      inv(geom.Hphys[1, 1]), -1,
                                      g1L, g2L, σ)
            Φ̇[1, 1] += dΦ̇
            Π̇[1, 1] += dΠ̇
            dΦ̇, dΠ̇ = _bc1d_increments(kindR, Φ[N, Ne], Π[N, Ne], DΦ[N, Ne],
                                      a[N, Ne], β[N, Ne],
                                      inv(geom.Hphys[N, Ne]), +1,
                                      g1R, g2R, σ)
            Φ̇[N, Ne] += dΦ̇
            Π̇[N, Ne] += dΠ̇
        end
    else
        _bc1d_kernel!(backend, 2)(Φ̇, Π̇, Φ, Π, DΦ, a, β, geom.Hphys,
                                  kindL, kindR, g1L, g2L, g1R, g2R,
                                  σ, Val(N); ndrange = 2)
    end
    return nothing
end
