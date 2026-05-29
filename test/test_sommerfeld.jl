# Tests for the Sommerfeld radiative boundary condition.
#
# These tests probe the wave-equation post-pass `_sommerfeld_pass_kernel!`
# layered on top of `apply_laplacian3d!`'s natural boundary lift.
# Mesh choice: inflated cube with `outer_bc = :sommerfeld`, the only
# combination that produces tag-7 outer faces on a spherical surface
# (where the BGT-1 correction `+ u/R` is exact for a radial source).
#
#   T1 — BC-residual probe: for the analytic outgoing pulse evaluated
#        at t = 0, evaluate `B₁u = u̇ + ∂_n u + u/R` at every Sommerfeld
#        face quadrature node. Should be tiny for BGT-1 (it's the
#        truncation error of the BC, which is exact in the radial limit).
#   T2 — Absorption: evolve the outgoing pulse to t = 1.5 with BGT-1 and
#        check that the field is bounded and the energy is substantially
#        dissipated.
#   T3 — BGT-1 vs BGT-0: BGT-1's `+u/R` spherical correction should
#        leave less residual than BGT-0's plane-wave Sommerfeld.

using WaveToySecondOrder
using OrdinaryDiffEqSymplecticRK
using StaticArrays
using LinearAlgebra
using Test

@isdefined(_progress) ||
    (_progress(msg) = (printstyled(stderr, "  • ", msg, "\n"; color = :cyan);
                       flush(stderr)))

# Walks every Sommerfeld face quadrature node and returns
#   (max |B₁u|, rms |B₁u|, num_face_nodes, max |n − r̂|)
# where B₁u = u̇ + ∂_n u + u/R is the BGT-1 residual and r̂ is the unit
# radial direction (which equals the outward normal on the outer sphere).
function _bc_residual_on_sommerfeld_faces(mesh, geom, u, u̇, R, ::Val{N}) where {N}
    T = eltype(u)
    inv_R = one(T) / R
    B1_max = zero(T)
    B1_sum_sq = zero(T)
    fc = 0
    norm_err = zero(T)
    for e in 1:mesh.Ne, f in 1:6
        geom.conn.neighbour[f, e] == 0 || continue
        geom.conn.bdry[f, e] == Int8(7) || continue
        a = (f + 1) >> 1
        fr = isodd(f) ? 1 : N
        sgn_f = isodd(f) ? -one(T) : one(T)
        sgn_c = a == 2 ? -one(T) : one(T)
        axis_p, axis_q = a == 1 ? (2, 3) : a == 2 ? (1, 3) : (1, 2)
        for q_local in 1:N, p_local in 1:N
            i, j, k = a == 1 ? (fr, p_local, q_local) :
                      a == 2 ? (p_local, fr, q_local) :
                                (p_local, q_local, fr)
            J = SMatrix{3, 3, T}(
                geom.jac[1, 1, i, j, k, e], geom.jac[2, 1, i, j, k, e], geom.jac[3, 1, i, j, k, e],
                geom.jac[1, 2, i, j, k, e], geom.jac[2, 2, i, j, k, e], geom.jac[3, 2, i, j, k, e],
                geom.jac[1, 3, i, j, k, e], geom.jac[2, 3, i, j, k, e], geom.jac[3, 3, i, j, k, e])
            tp = SVector(J[1, axis_p], J[2, axis_p], J[3, axis_p])
            tq = SVector(J[1, axis_q], J[2, axis_q], J[3, axis_q])
            sgn_out = sgn_f * sgn_c * T(geom.handedness[e])
            n_u = sgn_out * cross(tp, tq)
            JF = norm(n_u)
            n = n_u / JF
            x = SVector(geom.coords[1, i, j, k, e],
                        geom.coords[2, i, j, k, e],
                        geom.coords[3, i, j, k, e])
            norm_err = max(norm_err, norm(n - x / norm(x)))
            u_val = geom.face_trace[1, p_local, q_local, f, e]
            gx = geom.face_trace[2, p_local, q_local, f, e]
            gy = geom.face_trace[3, p_local, q_local, f, e]
            gz = geom.face_trace[4, p_local, q_local, f, e]
            Gn = n[1] * gx + n[2] * gy + n[3] * gz
            res = u̇[i, j, k, e] + Gn + inv_R * u_val
            B1_max = max(B1_max, abs(res))
            B1_sum_sq += res^2
            fc += 1
        end
    end
    return B1_max, sqrt(B1_sum_sq / max(fc, 1)), fc, norm_err
end

function _discrete_energy(u, u̇, geom, ops, τ)
    T = eltype(u)
    bdry = ntuple(_ -> zero(T), Val(6))
    Lu = similar(u)
    apply_laplacian3d!(Lu, u, bdry; geom, ops, τ)
    K = discrete_inner_product(u̇, u̇, geom, ops) / 2
    V = -discrete_inner_product(u, Lu, geom, ops) / 2
    return K + V
end

@testset "Sommerfeld BC (N=3, M=4, inflated cube)" begin

    T = Float64
    N = 3
    M = 4
    R2 = 1.0
    s0 = 0.5
    σ  = 0.15
    t_end = 1.5
    elem = make_element(T, N)
    ops  = make_operators(elem)
    τ    = T(3//2) * (N - 1)^2

    mesh = make_inflated_cube_mesh(T, T(0.1), T(0.3), T(R2), M;
                                    outer_bc = :sommerfeld)
    geom = make_geometry(mesh, elem)

    # IC: outgoing pulse, peak at r = s0, width σ.
    u  = Array{T, 4}(undef, N, N, N, mesh.Ne)
    u̇ = similar(u)
    outgoing_pulse!(u, u̇, geom.coords, zero(T); A = one(T), s0 = s0, σ = σ)

    # Populate geom.face_trace by running the operator once. This is what
    # the Sommerfeld post-pass reads inside `rhs_wave3d!`.
    ü_tmp = similar(u)
    apply_laplacian3d!(ü_tmp, u, ntuple(_ -> zero(T), Val(6)); geom, ops, τ)

    _progress("T1: BC residual small for analytic outgoing pulse")
    @testset "T1: BC residual on Sommerfeld faces (analytic pulse, t=0)" begin
        B1_max, _, fc, n_err = _bc_residual_on_sommerfeld_faces(
            mesh, geom, u, u̇, T(R2), Val(N))
        # 6·M² faces on the outer sphere with N² nodes each.
        @test fc == 6 * M^2 * N^2
        # Outward normal must be exactly radial on the outer sphere.
        @test n_err < 100 * eps(T)
        # BC residual is tiny — BGT-1 is exact for a radial outgoing wave
        # modulo the (exponentially small) ingoing-image contribution +
        # discretization error.
        @test B1_max / maximum(abs, u̇) < 0.01
    end

    _progress("T2: BGT-1 absorbs outgoing pulse")
    @testset "T2: outgoing pulse is absorbed by BGT-1 by t=$t_end" begin
        E0 = _discrete_energy(u, u̇, geom, ops, τ)
        params = Params3d(; A = one(T), k = (zero(T), zero(T), zero(T)),
                            ω = zero(T), τ = τ,
                            bdry_values = ntuple(_ -> zero(T), Val(6)),
                            sommerfeld_R = T(R2))
        dt = recommended_dt(geom, ops, τ; cfl_safety = T(1//2))
        f!(ü, u̇, u, p, t) = rhs_wave3d!(ü, u, u̇, p; geom, ops)
        prob = SecondOrderODEProblem(f!, copy(u̇), copy(u), (zero(T), T(t_end)), params)
        sol = solve(prob, VelocityVerlet(); dt,
                    save_everystep = false, save_start = false,
                    dense = false, save_end = true)
        u_end  = reshape(sol.u[end].x[2], N, N, N, mesh.Ne)
        u̇_end = reshape(sol.u[end].x[1], N, N, N, mesh.Ne)
        E_end  = _discrete_energy(u_end, u̇_end, geom, ops, τ)
        @test all(isfinite, u_end) && all(isfinite, u̇_end)
        # Field is bounded (no blowup) — initial peak was ~2.10, residual
        # should be well below 1.0 for this resolution.
        @test maximum(abs, u_end) < one(T)
        # ≥ 95% of energy has bled out through the outer sphere.
        @test E_end / E0 < T(0.05)
    end

    _progress("T3: BGT-1 absorbs more than BGT-0")
    @testset "T3: BGT-1 (R = $R2) leaves smaller residual than BGT-0 (R = ∞)" begin
        function evolve(R)
            params = Params3d(; A = one(T), k = (zero(T), zero(T), zero(T)),
                                ω = zero(T), τ = τ,
                                bdry_values = ntuple(_ -> zero(T), Val(6)),
                                sommerfeld_R = R)
            dt = recommended_dt(geom, ops, τ; cfl_safety = T(1//2))
            f!(ü, u̇, u, p, t) = rhs_wave3d!(ü, u, u̇, p; geom, ops)
            prob = SecondOrderODEProblem(f!, copy(u̇), copy(u), (zero(T), T(t_end)), params)
            sol = solve(prob, VelocityVerlet(); dt,
                        save_everystep = false, save_start = false,
                        dense = false, save_end = true)
            u_end = reshape(sol.u[end].x[2], N, N, N, mesh.Ne)
            return maximum(abs, u_end)
        end
        mx_bgt0 = evolve(T(Inf))
        mx_bgt1 = evolve(T(R2))
        # The +u/R BGT-1 correction should give ≥ 30% reduction at
        # this resolution. (Empirically ≈ 50% at N=3 M=4.)
        @test mx_bgt1 < T(0.7) * mx_bgt0
    end

end
