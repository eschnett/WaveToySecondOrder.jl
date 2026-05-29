# 1D wave-equation evolution test against the analytic solution. The
# generic operator-level identities (SBP property, build_global_laplacian
# symmetry, jump penalty) live in HexSBPSAT/test/test_kernels1d.jl.

using WaveToySecondOrder
using WaveToySecondOrder: make_element, make_domain, make_operators,
                          apply_laplacian!, initialize!
using OrdinaryDiffEqSymplecticRK
using Test

@isdefined(_progress) ||
    (_progress(msg) = (printstyled(stderr, "  • ", msg, "\n"; color = :cyan);
                       flush(stderr)))

@testset "wave1d evolution" begin

    _progress("wave evolution (1D, N=4, M=8)")
    @testset "wave evolution matches analytic solution (N=4, M=8)" begin
        # Evolve u_tt = u_xx with the analytic solution
        #   u(x,t) = sin(2π x) · cos(2π t)
        # on [0, 1] with homogeneous Dirichlet BC. The IC and target are both
        # produced by `initialize!`. Use M=8 elements of N=4 GLL nodes.
        N_e = 4
        M   = 8
        elem  = make_element(Float64, N_e)
        ops_f = make_operators(elem)
        dom   = make_domain(Float64, M, 0, 1)

        x   = [x + dom.h * a for a in elem.xs, x in dom.xs]
        dx  = dom.h * elem.h          # min spacing (GLL clusters at edges)

        Aamp = 1.0
        k    = 2π
        ω    = sqrt(k^2)              # wave speed = 1

        u  = similar(x);  u̇ = similar(x)
        initialize!(u, u̇, x, 0.0; A=Aamp, k, ω)

        bL, bR = 0.0, 0.0
        τ      = 64.0
        t0, t1 = 0.0, 1.0             # one full period

        f!(ü, u̇, u, p, t) = apply_laplacian!(ü, u, bL, bR; dom, ops=ops_f, τ)
        prob = SecondOrderODEProblem(f!, u̇, u, (t0, t1))
        sol  = solve(prob, CandyRoz4(); dt = dx/16)

        # SecondOrderODEProblem state layout: [du; u] (velocities first)
        n        = N_e * M
        u̇_num  = reshape(sol(t1)[1:n],     N_e, M)
        u_num    = reshape(sol(t1)[n+1:2n], N_e, M)

        u_exact  = similar(x);  u̇_exact = similar(x)
        initialize!(u_exact, u̇_exact, x, t1; A=Aamp, k, ω)

        # Actual errors at t1=1 are ≈ 2e-6 (u) and ≈ 8e-4 (u̇); allow ~10×
        # margin against compiler / integrator variation.
        @test maximum(abs, u_num  - u_exact)  < 1e-4
        @test maximum(abs, u̇_num - u̇_exact) < 1e-2
    end

end
