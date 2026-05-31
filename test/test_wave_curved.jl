using HexMeshes: make_uniform_hex
using HexSBPSAT: make_element, make_operators
using LinearAlgebra: Diagonal, Symmetric, dot, eigvals
using Random: Random
using SpacetimeMetrics: GaugeWave, Minkowski
using StaticArrays
using Test
using WaveToySecondOrder: eval_curved_background!,
                          wave_curved_rhs_element!,
                          wave_curved_rhs_mesh!,
                          wave_curved_rhs_conservative_element!,
                          wave_curved_rhs_conservative_mesh!,
                          wave_lap_strong_element!,
                          wave_lap_strong_conservative_element!

# Scalar wave equation on a prescribed 4-metric, fully second-order
# (mirrors `gh_rhs_element!`). The centred-flux SAT used here matches
# GH but is *not* energy-conservative — the multi-element discrete
# spectrum picks up small imaginary parts and the pointwise Rayleigh
# quotient does not converge spectrally. These tests therefore check
# what the discretisation *does* provide:
#
#   1. On Minkowski, the kernel reduces *exactly* to the centred-flux
#      scalar Laplacian (`wave_lap_strong_element!`) on a single
#      element with self-as-neighbour periodic face data.
#   2. Constant `φ` is annihilated to roundoff on the gauge wave at
#      every t.
#   3. Discrete spectrum (curved-RHS path, Minkowski) is bounded and
#      essentially on the negative real axis (small imaginary parts
#      from the non-conservative SAT are tolerated).
#   4. Time evolution from constant initial data on the gauge wave
#      stays bounded over thousands of steps (closes the time-dependent
#      metric re-evaluation loop).
#
# Spectral convergence in the variable-coefficient direction is *not*
# claimed here; that requires the energy-conservative Mattsson–
# Nordström SAT, which has its own follow-up plan.

@testset "wave_curved_rhs_mesh! — Minkowski + GaugeWave" begin

    @testset "Minkowski reduction matches wave_lap_strong_element!" begin
        _progress("curved-RHS Minkowski ↔ centred-flux Laplacian")
        T = Float64; N = 4
        # Single periodic element — wave_curved_rhs_mesh!'s face
        # extraction then mirrors the self-periodic case, which is
        # exactly how a single-element centred-flux Laplacian with
        # u_face = (self at opposite face) would be invoked.
        mesh = make_uniform_hex(T, 1, 1, 1, 0.0, 1.0; periodic = true)
        elem = make_element(T, N); ops = make_operators(elem)
        h    = 1.0
        ξs   = elem.xs

        ginv = Array{T,5}(undef, 10, N, N, N, 1)
        Γc   = Array{T,5}(undef,  4, N, N, N, 1)
        eval_curved_background!(ginv, Γc, mesh, ξs, Minkowski(), 0.0, h)

        # Random-ish φ; φ̇ should not affect output (g^{ti} = 0, Γ^t = 0).
        φ_3 = Float64[sin(2π * ξs[i]) * cos(2π * ξs[j]) * sin(2π * ξs[k])
                       for i in 1:N, j in 1:N, k in 1:N]
        φ  = reshape(copy(φ_3), N, N, N, 1)
        φ̇ = randn(N, N, N, 1)
        φ̈ = similar(φ)
        wave_curved_rhs_mesh!(φ̈, φ, φ̇, mesh, ginv, Γc, ops, h)

        # Build the expected centred-flux Laplacian via `wave_lap_strong_element!`
        # with face traces drawn from the same periodic mesh (so faces
        # 1↔2, 3↔4, 5↔6 mirror across the seam).
        L_φ = similar(φ_3)
        u_face = ntuple(Val(6)) do f
            row_nbr = (f == 1) ? N : (f == 2) ? 1 :
                      (f == 3) ? N : (f == 4) ? 1 :
                      (f == 5) ? N : 1
            uf = Matrix{Float64}(undef, N, N)
            if f ≤ 2
                for q in 1:N, p in 1:N; uf[p, q] = φ_3[row_nbr, p, q]; end
            elseif f ≤ 4
                for q in 1:N, p in 1:N; uf[p, q] = φ_3[p, row_nbr, q]; end
            else
                for q in 1:N, p in 1:N; uf[p, q] = φ_3[p, q, row_nbr]; end
            end
            uf
        end
        wave_lap_strong_element!(L_φ, φ_3, u_face, ops, h)

        # ∂_tt φ on Minkowski equals the centred-flux Laplacian of φ
        # (independent of φ̇ since g^{ti} = 0 and Γ^t = 0).
        @test maximum(abs, φ̈[:, :, :, 1] .- L_φ) < 1e-12
    end

    @testset "Constant φ on gauge wave: ∂_tt φ ≈ 0 across t" begin
        _progress("curved-RHS gauge wave: constant")
        T = Float64; N = 4
        mesh = make_uniform_hex(T, 2, 2, 2, 0.0, 1.0; periodic = true)
        elem = make_element(T, N); ops = make_operators(elem)
        h    = 0.5
        ξs   = elem.xs
        ginv = Array{T,5}(undef, 10, N, N, N, mesh.Ne)
        Γc   = Array{T,5}(undef,  4, N, N, N, mesh.Ne)
        φ  = fill(one(T), N, N, N, mesh.Ne)
        φ̇ = zeros(T, N, N, N, mesh.Ne)
        φ̈ = similar(φ)
        gw = GaugeWave(0.1, 1.0)
        for t in (0.0, 0.13, 0.25, 0.5)
            eval_curved_background!(ginv, Γc, mesh, ξs, gw, t, h)
            wave_curved_rhs_mesh!(φ̈, φ, φ̇, mesh, ginv, Γc, ops, h)
            @test maximum(abs, φ̈) < 1e-10
        end
    end

    @testset "Discrete spectrum bounded on Minkowski" begin
        _progress("curved-RHS Minkowski spectrum")
        T = Float64; N = 3
        mesh = make_uniform_hex(T, 2, 2, 2, 0.0, 1.0; periodic = true)
        elem = make_element(T, N); ops = make_operators(elem)
        h    = 0.5
        ξs   = elem.xs

        ginv = Array{T,5}(undef, 10, N, N, N, mesh.Ne)
        Γc   = Array{T,5}(undef,  4, N, N, N, mesh.Ne)
        eval_curved_background!(ginv, Γc, mesh, ξs, Minkowski(), 0.0, h)

        ndof  = N^3 * mesh.Ne
        L_mat = zeros(T, ndof, ndof)
        φ  = zeros(T, N, N, N, mesh.Ne);  φ̇ = zeros(T, N, N, N, mesh.Ne)
        φ̈ = similar(φ)
        φv = vec(φ);  φ̈v = vec(φ̈)
        for k in 1:ndof
            φv[k] = 1.0
            wave_curved_rhs_mesh!(φ̈, φ, φ̇, mesh, ginv, Γc, ops, h)
            φv[k] = 0.0
            L_mat[:, k] .= φ̈v
        end
        λ = eigvals(L_mat)
        # Centred-flux SAT on multi-element is not energy-conservative
        # and the spectrum picks up small imaginary parts. Real parts
        # may also pick up small positive bumps from the non-symmetric
        # discretisation. The discretisation is still stable (bounded
        # growth rate) — we just check magnitudes.
        @test maximum(real, λ)       <  0.1
        @test maximum(abs ∘ imag, λ) <  0.1
        @test minimum(real, λ)       < -1.0
        @test all(isfinite, λ)
    end

    @testset "Time evolution on gauge wave: bounded from constant IC" begin
        _progress("curved-RHS gauge wave: leapfrog stability")
        T = Float64; N = 4
        mesh = make_uniform_hex(T, 2, 2, 2, 0.0, 1.0; periodic = true)
        elem = make_element(T, N); ops = make_operators(elem)
        h    = 0.5
        ξs   = elem.xs

        ginv = Array{T,5}(undef, 10, N, N, N, mesh.Ne)
        Γc   = Array{T,5}(undef,  4, N, N, N, mesh.Ne)
        φ  = fill(one(T), N, N, N, mesh.Ne)
        φ̇ = zeros(T, N, N, N, mesh.Ne)
        φ̈ = similar(φ)

        gw = GaugeWave(0.05, 1.0)
        dt = 0.01
        n_steps = 1_000

        eval_curved_background!(ginv, Γc, mesh, ξs, gw, 0.0, h)
        wave_curved_rhs_mesh!(φ̈, φ, φ̇, mesh, ginv, Γc, ops, h)
        φ̇ .+= (dt / 2) .* φ̈

        bound_seen = maximum(abs, φ .- 1)
        t = 0.0
        for _ in 1:n_steps
            φ .+= dt .* φ̇
            t  += dt
            eval_curved_background!(ginv, Γc, mesh, ξs, gw, t, h)
            wave_curved_rhs_mesh!(φ̈, φ, φ̇, mesh, ginv, Γc, ops, h)
            φ̇ .+= dt .* φ̈
            bound_seen = max(bound_seen, maximum(abs, φ .- 1))
        end
        @test all(isfinite, φ)
        @test all(isfinite, φ̇)
        @test bound_seen < 0.1
    end
end

@testset "wave_curved_rhs_conservative_mesh! — Minkowski + GaugeWave" begin

    @testset "Minkowski reduction matches wave_lap_strong_conservative_element!" begin
        _progress("conservative-curved Minkowski ↔ flat conservative")
        T = Float64; N = 4
        mesh = make_uniform_hex(T, 1, 1, 1, 0.0, 1.0; periodic = true)
        elem = make_element(T, N); ops = make_operators(elem)
        h    = 1.0
        ξs   = elem.xs

        ginv = Array{T,5}(undef, 10, N, N, N, 1)
        Γc   = Array{T,5}(undef,  4, N, N, N, 1)
        eval_curved_background!(ginv, Γc, mesh, ξs, Minkowski(), 0.0, h)

        Random.seed!(2026_05_31)
        φ_3 = randn(N, N, N)
        φ  = reshape(copy(φ_3), N, N, N, 1)
        φ̇ = randn(N, N, N, 1)         # should not affect output
        φ̈ = similar(φ)
        wave_curved_rhs_conservative_mesh!(φ̈, φ, φ̇, mesh, ginv, Γc, ops, h)

        # Build the matching face data for the flat conservative kernel
        # (self-as-neighbour for single periodic element).
        u_face = ntuple(Val(6)) do f
            uf = Matrix{Float64}(undef, N, N)
            row = (f == 1) ? N : (f == 2) ? 1 :
                  (f == 3) ? N : (f == 4) ? 1 :
                  (f == 5) ? N : 1
            if f ≤ 2
                for q in 1:N, p in 1:N; uf[p, q] = φ_3[row, p, q]; end
            elseif f ≤ 4
                for q in 1:N, p in 1:N; uf[p, q] = φ_3[p, row, q]; end
            else
                for q in 1:N, p in 1:N; uf[p, q] = φ_3[p, q, row]; end
            end
            uf
        end
        L_φ = similar(φ_3)
        wave_lap_strong_conservative_element!(L_φ, φ_3, u_face, ops, h)

        @test maximum(abs, φ̈[:, :, :, 1] .- L_φ) < 1e-12
    end

    @testset "H_phys · L symmetric on Minkowski (conservative path)" begin
        _progress("conservative-curved Minkowski H-symmetry")
        T = Float64; N = 3
        mesh = make_uniform_hex(T, 2, 2, 2, 0.0, 1.0; periodic = true)
        elem = make_element(T, N); ops = make_operators(elem)
        h    = 0.5
        ξs   = elem.xs

        ginv = Array{T,5}(undef, 10, N, N, N, mesh.Ne)
        Γc   = Array{T,5}(undef,  4, N, N, N, mesh.Ne)
        eval_curved_background!(ginv, Γc, mesh, ξs, Minkowski(), 0.0, h)

        ndof  = N^3 * mesh.Ne
        L_mat = zeros(T, ndof, ndof)
        φ  = zeros(T, N, N, N, mesh.Ne); φ̇ = zeros(T, N, N, N, mesh.Ne)
        φ̈ = similar(φ)
        φv = vec(φ); φ̈v = vec(φ̈)
        for k in 1:ndof
            φv[k] = 1.0
            wave_curved_rhs_conservative_mesh!(φ̈, φ, φ̇, mesh, ginv, Γc, ops, h)
            φv[k] = 0.0
            L_mat[:, k] .= φ̈v
        end
        # Build H_phys diagonal.
        Hd = [ops.H[i, i] for i in 1:N]
        H_vec = T[]
        for _ in 1:mesh.Ne, k in 1:N, j in 1:N, i in 1:N
            push!(H_vec, Hd[i] * Hd[j] * Hd[k] * h^3)
        end
        H_diag = Diagonal(H_vec)
        HL     = H_diag * L_mat
        asym   = maximum(abs, HL .- transpose(HL))
        @test asym / maximum(abs, HL) < 1e-12
    end

    @testset "Spectrum on Minkowski: pure-real, negative axis (conservative)" begin
        _progress("conservative-curved Minkowski spectrum")
        T = Float64; N = 3
        mesh = make_uniform_hex(T, 2, 2, 2, 0.0, 1.0; periodic = true)
        elem = make_element(T, N); ops = make_operators(elem)
        h    = 0.5
        ξs   = elem.xs

        ginv = Array{T,5}(undef, 10, N, N, N, mesh.Ne)
        Γc   = Array{T,5}(undef,  4, N, N, N, mesh.Ne)
        eval_curved_background!(ginv, Γc, mesh, ξs, Minkowski(), 0.0, h)

        ndof  = N^3 * mesh.Ne
        L_mat = zeros(T, ndof, ndof)
        φ  = zeros(T, N, N, N, mesh.Ne); φ̇ = zeros(T, N, N, N, mesh.Ne)
        φ̈ = similar(φ)
        φv = vec(φ); φ̈v = vec(φ̈)
        for k in 1:ndof
            φv[k] = 1.0
            wave_curved_rhs_conservative_mesh!(φ̈, φ, φ̇, mesh, ginv, Γc, ops, h)
            φv[k] = 0.0
            L_mat[:, k] .= φ̈v
        end
        λ = eigvals(L_mat)
        # Contrast: centred-flux variant needed slack of 0.1 on both axes.
        # Conservative variant matches the flat conservative scheme on
        # Minkowski exactly, so the spectrum is pure-real and ≤ 0
        # (modulo roundoff and the constant-mode zero eigenvalue).
        @test maximum(real, λ)        < 1e-10
        @test maximum(abs ∘ imag, λ)   < 1e-10
        @test minimum(real, λ)        < -1.0
    end

    @testset "Constant φ on gauge wave: ∂_tt φ ≈ 0 (conservative)" begin
        _progress("conservative-curved gauge wave: constant")
        T = Float64; N = 4
        mesh = make_uniform_hex(T, 2, 2, 2, 0.0, 1.0; periodic = true)
        elem = make_element(T, N); ops = make_operators(elem)
        h    = 0.5
        ξs   = elem.xs
        ginv = Array{T,5}(undef, 10, N, N, N, mesh.Ne)
        Γc   = Array{T,5}(undef,  4, N, N, N, mesh.Ne)
        φ  = fill(one(T), N, N, N, mesh.Ne)
        φ̇ = zeros(T, N, N, N, mesh.Ne)
        φ̈ = similar(φ)
        gw = GaugeWave(0.1, 1.0)
        for t in (0.0, 0.13, 0.25, 0.5)
            eval_curved_background!(ginv, Γc, mesh, ξs, gw, t, h)
            wave_curved_rhs_conservative_mesh!(φ̈, φ, φ̇, mesh, ginv, Γc, ops, h)
            @test maximum(abs, φ̈) < 1e-10
        end
    end

    @testset "Time evolution on gauge wave: bounded from constant IC (conservative)" begin
        _progress("conservative-curved gauge wave: leapfrog stability")
        T = Float64; N = 4
        mesh = make_uniform_hex(T, 2, 2, 2, 0.0, 1.0; periodic = true)
        elem = make_element(T, N); ops = make_operators(elem)
        h    = 0.5
        ξs   = elem.xs

        ginv = Array{T,5}(undef, 10, N, N, N, mesh.Ne)
        Γc   = Array{T,5}(undef,  4, N, N, N, mesh.Ne)
        φ  = fill(one(T), N, N, N, mesh.Ne)
        φ̇ = zeros(T, N, N, N, mesh.Ne)
        φ̈ = similar(φ)

        gw = GaugeWave(0.05, 1.0)
        dt = 0.01
        n_steps = 1_000

        eval_curved_background!(ginv, Γc, mesh, ξs, gw, 0.0, h)
        wave_curved_rhs_conservative_mesh!(φ̈, φ, φ̇, mesh, ginv, Γc, ops, h)
        φ̇ .+= (dt / 2) .* φ̈

        bound_seen = maximum(abs, φ .- 1)
        t = 0.0
        for _ in 1:n_steps
            φ .+= dt .* φ̇
            t  += dt
            eval_curved_background!(ginv, Γc, mesh, ξs, gw, t, h)
            wave_curved_rhs_conservative_mesh!(φ̈, φ, φ̇, mesh, ginv, Γc, ops, h)
            φ̇ .+= dt .* φ̈
            bound_seen = max(bound_seen, maximum(abs, φ .- 1))
        end
        @test all(isfinite, φ)
        @test all(isfinite, φ̇)
        @test bound_seen < 0.1
    end

    @testset "Side-by-side: conservative ≠ centred-flux on gauge wave" begin
        _progress("conservative-curved vs centred-flux on gauge wave")
        T = Float64; N = 4
        mesh = make_uniform_hex(T, 2, 2, 2, 0.0, 1.0; periodic = true)
        elem = make_element(T, N); ops = make_operators(elem)
        h    = 0.5
        ξs   = elem.xs
        ginv = Array{T,5}(undef, 10, N, N, N, mesh.Ne)
        Γc   = Array{T,5}(undef,  4, N, N, N, mesh.Ne)
        eval_curved_background!(ginv, Γc, mesh, ξs, GaugeWave(0.1, 1.0), 0.0, h)

        Random.seed!(2026_06_01)
        φ  = randn(T, N, N, N, mesh.Ne)
        φ̇ = randn(T, N, N, N, mesh.Ne)
        φ̈_cf  = similar(φ); φ̈_cons = similar(φ)
        wave_curved_rhs_mesh!(             φ̈_cf,   φ, φ̇, mesh, ginv, Γc, ops, h)
        wave_curved_rhs_conservative_mesh!(φ̈_cons, φ, φ̇, mesh, ginv, Γc, ops, h)

        # The two SAT choices give observably different operators on a
        # curved background. Roundoff would mean one of them is wrong;
        # a huge difference would mean ill-conditioned penalty. We
        # expect a relative difference well above roundoff but bounded.
        rel = maximum(abs, φ̈_cons .- φ̈_cf) / maximum(abs, φ̈_cf)
        @test rel > 1e-6
        @test rel < 10.0
    end
end
