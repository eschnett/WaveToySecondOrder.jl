@testitem "wave_lap_strong" tags=[:cpu] begin
    _progress(m) = (printstyled(stderr, "  • ", m, "\n"; color = :cyan); flush(stderr))

using LinearAlgebra: eigvals
using StaticArrays
using Test
using WaveToySecondOrder: make_element, make_operators,
                          wave_lap_strong_element!

# Single-element sanity / stability tests for the strong-form scalar
# Laplacian `wave_lap_strong_element!`. Three orthogonal checks:
#
#   (1) Convergence on a smooth manufactured solution.
#   (2) The assembled discrete operator `L_h` has its spectrum on the
#       negative real axis (so leapfrog is stable for small enough dt).
#   (3) A direct leapfrog run from a smooth IC stays bounded over
#       many time steps.
#
# The centred-flux SAT used here is *not* H_phys-symmetric, so the
# usual quadratic energy is not conserved — empirical stability is
# the only claim. The H_phys-symmetric (energy-conserving) variant
# will get its own test file when it lands.

@testset "wave_lap_strong_element!" begin
    @testset "single-element convergence on sin(πx)·sin(πy)·sin(πz)" begin
        _progress("strong-form Laplacian convergence")
        N    = 4
        elem = make_element(Float64, N)
        ops  = make_operators(elem)
        T    = Float64
        u_fun(x, y, z)   = sin(π * x) * sin(π * y) * sin(π * z)
        lap_fun(x, y, z) = -3 * T(π)^2 * u_fun(x, y, z)
        origin           = SVector{3, T}(T(0.1), T(0.1), T(0.1))

        function build_data(h::T)
            ξs = elem.xs
            u  = Array{T, 3}(undef, N, N, N)
            for k in 1:N, j in 1:N, i in 1:N
                u[i, j, k] = u_fun(origin[1] + h * ξs[i],
                                    origin[2] + h * ξs[j],
                                    origin[3] + h * ξs[k])
            end
            function face(::Val{f}) where {f}
                row = isodd(f) ? 1 : N
                a   = (f + 1) ÷ 2
                uf  = Array{T, 2}(undef, N, N)
                for q in 1:N, p in 1:N
                    i, j, k = (a == 1 ? (row, p, q) :
                               a == 2 ? (p, row, q) :
                                        (p, q, row))
                    uf[p, q] = u_fun(origin[1] + h * ξs[i],
                                      origin[2] + h * ξs[j],
                                      origin[3] + h * ξs[k])
                end
                return uf
            end
            u, ntuple(f -> face(Val(f)), Val(6))
        end

        function exact_lap(h::T)
            ξs = elem.xs
            out = Array{T, 3}(undef, N, N, N)
            for k in 1:N, j in 1:N, i in 1:N
                out[i, j, k] = lap_fun(origin[1] + h * ξs[i],
                                        origin[2] + h * ξs[j],
                                        origin[3] + h * ξs[k])
            end
            return out
        end

        hs   = T[1.0, 0.5, 0.25, 0.125]
        errs = T[]
        for h in hs
            u, u_face = build_data(h)
            Lu = similar(u)
            wave_lap_strong_element!(Lu, u, u_face, ops, h)
            err = maximum(abs,
                Lu[2:N-1, 2:N-1, 2:N-1] .- exact_lap(h)[2:N-1, 2:N-1, 2:N-1])
            push!(errs, err)
        end
        @test all(errs .> 0)
        @test errs[end] < errs[1]
        gmean_ratio = (errs[1] / errs[end])^(1 / (length(hs) - 1))
        @test gmean_ratio > 4.0
        @test errs[end] < 1.0
    end

    @testset "spectrum on negative real axis (N=4, h=1)" begin
        _progress("strong-form Laplacian spectrum")
        N    = 4
        h    = 1.0
        elem = make_element(Float64, N)
        ops  = make_operators(elem)

        # Assemble L_h as a 64×64 dense matrix by applying
        # `wave_lap_strong_element!` to each canonical basis vector
        # with Dirichlet-zero face traces.
        ndof   = N^3
        L_mat  = zeros(ndof, ndof)
        u      = zeros(N, N, N);   Lu = similar(u)
        u_face = ntuple(_ -> zeros(N, N), Val(6))
        uf     = vec(u);          Lf = vec(Lu)
        for k in 1:ndof
            uf[k] = 1.0
            wave_lap_strong_element!(Lu, u, u_face, ops, h)
            uf[k] = 0.0
            L_mat[:, k] .= Lf
        end
        λ = eigvals(L_mat)
        @test maximum(real, λ)         < 1e-10
        @test maximum(abs ∘ imag, λ)    < 1e-10
        @test minimum(real, λ)         < -1.0
    end

    @testset "leapfrog stays bounded over 5 000 steps" begin
        _progress("strong-form Laplacian leapfrog stability")
        N    = 4
        h    = 1.0
        elem = make_element(Float64, N)
        ops  = make_operators(elem)

        # Smooth IC vanishing on the cube faces ⇒ matches Dirichlet-zero
        # face traces exactly at t = 0.
        ξs = elem.xs
        u  = Float64[sin(π * ξs[i]) * sin(π * ξs[j]) * sin(π * ξs[k])
                      for i in 1:N, j in 1:N, k in 1:N]
        u̇  = zeros(N, N, N)
        Lu = similar(u)
        u_face = ntuple(_ -> zeros(N, N), Val(6))

        # |λ_min| ≈ 90 for (N, h) = (4, 1) ⇒ stability boundary dt ≈ 0.21;
        # take half of that.
        dt = 0.1

        wave_lap_strong_element!(Lu, u, u_face, ops, h)
        u̇ .+= (dt / 2) .* Lu

        initial_max = maximum(abs, u)
        bound_seen  = initial_max
        n_steps     = 5_000
        for _ in 1:n_steps
            u .+= dt .* u̇
            wave_lap_strong_element!(Lu, u, u_face, ops, h)
            u̇ .+= dt .* Lu
            bound_seen = max(bound_seen, maximum(abs, u))
        end
        @test all(isfinite, u)
        @test all(isfinite, u̇)
        @test bound_seen < 5 * initial_max
    end
end

end
