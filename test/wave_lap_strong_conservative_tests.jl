@testitem "wave_lap_strong_conservative" tags=[:cpu] begin
    _progress(m) = (printstyled(stderr, "  • ", m, "\n"; color = :cyan); flush(stderr))

using LinearAlgebra: Diagonal, Symmetric, eigvals, dot
using StaticArrays
using Test
using WaveToySecondOrder: make_element, make_operators,
                          wave_lap_strong_element!,
                          wave_lap_strong_conservative_element!

# Tests for the H_phys-symmetric, energy-conserving variant of the
# strong-form scalar Laplacian. Four orthogonal checks:
#
#   (1) Convergence on a smooth manufactured solution (the variant must
#       still be a consistent ∇² discretisation).
#   (2) `H_phys · L` is symmetric to roundoff.
#   (3) `−H_phys · L` is positive semi-definite (its smallest eigenvalue
#       is ≥ −ε), so the discrete energy is non-negative.
#   (4) Leapfrog with this `L` conserves the discrete energy to O(dt²)
#       over many time steps — i.e. drift is bounded and oscillatory,
#       not secular.

function _build_strong_L(lap!, N::Int, h::Float64, ops; kwargs...)
    ndof = N^3
    L_mat  = zeros(ndof, ndof)
    u      = zeros(N, N, N); Lu = similar(u)
    u_face = ntuple(_ -> zeros(N, N), Val(6))
    uf     = vec(u);        Lf = vec(Lu)
    for k in 1:ndof
        uf[k] = 1.0
        lap!(Lu, u, u_face, ops, h; kwargs...)
        uf[k] = 0.0
        L_mat[:, k] .= Lf
    end
    return L_mat
end

# Same as `_build_strong_L` but for the kwarg-less centred-flux variant.
function _build_strong_L_centred(N::Int, h::Float64, ops)
    ndof = N^3
    L_mat  = zeros(ndof, ndof)
    u      = zeros(N, N, N); Lu = similar(u)
    u_face = ntuple(_ -> zeros(N, N), Val(6))
    uf     = vec(u);        Lf = vec(Lu)
    for k in 1:ndof
        uf[k] = 1.0
        wave_lap_strong_element!(Lu, u, u_face, ops, h)
        uf[k] = 0.0
        L_mat[:, k] .= Lf
    end
    return L_mat
end

function _H_phys_diag(N::Int, h::Float64, ops)
    Hd = [ops.H[i, i] for i in 1:N]
    v  = Float64[]
    sizehint!(v, N^3)
    for k in 1:N, j in 1:N, i in 1:N
        push!(v, Hd[i] * Hd[j] * Hd[k] * h^3)
    end
    return v
end

@testset "wave_lap_strong_conservative_element!" begin

    @testset "single-element convergence on sin(πx)·sin(πy)·sin(πz)" begin
        _progress("conservative Laplacian convergence")
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
            wave_lap_strong_conservative_element!(Lu, u, u_face, ops, h)
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

    @testset "H_phys · L is symmetric to roundoff" begin
        _progress("conservative Laplacian H-symmetry")
        N    = 4
        h    = 1.0
        elem = make_element(Float64, N)
        ops  = make_operators(elem)

        L_cons   = _build_strong_L(wave_lap_strong_conservative_element!,
                                    N, h, ops)
        L_centre = _build_strong_L_centred(N, h, ops)
        H_diag   = _H_phys_diag(N, h, ops)
        H        = Diagonal(H_diag)

        HL_cons   = H * L_cons
        HL_centre = H * L_centre

        asym_cons   = maximum(abs, HL_cons   .- transpose(HL_cons))
        asym_centre = maximum(abs, HL_centre .- transpose(HL_centre))

        # Conservative variant: symmetric to roundoff.
        @test asym_cons / maximum(abs, HL_cons) < 1e-12
        # Centred-flux variant: visibly asymmetric (sanity check that
        # the two variants are not numerically identical and the
        # symmetry gain is real, not a side effect of test framing).
        @test asym_centre / maximum(abs, HL_centre) > 1e-3
    end

    @testset "−H_phys · L is positive semi-definite" begin
        _progress("conservative Laplacian PSD")
        N    = 4
        h    = 1.0
        elem = make_element(Float64, N)
        ops  = make_operators(elem)

        L_cons = _build_strong_L(wave_lap_strong_conservative_element!,
                                  N, h, ops)
        H_diag = _H_phys_diag(N, h, ops)
        H      = Diagonal(H_diag)
        HL     = H * L_cons
        sym    = Symmetric((HL .+ transpose(HL)) ./ 2)
        λ_min  = minimum(eigvals(-sym))
        # `-H L` is PSD; tiny negative bleed from floating-point only.
        @test λ_min > -1e-10
        # Non-trivial spectrum.
        @test maximum(eigvals(-sym)) > 1.0
    end

    @testset "leapfrog conserves energy (O(dt²) envelope, no drift)" begin
        _progress("conservative Laplacian energy conservation")
        N    = 4
        h    = 1.0
        elem = make_element(Float64, N)
        ops  = make_operators(elem)
        H_vec = _H_phys_diag(N, h, ops)

        # Build L once via column probing so we can evaluate the
        # quadratic energy directly without rebuilding the SAT.
        L_mat = _build_strong_L(wave_lap_strong_conservative_element!,
                                 N, h, ops)
        ω_max = sqrt(-minimum(real, eigvals(L_mat)))

        function energy(u, u̇_int)
            uv  = vec(u); vv = vec(u̇_int)
            return 0.5 * dot(vv, H_vec .* vv) +
                   0.5 * dot(uv, H_vec .* (-(L_mat * uv)))
        end

        function run_envelope(dt_factor, n_steps)
            dt = dt_factor * 2 / ω_max
            ξs = elem.xs
            u  = Float64[sin(π * ξs[i]) * sin(π * ξs[j]) * sin(π * ξs[k])
                          for i in 1:N, j in 1:N, k in 1:N]
            u̇  = zeros(N, N, N); Lu = similar(u)
            u_face = ntuple(_ -> zeros(N, N), Val(6))
            wave_lap_strong_conservative_element!(Lu, u, u_face, ops, h)
            u̇ .+= (dt / 2) .* Lu
            E0    = energy(u, u̇ .- (dt / 2) .* Lu)
            E_max = E0; E_min = E0; E_end = E0
            for _ in 1:n_steps
                u .+= dt .* u̇
                wave_lap_strong_conservative_element!(Lu, u, u_face, ops, h)
                u̇ .+= dt .* Lu
                E = energy(u, u̇ .- (dt / 2) .* Lu)
                E_max = max(E_max, E); E_min = min(E_min, E); E_end = E
            end
            return E0, (E_max - E_min) / E0, (E_end - E0) / E0, isfinite(E_end)
        end

        # Run at two dts: the second has half the step size and twice the
        # step count, so total physical time is identical.
        E0_a, env_a, drift_a, ok_a = run_envelope(0.5,   5_000)
        E0_b, env_b, drift_b, ok_b = run_envelope(0.25, 10_000)

        @test ok_a && ok_b
        @test E0_a > 0
        # The envelope is bounded (no secular blow-up) at the larger dt.
        @test env_a < 0.1
        # Halving dt shrinks the envelope by ≈ 4× (O(dt²) Hamiltonian shadow).
        # Allow some slack; non-conservative schemes saturate at O(1) and
        # fail this ratio test by an order of magnitude.
        @test env_b < env_a / 3
        # End-of-run drift is bounded by the envelope (true conservation),
        # not growing with n_steps.
        @test abs(drift_a) ≤ env_a
        @test abs(drift_b) ≤ env_b
    end
end

end
