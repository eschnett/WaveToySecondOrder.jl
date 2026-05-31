using HexMeshes: make_uniform_hex
using HexSBPSAT: make_element, make_operators
using LinearAlgebra: Diagonal, Symmetric, eigvals, dot
using Test
using WaveToySecondOrder: wave_lap_strong_conservative_mesh!

# Multi-element tests for the strong-form conservative Laplacian on a
# periodic uniform-hex mesh. Periodic boundaries simplify life: with
# no outer Dirichlet faces fighting it, the constant function lives in
# the kernel exactly, and `H · L` is symmetric to roundoff.

function _build_mesh_L(mesh, ops, h::Float64)
    N  = size(ops.G, 1)
    Ne = mesh.Ne
    ndof = N^3 * Ne
    L_mat = zeros(Float64, ndof, ndof)
    u  = zeros(Float64, N, N, N, Ne); Lu = similar(u)
    uf = vec(u); Lf = vec(Lu)
    for k in 1:ndof
        uf[k] = 1.0
        wave_lap_strong_conservative_mesh!(Lu, u, mesh, ops, h)
        uf[k] = 0.0
        L_mat[:, k] .= Lf
    end
    return L_mat
end

function _H_phys_diag_mesh(mesh, ops, h::Float64)
    N  = size(ops.G, 1)
    Ne = mesh.Ne
    Hd = [ops.H[i, i] for i in 1:N]
    v  = Float64[]
    sizehint!(v, N^3 * Ne)
    for e in 1:Ne, k in 1:N, j in 1:N, i in 1:N
        push!(v, Hd[i] * Hd[j] * Hd[k] * h^3)
    end
    return v
end

@testset "wave_lap_strong_conservative_mesh! (periodic 3D)" begin

    @testset "constant function in the kernel" begin
        _progress("multi-element periodic: L·1 = 0")
        T = Float64; N = 4; h = 0.5
        mesh = make_uniform_hex(T, 2, 2, 2, 0.0, 1.0; periodic = true)
        elem = make_element(T, N); ops = make_operators(elem)
        u  = fill(one(T), N, N, N, mesh.Ne); Lu = similar(u)
        wave_lap_strong_conservative_mesh!(Lu, u, mesh, ops, h)
        @test maximum(abs, Lu) < 1e-12
    end

    @testset "H_phys · L symmetric to roundoff (2x2x2, N=3)" begin
        _progress("multi-element periodic: H·L symmetric")
        T = Float64; N = 3; h = 0.5
        mesh = make_uniform_hex(T, 2, 2, 2, 0.0, 1.0; periodic = true)
        elem = make_element(T, N); ops = make_operators(elem)
        L_mat  = _build_mesh_L(mesh, ops, h)
        H_diag = _H_phys_diag_mesh(mesh, ops, h)
        H      = Diagonal(H_diag)
        HL     = H * L_mat
        asym   = maximum(abs, HL .- transpose(HL))
        @test asym / maximum(abs, HL) < 1e-12
    end

    @testset "−H_phys · L positive semi-definite (2x2x2, N=3)" begin
        _progress("multi-element periodic: -H·L PSD")
        T = Float64; N = 3; h = 0.5
        mesh = make_uniform_hex(T, 2, 2, 2, 0.0, 1.0; periodic = true)
        elem = make_element(T, N); ops = make_operators(elem)
        L_mat  = _build_mesh_L(mesh, ops, h)
        H_diag = _H_phys_diag_mesh(mesh, ops, h)
        H      = Diagonal(H_diag)
        HL     = H * L_mat
        sym    = Symmetric((HL .+ transpose(HL)) ./ 2)
        λ_min  = minimum(eigvals(-sym))
        # The constant function gives a zero mode on the periodic torus
        # (Laplacian eigenvalue zero); allow it.
        @test λ_min > -1e-10
        @test maximum(eigvals(-sym)) > 1.0
    end

    @testset "Rayleigh-quotient eigenvalue converges to -3(2π)²" begin
        _progress("multi-element periodic: Rayleigh-quotient convergence")
        T = Float64; N = 4
        # u = sin(2π x) sin(2π y) sin(2π z) is an eigenfunction of
        # ∇² on the periodic unit cube with eigenvalue -3(2π)². The
        # SBP-SAT wide-stencil + Mattsson-Nordström SAT does *not*
        # give classical pointwise accuracy at face nodes (boundary
        # pollution), but it is spectrally consistent in the
        # H_phys-weighted Rayleigh quotient
        #     λ_h = ⟨u, -H_phys L u⟩ / ⟨u, H_phys u⟩
        # which is the relevant quantity for wave dispersion and
        # energy conservation.
        u_fun(x, y, z) = sin(2π * x) * sin(2π * y) * sin(2π * z)
        λ_ex = 3 * (2π)^2
        function rayleigh(M)
            mesh = make_uniform_hex(T, M, M, M, 0.0, 1.0; periodic = true)
            elem = make_element(T, N); ops = make_operators(elem)
            h    = 1.0 / M
            ξs   = elem.xs
            Hd   = [ops.H[i, i] for i in 1:N]
            H_vec = T[]
            for e in 1:mesh.Ne, k in 1:N, j in 1:N, i in 1:N
                push!(H_vec, Hd[i] * Hd[j] * Hd[k] * h^3)
            end
            u = Array{T, 4}(undef, N, N, N, mesh.Ne)
            for e in 1:mesh.Ne
                v_lo = mesh.vertex_idx[1, e]
                x0 = mesh.vertex_coords[1, v_lo]
                y0 = mesh.vertex_coords[2, v_lo]
                z0 = mesh.vertex_coords[3, v_lo]
                for k in 1:N, j in 1:N, i in 1:N
                    u[i, j, k, e] = u_fun(x0 + h * ξs[i],
                                            y0 + h * ξs[j],
                                            z0 + h * ξs[k])
                end
            end
            Lu = similar(u)
            wave_lap_strong_conservative_mesh!(Lu, u, mesh, ops, h)
            uv = vec(u); Lv = vec(Lu)
            return -dot(uv, H_vec .* Lv) / dot(uv, H_vec .* uv)
        end

        errs = [abs(rayleigh(M) - λ_ex) for M in (1, 2, 4, 8)]
        @test all(errs .> 0)
        # Each halving shrinks the error by orders of magnitude
        # (spectral convergence on a smooth eigenmode).
        @test errs[2] < errs[1] / 5      # M=1 → M=2: ≥ 5×
        @test errs[3] < errs[2] / 50     # M=2 → M=4: ≥ 50×
        @test errs[4] < errs[3] / 10     # M=4 → M=8: ≥ 10× (roundoff-limited)
        # Finest mesh is essentially the analytic eigenvalue.
        @test errs[end] < 1e-3
    end

    @testset "leapfrog energy O(dt²) on periodic 2x2x2" begin
        _progress("multi-element periodic: energy conservation")
        T = Float64; N = 3; h = 0.5
        mesh = make_uniform_hex(T, 2, 2, 2, 0.0, 1.0; periodic = true)
        elem = make_element(T, N); ops = make_operators(elem)
        L_mat = _build_mesh_L(mesh, ops, h)
        H_vec = _H_phys_diag_mesh(mesh, ops, h)
        ω_max = sqrt(-minimum(real, eigvals(L_mat)))

        function energy(u, u̇_int)
            uv = vec(u); vv = vec(u̇_int)
            return 0.5 * dot(vv, H_vec .* vv) +
                   0.5 * dot(uv, H_vec .* (-(L_mat * uv)))
        end

        function run_envelope(dt_factor, n_steps)
            dt = dt_factor * 2 / ω_max
            ξs = elem.xs
            u  = Array{T, 4}(undef, N, N, N, mesh.Ne)
            for e in 1:mesh.Ne
                v_lo = mesh.vertex_idx[1, e]
                x0 = mesh.vertex_coords[1, v_lo]
                y0 = mesh.vertex_coords[2, v_lo]
                z0 = mesh.vertex_coords[3, v_lo]
                for k in 1:N, j in 1:N, i in 1:N
                    u[i, j, k, e] = sin(2π * (x0 + h * ξs[i])) *
                                     sin(2π * (y0 + h * ξs[j])) *
                                     sin(2π * (z0 + h * ξs[k]))
                end
            end
            u̇  = zeros(T, N, N, N, mesh.Ne); Lu = similar(u)
            wave_lap_strong_conservative_mesh!(Lu, u, mesh, ops, h)
            u̇ .+= (dt / 2) .* Lu
            E0    = energy(u, u̇ .- (dt / 2) .* Lu)
            E_max = E0; E_min = E0; E_end = E0
            for _ in 1:n_steps
                u .+= dt .* u̇
                wave_lap_strong_conservative_mesh!(Lu, u, mesh, ops, h)
                u̇ .+= dt .* Lu
                E = energy(u, u̇ .- (dt / 2) .* Lu)
                E_max = max(E_max, E); E_min = min(E_min, E); E_end = E
            end
            return E0, (E_max - E_min) / E0, isfinite(E_end)
        end

        E0_a, env_a, ok_a = run_envelope(0.5,   2_000)
        E0_b, env_b, ok_b = run_envelope(0.25, 4_000)
        @test ok_a && ok_b
        @test E0_a > 0
        @test env_a < 0.5            # bounded oscillation at the larger dt
        @test env_b < env_a / 3      # ≈ 4× shrink when dt halves ⇒ O(dt²)
    end
end
