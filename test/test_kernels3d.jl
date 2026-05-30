# Tests for `src/kernels3d.jl`: curvilinear-aware `rhs_wave3d!`. The full
# wave-evolution check against the analytic separable eigenmode is the
# end-to-end correctness gate. Each wave-evolution / stability test runs
# in both `Float64` and `Float32` to keep the GPU-friendly (Float32-only)
# path honest.

using WaveToySecondOrder
using WaveToySecondOrder: make_element, make_operators,
    make_geometry, make_workspace,
    initialize3d!, rhs_wave3d!, recommended_dt,
    discrete_inner_product, eigenmode_radial!, Params3d, to_device
using HexMeshes: make_uniform_hex, make_cubed_cube_mesh, make_inflated_cube_mesh
using KernelAbstractions: CPU
using OrdinaryDiffEqSymplecticRK
using Random
using Test

# `Metal` is a weak dep (`[weakdeps]` in Project.toml). Try to load it;
# if it's not installed or not functional on this machine, the GPU
# testset further down is skipped silently. Non-Apple CI runners and
# pure-CPU users therefore see no failure here.
const HAS_METAL = try
    @eval using Metal
    Metal.functional()
catch
    false
end

# Discrete Hamiltonian for the SBP-DG wave equation:
#
#   E = ½ ⟨u̇, u̇⟩_{H_phys}  +  ½ ⟨u, −L_h u⟩_{H_phys}
#
# Kinetic + (positive) potential. For a symplectic integrator on a
# negative-semi-definite `L_h` this should oscillate around a constant
# within an O(dt^p) modified-Hamiltonian envelope.
function discrete_energy(u::AbstractArray{T,4}, u̇::AbstractArray{T,4},
                          geom, ops, work, τ) where {T}
    bdry = ntuple(_ -> zero(T), Val(6))
    Lu   = similar(u)
    rhs_wave3d!(Lu, u, u̇, bdry; geom, ops, work, τ)
    K = discrete_inner_product(u̇, u̇, geom, ops) / 2
    V = -discrete_inner_product(u, Lu, geom, ops) / 2
    return K + V
end

# `_progress` is defined in `runtests.jl`; fallback for stand-alone runs.
@isdefined(_progress) ||
    (_progress(msg) = (printstyled(stderr, "  • ", msg, "\n"; color = :cyan);
                       flush(stderr)))

@testset "kernels3d (T=$T)" for T in (Float64, Float32)

    # Float32 reaches the ~1e-6 round-off floor in a few hundred
    # timesteps; widen the analytic-error tolerance accordingly. The
    # qualitative correctness check (no blow-up, energy conserved) is
    # identical for both precisions.
    analytic_tol = T === Float64 ? 1.0e-2 : 5.0e-2
    energy_tol   = T === Float64 ? 0.10   : 0.15

    _progress("wave evolution (M=4, N=4, T=$T)")
    @testset "wave evolution matches analytic solution (M=4, N=4, T=$T)" begin
        # Evolve u_tt = ∇²u with the separable analytic eigenmode
        #   u(x,y,z,t) = sin(2π x)·sin(2π y)·sin(2π z) · cos(ω·t)
        # on the unit cube with homogeneous Dirichlet BC on all six
        # outer faces. ω² = kx² + ky² + kz² = 3·(2π)².
        N    = 4
        M    = 4
        elem = make_element(T, N)
        ops  = make_operators(elem)
        mesh = make_uniform_hex(T, M, zero(T), one(T))
        geom = make_geometry(mesh, elem)
        work = make_workspace(geom)
        coords = geom.coords

        dx = elem.h * (one(T) / M)

        u  = Array{T, 4}(undef, N, N, N, mesh.Ne)
        u̇  = similar(u)

        params = Params3d(;
            A           = one(T),
            k           = (T(2π), T(2π), T(2π)),
            ω           = T(sqrt(3 * (2π)^2)),
            τ           = T(3//2) * (N-1)^2,
            bdry_values = ntuple(_ -> zero(T), Val(6)),
        )
        initialize3d!(u, u̇, coords, zero(T), params)

        dt = (T(1//2) * dx) / sqrt(T(3))
        t1 = T(1//2)   # ≈ 0.87 periods of the eigenmode — enough to
                       # catch dispersion error without doubling runtime.

        f!(ü, u̇, u, p::Params3d, t) = rhs_wave3d!(ü, u, u̇, p; geom, ops, work)
        prob = SecondOrderODEProblem(f!, u̇, u, (zero(T), t1), params)
        sol  = solve(prob, KahanLi8(); dt)

        u_exact = similar(u);  u̇_exact = similar(u)
        initialize3d!(u_exact, u̇_exact, coords, t1, params)

        n     = N^3 * mesh.Ne
        final = sol(t1)
        u_num = reshape(final[n+1 : 2n], N, N, N, mesh.Ne)

        @test maximum(abs, u_num - u_exact) < analytic_tol
    end

    _progress("cubed cube bounded (M=2, N=4, T=$T)")
    @testset "cubed cube evolution stays bounded (M=2, N=4, R=0.3, T=$T)" begin
        # Quick stability check on the curvilinear / multi-patch mesh.
        # With `τ ≈ 8·(N−1)²` the discrete operator is NSD, so a
        # symplectic integrator preserves bounded energy.
        N    = 4
        M    = 2
        R    = T(0.3)
        elem = make_element(T, N)
        ops  = make_operators(elem)
        mesh = make_cubed_cube_mesh(T, M, R)
        geom = make_geometry(mesh, elem)
        work = make_workspace(geom)
        coords = geom.coords

        # Domain-normalised sine IC on [-1, +1]³ (vanishes on outer faces).
        x0, x1, L_ = -one(T), one(T), T(2)
        params = Params3d(;
            A           = one(T),
            k           = (T(3π), T(3π), T(3π)),
            ω           = T(sqrt(3 * (3π)^2)) / L_,
            τ           = T(8) * (N-1)^2,
            bdry_values = ntuple(_ -> zero(T), Val(6)),
        )
        kx, ky, kz = params.k
        u  = Array{T, 4}(undef, N, N, N, mesh.Ne)
        u̇  = similar(u)
        for e in 1:mesh.Ne, k in 1:N, j in 1:N, i in 1:N
            X = (coords[1, i, j, k, e] - x0) / L_
            Y = (coords[2, i, j, k, e] - x0) / L_
            Z = (coords[3, i, j, k, e] - x0) / L_
            u[i, j, k, e]  = params.A * sin(kx*X) * sin(ky*Y) * sin(kz*Z)
            u̇[i, j, k, e] = zero(T)
        end
        u0_max = maximum(abs, u)

        dt = recommended_dt(geom, ops, params.τ)
        f!(ü, u̇, u, p::Params3d, t) = rhs_wave3d!(ü, u, u̇, p; geom, ops, work)
        prob = SecondOrderODEProblem(f!, u̇, u, (zero(T), T(0.1)), params)
        sol  = solve(prob, KahanLi8(); dt,
                     save_everystep = false, save_start = false,
                     dense = false, save_end = true)

        n     = N^3 * mesh.Ne
        final = sol.u[end]
        u_end = reshape(view(final, n+1 : 2n), N, N, N, mesh.Ne)

        @test all(isfinite, u_end)
        @test maximum(abs, u_end) < 2 * u0_max
    end

    _progress("robust stability — cubical (T=$T)")
    @testset "robust stability — cubical M=3, N=3, noise IC (T=$T)" begin
        # Coarse axis-aligned mesh, broad-spectrum (random) IC, short
        # integration. Discrete operator is symmetric NSD here so the
        # symplectic integrator should keep the Hamiltonian within an
        # O(dt^p) envelope of its initial value.
        N    = 3
        M    = 3
        elem = make_element(T, N)
        ops  = make_operators(elem)
        mesh = make_uniform_hex(T, M, zero(T), one(T))
        geom = make_geometry(mesh, elem)
        work = make_workspace(geom)

        Random.seed!(20250522)
        u  = randn(T, N, N, N, mesh.Ne)
        u̇  = randn(T, N, N, N, mesh.Ne)

        params = Params3d(;
            A           = zero(T),
            k           = (zero(T), zero(T), zero(T)),
            ω           = zero(T),
            τ           = T(3//2) * (N - 1)^2,
            bdry_values = ntuple(_ -> zero(T), Val(6)),
        )
        dt      = recommended_dt(geom, ops, params.τ; cfl_safety = T(0.5))
        n_steps = 50
        t_end   = n_steps * dt

        E0 = discrete_energy(u, u̇, geom, ops, work, params.τ)

        f!(ü, u̇, u, p::Params3d, t) = rhs_wave3d!(ü, u, u̇, p; geom, ops, work)
        prob = SecondOrderODEProblem(f!, u̇, u, (zero(T), t_end), params)
        sol  = solve(prob, KahanLi8(); dt,
                     save_everystep = false, save_start = false,
                     dense = false, save_end = true)
        final = sol.u[end]
        u̇_end = final.x[1]
        u_end = final.x[2]

        @test all(isfinite, u_end) && all(isfinite, u̇_end)
        # Energy conservation is the meaningful "no amplification"
        # test for a broad-spectrum IC: on a coercive (NSD) `L_h`, a
        # symplectic integrator keeps the discrete Hamiltonian within
        # an O(dt^p) envelope of its initial value. (Pointwise
        # `max|u|`/`max|u̇|` is *not* a good amplification check —
        # energy can redistribute from `u` into the fastest modes
        # whose `u̇` magnitude is `ω_max·‖u‖`, i.e. easily
        # 10²–10³× initial, without any instability.)
        E_end = discrete_energy(u_end, u̇_end, geom, ops, work, params.τ)
        @test abs(E_end - E0) < energy_tol * abs(E0)
    end

    _progress("robust stability — cubed cube (T=$T)")
    @testset "robust stability — cubed cube M=2, N=3, R=0.3, noise IC (T=$T)" begin
        # Same diagnostic on the curvilinear / multi-patch mesh.
        N    = 3
        M    = 2
        R    = T(0.3)
        elem = make_element(T, N)
        ops  = make_operators(elem)
        mesh = make_cubed_cube_mesh(T, M, R)
        geom = make_geometry(mesh, elem)
        work = make_workspace(geom)

        Random.seed!(20250522)
        u  = randn(T, N, N, N, mesh.Ne)
        u̇  = randn(T, N, N, N, mesh.Ne)

        params = Params3d(;
            A           = zero(T),
            k           = (zero(T), zero(T), zero(T)),
            ω           = zero(T),
            τ           = T(8) * (N - 1)^2,
            bdry_values = ntuple(_ -> zero(T), Val(6)),
        )
        dt      = recommended_dt(geom, ops, params.τ; cfl_safety = T(0.5))
        n_steps = 50
        t_end   = n_steps * dt

        E0 = discrete_energy(u, u̇, geom, ops, work, params.τ)

        f!(ü, u̇, u, p::Params3d, t) = rhs_wave3d!(ü, u, u̇, p; geom, ops, work)
        prob = SecondOrderODEProblem(f!, u̇, u, (zero(T), t_end), params)
        sol  = solve(prob, KahanLi8(); dt,
                     save_everystep = false, save_start = false,
                     dense = false, save_end = true)
        final = sol.u[end]
        u̇_end = final.x[1]
        u_end = final.x[2]

        @test all(isfinite, u_end) && all(isfinite, u̇_end)
        E_end = discrete_energy(u_end, u̇_end, geom, ops, work, params.τ)
        @test abs(E_end - E0) < energy_tol * abs(E0)
    end

    _progress("robust stability — inflated cube (T=$T)")
    @testset "robust stability — inflated cube M=2, N=3, noise IC (T=$T)" begin
        # Same diagnostic on the 13-patch inflated cube mesh: random IC
        # exercises the full spectrum of the discrete operator on the
        # curved inflation + shell patches, and energy must stay bounded
        # under a symplectic integrator on the NSD `L_h`. The outer
        # sphere `r = R2` carries homogeneous Dirichlet via
        # `bdry_values[1] = 0`.
        N    = 3
        M    = 2
        elem = make_element(T, N)
        ops  = make_operators(elem)
        mesh = make_inflated_cube_mesh(T, 1.0, 2.0, 3.0, M)
        geom = make_geometry(mesh, elem)
        work = make_workspace(geom)

        Random.seed!(20250524)
        u  = randn(T, N, N, N, mesh.Ne)
        u̇  = randn(T, N, N, N, mesh.Ne)

        params = Params3d(;
            A           = zero(T),
            k           = (zero(T), zero(T), zero(T)),
            ω           = zero(T),
            τ           = T(8) * (N - 1)^2,
            bdry_values = ntuple(_ -> zero(T), Val(6)),
        )
        dt      = recommended_dt(geom, ops, params.τ; cfl_safety = T(0.5))
        n_steps = 50
        t_end   = n_steps * dt

        E0 = discrete_energy(u, u̇, geom, ops, work, params.τ)
        @test isfinite(E0) && E0 > 0       # NSD `L_h` ⇒ `V ≥ 0`, `K ≥ 0`

        f!(ü, u̇, u, p::Params3d, t) = rhs_wave3d!(ü, u, u̇, p; geom, ops, work)
        prob = SecondOrderODEProblem(f!, u̇, u, (zero(T), t_end), params)
        sol  = solve(prob, KahanLi8(); dt,
                     save_everystep = false, save_start = false,
                     dense = false, save_end = true)
        final = sol.u[end]
        u̇_end = final.x[1]
        u_end = final.x[2]

        @test all(isfinite, u_end) && all(isfinite, u̇_end)
        E_end = discrete_energy(u_end, u̇_end, geom, ops, work, params.τ)
        @test abs(E_end - E0) < energy_tol * abs(E0)
    end

    _progress("Sommerfeld outer BC bleeds energy (T=$T)")
    @testset "Sommerfeld outer BC: energy decreases (T=$T)" begin
        # Inflated cube with `outer_bc=:sommerfeld` should bleed energy
        # through the outer sphere. A radial Bessel pulse evolved with
        # Dirichlet conserves energy (symplectic conservation, ratified
        # by the cubical robust-stability test); with Sommerfeld the
        # discrete operator becomes dissipative and the energy must
        # decrease. Coarse mesh + short integration so the test stays
        # cheap.
        N    = 3
        M    = 2
        elem = make_element(T, N)
        ops  = make_operators(elem)
        mesh = make_inflated_cube_mesh(T, T(0.1), T(0.3), T(1.0), M;
                                        outer_bc = :sommerfeld)
        geom = make_geometry(mesh, elem)
        work = make_workspace(geom)

        # Sanity: every outer face carries the new `bdry == 7` tag, and
        # no Dirichlet tag (1..6) leaked through.
        @test count(==(Int8(7)), mesh.conn.bdry) == 6 * M^2
        @test count(t -> 1 ≤ t ≤ 6, mesh.conn.bdry) == 0

        u  = Array{T, 4}(undef, N, N, N, mesh.Ne)
        u̇  = similar(u)
        eigenmode_radial!(u, u̇, geom.coords, zero(T);
                          A = one(T), R = one(T), n = 1)

        params = Params3d(;
            A           = one(T),
            k           = (T(π), T(π), T(π)),
            ω           = T(π),
            τ           = T(3//2) * (N - 1)^2,
            bdry_values = ntuple(_ -> zero(T), Val(6)),
        )
        dt    = recommended_dt(geom, ops, params.τ; cfl_safety = T(0.5))
        t_end = T(0.5)
        E0    = discrete_energy(u, u̇, geom, ops, work, params.τ)

        f!(ü, u̇, u, p::Params3d, t) = rhs_wave3d!(ü, u, u̇, p; geom, ops, work)
        prob = SecondOrderODEProblem(f!, u̇, u, (zero(T), t_end), params)
        sol  = solve(prob, Ruth3(); dt,
                     save_everystep = false, save_start = false,
                     dense = false, save_end = true)
        final = sol.u[end]
        u̇_end = final.x[1]
        u_end = final.x[2]
        @test all(isfinite, u_end) && all(isfinite, u̇_end)

        E_end = discrete_energy(u_end, u̇_end, geom, ops, work, params.τ)
        # By `t = 0.5` (half a wave-crossing time at R = 1) the pulse
        # has started bleeding out through the sphere — at least a
        # noticeable fraction of E₀ should be gone. Generous margin so
        # the test stays robust against small parameter shifts.
        @test E_end < T(0.9) * E0
        @test E_end > zero(T)        # not negative — dissipative, not blowing up
    end

    _progress("device migration round-trip (T=$T)")
    @testset "to_device round-trip on CPU backend (T=$T)" begin
        # Exercises the GPU-staging code path even though we never leave
        # the CPU: `to_device(geom, CPU())` allocates fresh backing arrays
        # via `KernelAbstractions.allocate` and `copyto!`s every kernel-
        # read field. The kernel then runs on those arrays. Result must
        # be bit-identical to the original `geom` path because nothing
        # has been promoted, demoted, or rearranged.
        N    = 3
        M    = 2
        elem = make_element(T, N)
        ops  = make_operators(elem)
        mesh = make_uniform_hex(T, M, zero(T), one(T))
        geom = make_geometry(mesh, elem)
        work = make_workspace(geom)

        u  = randn(T, N, N, N, mesh.Ne)
        u̇  = randn(T, N, N, N, mesh.Ne)
        ü_host = similar(u);  ü_dev = similar(u)

        params = Params3d(;
            A           = zero(T),
            k           = (zero(T), zero(T), zero(T)),
            ω           = zero(T),
            τ           = T(3//2) * (N - 1)^2,
            bdry_values = ntuple(_ -> zero(T), Val(6)),
        )

        geom_dev = to_device(geom, CPU())
        work_dev = to_device(work, CPU())
        @test geom_dev.coords ≈ geom.coords   # round-trip preserves data
        @test geom_dev !== geom               # …in a new container
        @test geom_dev.conn.neighbour == geom.conn.neighbour

        rhs_wave3d!(ü_host, u, u̇, params; geom,            ops, work)
        rhs_wave3d!(ü_dev,  u, u̇, params; geom = geom_dev, ops, work = work_dev)
        @test ü_host == ü_dev
    end
end

# GPU evolution on Metal. Gated on `Metal.functional()` so the test
# silently skips on machines without an Apple GPU (e.g. CI runners).
# Verifies that the full chain works end-to-end on the GPU:
#   • `to_device(geom, MetalBackend())` migrates geometry
#   • state allocated as `MtlArray{Float32}`
#   • `rhs_wave3d!` runs on Metal
#   • `recommended_dt` runs on Metal (exercises the new device-aware
#     `spectral_radius_estimate`)
#   • `discrete_inner_product` runs on Metal (exercises the new
#     `mapreduce`-based reduction)
#   • short evolve via `SecondOrderODEProblem + CandyRoz4`
# and that the final result is bit-identical to the host path.
if HAS_METAL
    @testset "GPU evolution on Metal (Float32)" begin
        _progress("Metal evolve + recommended_dt + discrete_energy")
        T    = Float32
        N    = 4
        M    = 2
        elem = make_element(T, N)
        ops  = make_operators(elem)
        mesh = make_uniform_hex(T, M, zero(T), one(T))
        geom = make_geometry(mesh, elem)
        work = make_workspace(geom)

        params = Params3d(;
            A           = one(T),
            k           = (T(2π), T(2π), T(2π)),
            ω           = T(sqrt(3 * (2π)^2)),
            τ           = T(3//2) * (N - 1)^2,
            bdry_values = ntuple(_ -> zero(T), Val(6)),
        )

        # Host reference.
        u_host  = Array{T, 4}(undef, N, N, N, mesh.Ne)
        u̇_host  = similar(u_host)
        initialize3d!(u_host, u̇_host, geom.coords, T(0), params)
        E0_host = discrete_inner_product(u̇_host, u̇_host, geom, ops) / 2
        Random.seed!(20260523)
        dt_host = recommended_dt(geom, ops, params.τ; cfl_safety = T(0.5))
        @test isfinite(dt_host) && dt_host > 0

        # Migrate geometry + state to Metal.
        backend  = MetalBackend()
        geom_dev = to_device(geom, backend)
        work_dev = to_device(work, backend)
        u_dev    = MtlArray(u_host)
        u̇_dev    = MtlArray(u̇_host)

        # `discrete_inner_product` on device → should match host.
        E0_dev = discrete_inner_product(u̇_dev, u̇_dev, geom_dev, ops) / 2
        @test isapprox(E0_dev, E0_host; rtol = sqrt(eps(T)))

        # `recommended_dt` on device → should match host within power-
        # iteration tolerance (random seed differs between backends, so
        # don't expect bit-identity, just same order of magnitude).
        Random.seed!(20260523)
        dt_dev = recommended_dt(geom_dev, ops, params.τ; cfl_safety = T(0.5))
        @test isfinite(dt_dev) && dt_dev > 0
        @test 0.5f0 * dt_host < dt_dev < 2 * dt_host

        # Short evolution. Use `dt_host` for both so the integrator does
        # exactly the same arithmetic on both backends → bit-identity.
        t_end = T(20) * dt_host
        f_host!(ü, u̇, u, p::Params3d, t) = rhs_wave3d!(ü, u, u̇, p; geom,     ops, work)
        f_dev!( ü, u̇, u, p::Params3d, t) = rhs_wave3d!(ü, u, u̇, p; geom = geom_dev, ops, work = work_dev)

        prob_host = SecondOrderODEProblem(f_host!, u̇_host, u_host, (T(0), t_end), params)
        prob_dev  = SecondOrderODEProblem(f_dev!,  u̇_dev,  u_dev,  (T(0), t_end), params)

        sol_host = solve(prob_host, CandyRoz4(); dt = dt_host,
                         save_everystep = false, save_start = false,
                         dense = false, save_end = true)
        sol_dev  = solve(prob_dev,  CandyRoz4(); dt = dt_host,
                         save_everystep = false, save_start = false,
                         dense = false, save_end = true)

        u_host_end = sol_host.u[end].x[2]
        u_dev_end  = Array(sol_dev.u[end].x[2])
        @test all(isfinite, u_host_end)
        @test all(isfinite, u_dev_end)
        @test u_host_end == u_dev_end   # bit-identical
    end
end
