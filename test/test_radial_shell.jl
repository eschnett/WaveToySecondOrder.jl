using HexMeshes: make_radial_shell_mesh, Shell
using HexSBPSAT: make_element, make_operators, make_geometry
using Test
using WaveToySecondOrder: wave_strong_rhs_mesh!, evolve3d

# Smoke tests for the new 6-patch radial-shell mesh (BH-excision use
# case) and the new `bdry == 8` excision branch in the strong-form
# scalar kernel. The shell mesh has no inner cube and no inflation
# layer — just six `PatchShell` patches with constant radial element
# spacing covering R1 ≤ |x| ≤ R2.

@testset "Radial shell mesh + excision BC (scalar wave)" begin

    _progress("radial shell mesh sanity")
    @testset "mesh sanity: 6 patches, tags {0, 1, 8}, all Shell" begin
        T = Float64
        mesh = make_radial_shell_mesh(T, T(0.3), T(1.0), 4)
        @test mesh.Ne > 0
        @test length(mesh.patch_desc) == 6
        @test all(pd.kind === Shell for pd in mesh.patch_desc)
        # Default (outer = :dirichlet, inner = :excision) →
        # tags {0, 1, 8}.
        @test sort(unique(mesh.conn.bdry)) == Int8[0, 1, 8]
    end

    _progress("radial shell kernel smoke: finite ü")
    @testset "wave_strong_rhs_mesh! on shell mesh is finite" begin
        T = Float64; N = 4
        mesh = make_radial_shell_mesh(T, T(0.3), T(1.0), 2)
        elem = make_element(T, N); ops = make_operators(elem)
        geom = make_geometry(mesh, elem)
        u = Array{T, 4}(undef, N, N, N, mesh.Ne)
        # Smooth field: u = sin(πr) — analytic radial wave, vanishes at
        # r = 1 (outer Dirichlet wall) but not at r = 0.3 (excision).
        for e in 1:mesh.Ne, kk in 1:N, jj in 1:N, ii in 1:N
            x = geom.coords[1, ii, jj, kk, e]
            y = geom.coords[2, ii, jj, kk, e]
            z = geom.coords[3, ii, jj, kk, e]
            r = sqrt(x*x + y*y + z*z)
            u[ii, jj, kk, e] = sin(π * r)
        end
        u̇ = zeros(T, N, N, N, mesh.Ne); ü = similar(u)
        wave_strong_rhs_mesh!(ü, u, u̇, mesh, geom, ops)
        @test all(isfinite, ü)
        @test maximum(abs, ü) > 0      # non-trivial output
    end

    _progress("evolve3d :radial_shell drives to completion")
    @testset "evolve3d :radial_shell (outgoing pulse, default excision inner)" begin
        # Short driver run. With ic_kind = :outgoing the Gaussian pulse
        # is centred at the origin (`ic_center = (0, 0, 0)`) with
        # default `ic_pulse_offset = L_/4 = 0.5` and width `s0/5 = 0.1`,
        # both inside the shell R1 = 0.3 .. R2 = 1.0.
        res = evolve3d(; mesh_kind = :radial_shell,
                         ic_kind   = :outgoing,
                         R1 = 0.3, R2 = 1.0, M = 2, N = 4,
                         t1 = 0.05, Nt = 5)
        @test all(isfinite, res.u_final)
        @test res.mesh_kind === :radial_shell
        # Mesh carries the excision tag 8 on the inner face.
        @test Int8(8) in res.mesh.conn.bdry
    end

    _progress("evolve3d :radial_shell honours inner_bc kwarg")
    @testset "inner_bc = :dirichlet swaps tag 8 for tag 2" begin
        res = evolve3d(; mesh_kind = :radial_shell,
                         ic_kind   = :outgoing,
                         R1 = 0.3, R2 = 1.0, M = 2, N = 4,
                         inner_bc  = :dirichlet,
                         t1 = 0.05, Nt = 5)
        @test all(isfinite, res.u_final)
        tags = sort(unique(res.mesh.conn.bdry))
        @test Int8(8) ∉ tags                    # no excision
        @test Int8(2) ∈ tags                    # inner Dirichlet tag
    end

end
