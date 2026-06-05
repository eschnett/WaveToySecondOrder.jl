@testitem "wave3d" tags=[:cpu] begin
    _progress(m) = (printstyled(stderr, "  • ", m, "\n"; color = :cyan); flush(stderr))

# Conservative first-order 3D scalar wave on a 3+1 ADM background,
# axis-aligned affine hex (Milestone 1): RHS / energy / boundary pass.
# Mirrors wave2d_tests.jl.

using HexMeshes: make_uniform_hex, make_cubed_cube_mesh, make_inflated_cube_mesh,
                 make_radial_shell_mesh
using HexSBPSAT: make_element, make_operators, make_geometry, make_metric_terms3d,
                 make_workspace
using LinearAlgebra, Random, Test
using WaveToySecondOrder: AnalyticBackground3D, make_coef3d,
                          sample_background3d!, make_wave3d_workspace,
                          wave3d_curved_rhs!, wave3d_energy, make_bc3d,
                          classify_face3d, FACE_SUBLUMINAL

_flat3d(::Type{T}) where {T} =
    AnalyticBackground3D((t,x,y,z)->one(T), (t,x,y,z)->(zero(T),zero(T),zero(T)),
                         (t,x,y,z)->(one(T),zero(T),zero(T),one(T),zero(T),one(T)))

function _setup3d(::Type{T}, N, M; periodic = true) where {T}
    mesh = make_uniform_hex(T, M, T(0), T(1); periodic = periodic)
    elem = make_element(T, N); ops = make_operators(elem)
    geom = make_geometry(mesh, elem)
    ws = make_wave3d_workspace(geom, ops); coef = make_coef3d(geom)
    xg = geom.coords[1,:,:,:,:]; yg = geom.coords[2,:,:,:,:]; zg = geom.coords[3,:,:,:,:]
    dxmin = minimum(diff(elem.xs)) * (T(1)/M)
    return (; mesh, elem, ops, geom, ws, coef, xg, yg, zg, dxmin)
end

function _rk4_3d!(Φ, Π, dt, nst, bg, s; ε_KO, bc3d, k, Φs, Πs)
    rhs(a,b,c,d) = begin
        sample_background3d!(s.coef, bg, 0.0, s.xg, s.yg, s.zg)
        wave3d_curved_rhs!(a,b,c,d, s.coef; s.geom, s.ops, s.ws, ε_KO, bc3d)
    end
    for _ in 1:nst
        rhs(k[1],k[2],Φ,Π);          @. Φs=Φ+dt/2*k[1]; @. Πs=Π+dt/2*k[2]
        rhs(k[3],k[4],Φs,Πs);        @. Φs=Φ+dt/2*k[3]; @. Πs=Π+dt/2*k[4]
        rhs(k[5],k[6],Φs,Πs);        @. Φs=Φ+dt*k[5];   @. Πs=Π+dt*k[6]
        rhs(k[7],k[8],Φs,Πs)
        @. Φ += dt/6*(k[1]+2k[3]+2k[5]+k[7]); @. Π += dt/6*(k[2]+2k[4]+2k[6]+k[8])
    end
end

@testset "3D conservative wave (affine)" begin
    T = Float64; N = 4

    _progress("3D periodic energy conservation + convergence")
    @testset "periodic plane wave: energy + convergence" begin
        bg = _flat3d(T); κ = 2π; ω = κ*sqrt(3.0)
        Φe(t,x,y,z) =  sin(κ*(x+y+z) - ω*t)
        Πe(t,x,y,z) = -ω*cos(κ*(x+y+z) - ω*t)
        errs = T[]
        for M in (4, 8)
            s = _setup3d(T, N, M; periodic = true)
            sample_background3d!(s.coef, bg, 0.0, s.xg, s.yg, s.zg)
            Φ = Φe.(0.0, s.xg, s.yg, s.zg); Π = Πe.(0.0, s.xg, s.yg, s.zg)
            E0 = wave3d_energy(Φ, Π, s.coef; s.geom, s.ops, s.ws)
            k = [similar(Φ) for _ in 1:8]; Φs = similar(Φ); Πs = similar(Π)
            dt = T(0.1)*s.dxmin; nst = ceil(Int, 0.2/dt)
            _rk4_3d!(Φ, Π, dt, nst, bg, s; ε_KO=0.0, bc3d=nothing, k, Φs, Πs)
            E1 = wave3d_energy(Φ, Π, s.coef; s.geom, s.ops, s.ws)
            @test abs(E1/E0 - 1) < 1e-3
            push!(errs, sqrt(sum(@. (Φ - Φe(nst*dt, s.xg, s.yg, s.zg))^2 * s.geom.Hphys)))
        end
        @test all(isfinite, errs)
        @test errs[1]/errs[2] > 4          # ~3rd-order over a 2× refinement
    end

    _progress("3D RHS spectrum stable (periodic + Sommerfeld)")
    @testset "spectrum: max Re(λ) ≤ round-off" begin
        bg = _flat3d(T)
        for (lbl, periodic, bc) in (("periodic", true, nothing),
                                    ("sommerfeld", false,
                                     make_bc3d(ntuple(_->:sommerfeld, 6))))
            s = _setup3d(T, N, 2; periodic = periodic)
            sample_background3d!(s.coef, bg, 0.0, s.xg, s.yg, s.zg)
            Ne = s.geom.Ne; nn = N^3*Ne; n = 2nn; A = zeros(T,n,n)
            Φ = zeros(T,N,N,N,Ne); Π=similar(Φ); Φ̇=similar(Φ); Π̇=similar(Φ)
            for jc in 1:n
                fill!(Φ,0); fill!(Π,0); jc ≤ nn ? (Φ[jc]=1) : (Π[jc-nn]=1)
                wave3d_curved_rhs!(Φ̇,Π̇,Φ,Π,s.coef; s.geom,s.ops,s.ws,ε_KO=0.1,bc3d=bc)
                A[1:nn,jc]=vec(Φ̇); A[nn+1:end,jc]=vec(Π̇)
            end
            λ = eigvals(A)
            @test maximum(real, λ) ≤ 1e-5 * maximum(abs, λ)
        end
    end

    _progress("3D noise robustness (Sommerfeld + KO)")
    @testset "noise bounded (Sommerfeld)" begin
        bg = _flat3d(T); s = _setup3d(T, N, 2; periodic = false)
        Random.seed!(20260604); amp = sqrt(eps(T))
        Φ = amp.*randn(T,N,N,N,s.geom.Ne); Π = amp.*randn(T,N,N,N,s.geom.Ne)
        bc = make_bc3d(ntuple(_->:sommerfeld, 6))
        k = [similar(Φ) for _ in 1:8]; Φs=similar(Φ); Πs=similar(Π)
        dt = T(0.1)*s.dxmin; nst = ceil(Int, 8/dt)
        _rk4_3d!(Φ, Π, dt, nst, bg, s; ε_KO=0.1, bc3d=bc, k, Φs, Πs)
        @test all(isfinite, Φ) && maximum(abs, Φ) < 1000*amp
    end

    _progress("3D face classification (flat ⇒ subluminal)")
    @testset "classify_face3d" begin
        @test classify_face3d(1.0,0.0,0.0,0.0,1.0,1.0,1.0, 1, 1) == FACE_SUBLUMINAL
        @test classify_face3d(1.0,0.0,0.0,0.0,1.0,1.0,1.0, 3, -1) == FACE_SUBLUMINAL
    end

    _progress("3D curvilinear: free-stream RHS + curved Sommerfeld spectrum")
    @testset "curvilinear RHS free-stream + spectrum" begin
        bg = _flat3d(T)
        meshes = (("cubed_cube",   make_cubed_cube_mesh(T, 2, T(0.3))),
                  ("inflated_cube", make_inflated_cube_mesh(T, T(0.2), T(0.5), T(1.0), 1)),
                  ("radial_shell",  make_radial_shell_mesh(T, T(0.5), T(1.0), 2)))
        for (_, mesh) in meshes
            elem = make_element(T, N); ops = make_operators(elem)
            geom = make_geometry(mesh, elem); metric = make_metric_terms3d(geom, ops)
            ws = make_wave3d_workspace(geom, ops); coef = make_coef3d(geom); Ne = geom.Ne
            xg=geom.coords[1,:,:,:,:]; yg=geom.coords[2,:,:,:,:]; zg=geom.coords[3,:,:,:,:]
            sample_background3d!(coef, bg, 0.0, xg, yg, zg)
            Φ = fill(T(2.5),N,N,N,Ne); Π = zeros(T,N,N,N,Ne); Φ̇=similar(Φ); Π̇=similar(Π)
            bc = make_bc3d(ntuple(_->:sommerfeld,6))
            wave3d_curved_rhs!(Φ̇,Π̇,Φ,Π,coef; geom,ops,ws,ε_KO=0.0,bc3d=bc,metric)
            @test maximum(abs,Φ̇) ≤ 1e-9 && maximum(abs,Π̇) ≤ 1e-9   # free-stream
        end
        # Curved Sommerfeld spectrum stable on the cubed-cube (flat outer).
        mesh = make_cubed_cube_mesh(T, 1, T(0.3))
        elem = make_element(T, N); ops = make_operators(elem)
        geom = make_geometry(mesh, elem); metric = make_metric_terms3d(geom, ops)
        ws = make_wave3d_workspace(geom, ops); coef = make_coef3d(geom); Ne = geom.Ne
        sample_background3d!(coef, bg, 0.0, geom.coords[1,:,:,:,:],
                             geom.coords[2,:,:,:,:], geom.coords[3,:,:,:,:])
        bc = make_bc3d(ntuple(_->:sommerfeld,6))
        nn = N^3*Ne; n = 2nn; A = zeros(T,n,n)
        Φ=zeros(T,N,N,N,Ne); Π=similar(Φ); Φ̇=similar(Φ); Π̇=similar(Φ)
        for jc in 1:n
            fill!(Φ,0); fill!(Π,0); jc ≤ nn ? (Φ[jc]=1) : (Π[jc-nn]=1)
            wave3d_curved_rhs!(Φ̇,Π̇,Φ,Π,coef; geom,ops,ws,ε_KO=0.1,bc3d=bc,metric)
            A[1:nn,jc]=vec(Φ̇); A[nn+1:end,jc]=vec(Π̇)
        end
        @test maximum(real, eigvals(A)) ≤ 1e-5 * maximum(abs, eigvals(A))
    end
end

end
