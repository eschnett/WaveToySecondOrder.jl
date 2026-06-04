@testitem "curvilinear_types" tags=[:cpu] begin
    _progress(m) = (printstyled(stderr, "  • ", m, "\n"; color = :cyan); flush(stderr))

# Type-genericity of the curvilinear 2D path (cubed-square): the
# free-stream-preserving operators and the curved Dirichlet BC run at
# Float64x2 (MultiFloats, CPU) and Float32. The HexMeshes cubed-square
# builder is type-generic (patch counts computed in Float64). Mirrors
# the affine Float64x2 test in wave2d_tests.jl.

using HexMeshes: make_cubed_square_mesh
using HexSBPSAT: make_element, make_operators, make_geometry,
                 make_metric_terms2d, apply_gradient2d!
using MultiFloats
using Test
using WaveToySecondOrder: AnalyticBackground2D, make_coef2d,
                          sample_background2d!, make_wave2d_workspace,
                          wave2d_curved_rhs!, make_bc2d

# Short curved-Dirichlet RK4 evolution; exact plane-wave data built in
# Float64 and converted to T (MultiFloat trig is unreliable). Returns
# the final Φ as Float64 for cross-type comparison.
function _run_curv(::Type{T}; nsteps = 40) where {T}
    N = 4
    mesh = make_cubed_square_mesh(T, 2, T(0.3))
    elem = make_element(T, N); ops = make_operators(elem)
    geom = make_geometry(mesh, elem); metric = make_metric_terms2d(geom, ops)
    ws = make_wave2d_workspace(geom, ops); coef = make_coef2d(geom); Ne = geom.Ne
    flat = AnalyticBackground2D((t,x,y)->one(T), (t,x,y)->(zero(T),zero(T)),
                                (t,x,y)->(one(T),zero(T),one(T)))
    xg = geom.coords[1,:,:,:]; yg = geom.coords[2,:,:,:]
    x64 = Float64.(xg); y64 = Float64.(yg)
    κ = 2.0; ω = κ*sqrt(2.0); s64 = κ .* (x64 .+ y64)
    Φ0(t) = sin.(s64 .- ω*t); Π0(t) = (-ω) .* cos.(s64 .- ω*t)
    D0(t)  = κ .* cos.(s64 .- ω*t)
    Φ = T.(Φ0(0.0)); Π = T.(Π0(0.0))
    k = [similar(Φ) for _ in 1:8]; Φs = similar(Φ); Πs = similar(Π)
    dt = T(1) / 512
    # Build the Dirichlet data at the Float64 stage time (the closures
    # use cos, which MultiFloats does not provide), then convert to T.
    rhs(a,b,c,d,t) = begin
        sample_background2d!(coef, flat, t, xg, yg)
        tf = Float64(t)
        bc = make_bc2d((:dirichlet,:dirichlet,:dirichlet,:dirichlet);
                       gΠ = T.(Π0(tf)), gDx = T.(D0(tf)), gDy = T.(D0(tf)))
        wave2d_curved_rhs!(a,b,c,d, coef; geom, ops, ws, ε_KO=T(0.05),
                           bc2d=bc, metric)
    end
    t = zero(T)
    for _ in 1:nsteps
        rhs(k[1],k[2],Φ,Π,t);            @. Φs=Φ+dt/2*k[1]; @. Πs=Π+dt/2*k[2]
        rhs(k[3],k[4],Φs,Πs,t+dt/2);     @. Φs=Φ+dt/2*k[3]; @. Πs=Π+dt/2*k[4]
        rhs(k[5],k[6],Φs,Πs,t+dt/2);     @. Φs=Φ+dt*k[5];   @. Πs=Π+dt*k[6]
        rhs(k[7],k[8],Φs,Πs,t+dt)
        @. Φ += dt/6*(k[1]+2k[3]+2k[5]+k[7]); @. Π += dt/6*(k[2]+2k[4]+2k[6]+k[8])
        t += dt
    end
    return Float64.(Φ)
end

@testset "curvilinear type-genericity" begin
    _progress("free-stream at Float64x2 and Float32")
    @testset "free-stream type-generic" begin
        for (T, tol) in ((Float64x2, 1e-12), (Float32, 1e-3))
            mesh = make_cubed_square_mesh(T, 2, T(0.3))
            elem = make_element(T, 4); ops = make_operators(elem)
            geom = make_geometry(mesh, elem); metric = make_metric_terms2d(geom, ops)
            Ne = geom.Ne
            g1 = zeros(T,4,4,Ne); g2 = similar(g1)
            apply_gradient2d!(g1, g2, fill(T(5)/2, 4,4,Ne); geom, ops, metric)
            @test Float64(maximum(abs, g1)) ≤ tol
            @test Float64(maximum(abs, g2)) ≤ tol
        end
    end

    _progress("Float64x2 trajectory matches Float64")
    @testset "Float64x2 ≈ Float64 (curved Dirichlet)" begin
        Φ64  = _run_curv(Float64)
        Φx2  = _run_curv(Float64x2)
        @test all(isfinite, Φ64) && all(isfinite, Φx2)
        @test maximum(abs, Φ64 .- Φx2) < 1e-11
    end

    _progress("Float32 evolution finite & bounded")
    @testset "Float32 curvilinear run is finite" begin
        Φ32 = _run_curv(Float32)
        @test all(isfinite, Φ32)
        @test maximum(abs, Φ32) < 10
    end
end

end
