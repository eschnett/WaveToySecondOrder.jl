using BenchmarkTools, LinearAlgebra, StaticArrays
using WaveToySecondOrder
const W = WaveToySecondOrder

println("=== rhs3d! (curvilinear-aware, geom-driven) — N=5, varying mesh ===")
N    = 5
elem = W.make_element(Float64, N)
ops  = W.make_operators(elem)
τ    = 1.5 * (N - 1)^2
bdry_values = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

# Cubical sweep
println("\n--- cubical mesh ---")
for M in (2, 4, 8, 12)
    mesh = W.make_cubical_mesh(Float64, M, 0.0, 1.0)
    geom = W.make_geometry(mesh, elem)
    u  = randn(N, N, N, mesh.Ne)
    u̇  = randn(N, N, N, mesh.Ne)
    ü  = similar(u)

    b = @benchmark W.rhs3d!($ü, $u, $u̇, $bdry_values; geom=$geom, ops=$ops, τ=$τ)
    t_per_elt = minimum(b.times) / mesh.Ne
    println("  M=$M  ($(mesh.Ne) elements)  total=$(round(minimum(b.times)/1e3, digits=1)) μs  ",
            "per-element=$(round(t_per_elt, digits=1)) ns  allocs=$(b.allocs)  bytes=$(b.memory)")
end

# Inflated-cube sweep (M = patch subdivisions, independent of N)
println("\n--- inflated cube ---")
for (M, R) in ((4, 0.1), (8, 0.1), (4, 0.3))
    mesh = W.make_inflated_cube_mesh(Float64, M, R)
    geom = W.make_geometry(mesh, elem)
    u  = randn(N, N, N, mesh.Ne)
    u̇  = randn(N, N, N, mesh.Ne)
    ü  = similar(u)

    b = @benchmark W.rhs3d!($ü, $u, $u̇, $bdry_values; geom=$geom, ops=$ops, τ=$τ)
    t_per_elt = minimum(b.times) / mesh.Ne
    println("  M=$M  R=$R  ($(mesh.Ne) elements)  total=$(round(minimum(b.times)/1e3, digits=1)) μs  ",
            "per-element=$(round(t_per_elt, digits=1)) ns  allocs=$(b.allocs)  bytes=$(b.memory)")
end
