using BenchmarkTools, LinearAlgebra, StaticArrays
using WaveToySecondOrder
const W = WaveToySecondOrder

println("=== rhs3d! (4D state, mesh-driven) — N=5, varying M ===")
N = 5
elem = W.make_element(Float64, N)
ops  = W.make_operators(elem)
for M in (2, 4, 8, 12)
    mesh = W.make_cubical_mesh(Float64, M, 0.0, 1.0)
    u  = randn(N, N, N, mesh.Ne)
    u̇  = randn(N, N, N, mesh.Ne)
    ü  = similar(u)
    τ  = 1.5 * (N - 1)^2
    bdry_values = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    b = @benchmark W.rhs3d!($ü, $u, $u̇, $bdry_values; mesh=$mesh, ops=$ops, τ=$τ)
    t_per_elt = minimum(b.times) / mesh.Ne
    println("  M=$M  ($(mesh.Ne) elements)  total=$(round(minimum(b.times)/1e3, digits=1)) μs  ",
            "per-element=$(round(t_per_elt, digits=1)) ns  allocs=$(b.allocs)  bytes=$(b.memory)")
end
