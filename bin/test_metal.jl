using WaveToySecondOrder
using KernelAbstractions
using Metal
using OrdinaryDiffEqSymplecticRK
using StaticArrays

const W = WaveToySecondOrder

T = Float32                        # Apple GPU constraint
N = 4
M = 4

println("=== Stage 1: host-side build ===")
elem = W.make_element(T, N)
ops  = W.make_operators(elem)
mesh = W.make_cubical_mesh(T, M, T(0), T(1))
geom = W.make_geometry(mesh, elem)
params = W.Params3d(;
    A           = one(T),
    k           = (T(2π), T(2π), T(2π)),
    ω           = T(sqrt(3 * (2π)^2)),
    τ           = T(3//2) * (N - 1)^2,
    bdry_values = ntuple(_ -> zero(T), Val(6)),
)
println("  Ne = ", mesh.Ne, "  T = ", T)

u_host  = randn(T, N, N, N, mesh.Ne)
u̇_host  = randn(T, N, N, N, mesh.Ne)
ü_host  = similar(u_host)
W.rhs_wave3d!(ü_host, u_host, u̇_host, params; geom, ops)
println("  CPU rhs_wave3d! ok, max|ü| = ", maximum(abs, ü_host))

println("\n=== Stage 2: migrate geometry to Metal ===")
backend = MetalBackend()
geom_dev = W.to_device(geom, backend)
println("  geom_dev.coords type:        ", typeof(geom_dev.coords))
println("  geom_dev.conn.neighbour type: ", typeof(geom_dev.conn.neighbour))

println("\n=== Stage 3: state on Metal ===")
u_dev  = MtlArray(u_host)
u̇_dev  = MtlArray(u̇_host)
ü_dev  = similar(u_dev)
@assert eltype(u_dev) === T
println("  u_dev type:  ", typeof(u_dev))

println("\n=== Stage 4: rhs_wave3d! single call on Metal ===")
W.rhs_wave3d!(ü_dev, u_dev, u̇_dev, params; geom = geom_dev, ops)
KernelAbstractions.synchronize(backend)
ü_back = Array(ü_dev)
println("  device max|ü| = ", maximum(abs, ü_back))
println("  host   max|ü| = ", maximum(abs, ü_host))
maxdiff = maximum(abs, ü_back .- ü_host)
println("  max |Δü|      = ", maxdiff,
        "   relative = ", maxdiff / maximum(abs, ü_host))

# Float32 round-off floor for this kernel is ~few×eps(Float32)·‖ü‖.
# A few hundredths of a percent relative error is the expected zone.
@assert maxdiff < 1f-3 * maximum(abs, ü_host) "device result deviates by more than 0.1%"
println("\n  >>> Metal single-call rhs_wave3d! matches CPU within Float32 round-off <<<")

println("\n=== Stage 5: SecondOrderODEProblem on Metal (short evolve) ===")
# Short integration with the analytic separable eigenmode IC. Goal:
# verify that OrdinaryDiffEqSymplecticRK runs unchanged on `MtlArray`
# state. We use the same setup as the wave-evolution test but in
# Float32 with t1 = 0.1 (a small fraction of a period) to keep the
# call short. Pass: the integrator finishes, all entries finite, and
# the host vs Metal result agrees to Float32 round-off.

W.initialize3d!(u_host, u̇_host, geom.coords, T(0), params)
copyto!(u_dev, u_host)
copyto!(u̇_dev, u̇_host)

dx = elem.h * (one(T) / M)
dt = (T(1//2) * dx) / sqrt(T(3))
t1 = T(0.1f0)

f_host!(ü, u̇, u, p::W.Params3d, t) = W.rhs_wave3d!(ü, u, u̇, p; geom = geom,     ops)
f_dev!(ü,  u̇, u, p::W.Params3d, t) = W.rhs_wave3d!(ü, u, u̇, p; geom = geom_dev, ops)

prob_host = SecondOrderODEProblem(f_host!, u̇_host, u_host, (T(0), t1), params)
prob_dev  = SecondOrderODEProblem(f_dev!,  u̇_dev,  u_dev,  (T(0), t1), params)

println("  solving on host (CPU) ...")
sol_host = solve(prob_host, CandyRoz4(); dt,
                 save_everystep = false, save_start = false,
                 dense = false, save_end = true)

println("  solving on device (Metal) ...")
sol_dev  = solve(prob_dev,  CandyRoz4(); dt,
                 save_everystep = false, save_start = false,
                 dense = false, save_end = true)

u_host_end = sol_host.u[end].x[2]
u_dev_end  = Array(sol_dev.u[end].x[2])

@assert all(isfinite, u_host_end)
@assert all(isfinite, u_dev_end)

reldiff = maximum(abs, u_host_end .- u_dev_end) / maximum(abs, u_host_end)
println("  ‖u_host − u_metal‖_∞ / ‖u_host‖_∞ = ", reldiff)
@assert reldiff < 1f-3 "Metal evolve deviates from host beyond Float32 envelope"
println("\n  >>> SecondOrderODEProblem + CandyRoz4 on Metal: OK <<<")
