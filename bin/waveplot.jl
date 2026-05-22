using CairoMakie
using OrdinaryDiffEqSymplecticRK
using SixelTerm
using WaveToySecondOrder

const W = WaveToySecondOrder

# Evolve

x0 = 0.0                        # domain
x1 = 1.0
M  = 8                          # number of elements
N  = 17                         # number of points per element

elem = W.make_element(Float64, N)
ops  = W.make_operators(elem)
dom  = W.make_domain(Float64, M, x0, x1)

x  = [x + dom.h * a for a in elem.xs, x in dom.xs]
dx = dom.h * elem.h

u  = similar(x)
u̇  = similar(x)

t0, t1 = 0.0, 1.0
A      = 1.0
k      = 2π
ω      = sqrt(k^2)
W.initialize!(u, u̇, x, t0; A, k, ω)

bL, bR = 0.0, 0.0
τ      = 3//2 * (N-1)^2         # SIPG threshold ~ 2·(N−1)² in unit coords
dt     = 1//2 * dx              # CFL-safe for KahanLi8

f!(ü, u̇, u, p, t) = W.rhs!(ü, u, u̇, bL, bR; dom, ops, τ)
prob = SecondOrderODEProblem(f!, u̇, u, (t0, t1))
sol  = solve(prob, KahanLi8(); dt)

# Figure

fig = Figure(; size=(800, 450))
ax1 = Axis(fig[1, 1]; aspect=DataAspect(), title="u",  xlabel="x", ylabel="t")
ax2 = Axis(fig[1, 3]; aspect=DataAspect(), title="u̇", xlabel="x", ylabel="t")

xs = vec(x)
Ng = length(xs)
ts = range(t0, t1, Ng)
us1 = Array{Float64}(undef, length(xs), length(ts))
us2 = Array{Float64}(undef, length(xs), length(ts))
for (n, t) in enumerate(ts)
    solt = sol(t)
    @assert all(isfinite, solt)
    us1[:, n] .= vec(solt[Ng+1:2Ng])
    us2[:, n] .= vec(solt[1:Ng])
end
hm1 = heatmap!(ax1, xs, ts, us1; colormap=:plasma)
hm2 = heatmap!(ax2, xs, ts, us2; colormap=:plasma)
Colorbar(fig[1, 2], hm1)
Colorbar(fig[1, 4], hm2)

display(fig)

nothing
