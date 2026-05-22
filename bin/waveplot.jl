using CairoMakie
using SixelTerm
using WaveToySecondOrder

W = WaveToySecondOrder

# Evolve

x0 = 0                          # domain
x1 = 1
M = 8                           # number of elements
N = 17                          # number of points per element
res = W.evolve(x0, x1, M, N)
t0, t1, x, sol = res.t0, res.t1, res.x, res.sol

# Figure

fig = Figure(; size=(800, 450))
ax1 = Axis(fig[1, 1]; aspect=DataAspect(), title="u", xlabel="x", ylabel="t")
ax2 = Axis(fig[1, 3]; aspect=DataAspect(), title="u̇", xlabel="x", ylabel="t")

xs = vec(x)
N = length(xs)
ts = range(t0, t1, N)
us1 = Array{Float64}(undef, length(xs), length(ts))
us2 = Array{Float64}(undef, length(xs), length(ts))
for (n,t) in enumerate(ts)
    solt = sol(t)
    @assert all(isfinite, solt)
    us1[:,n] .= vec(solt[N+1:2N])
    us2[:,n] .= vec(solt[1:N])
end
hm1 = heatmap!(ax1, xs, ts, us1; colormap=:plasma)
hm2 = heatmap!(ax2, xs, ts, us2; colormap=:plasma)
Colorbar(fig[1, 2], hm1)
Colorbar(fig[1, 4], hm2)

display(fig)

nothing
