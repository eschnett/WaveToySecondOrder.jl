# Exploratory: 2D quadrilateral mesh on [−1, 1]² with the resolution at the
# centre 10× higher than at the boundary. Generates a tensor-product mesh
# with non-uniform 1D node distribution and plots it.
#
# Stretching function: x = sinh(α·ξ) / sinh(α), ξ ∈ [−1, 1] uniform.
# Local cell width is dx ∝ cosh(α·ξ), so
#     dx(±1) / dx(0) = cosh(α).
# We choose α = acosh(10) ≈ 2.993 to hit the 10× ratio exactly.
#
# The mesh is structured: every cell is an axis-aligned rectangle whose
# size varies smoothly with position. The aspect ratio is 1 everywhere
# (same stretching applied to x and y), so the cells "look reasonable" —
# small squares in the centre, larger squares near the boundary.

using CairoMakie
using SixelTerm

# ----- parameters ----------------------------------------------------------

target_ratio = 10.0             # edge cell size / centre cell size
α            = acosh(target_ratio)
M            = 24               # cells per axis
ξ            = range(-1, 1, M + 1)

stretch(ξ) = sinh(α * ξ) / sinh(α)
xs = stretch.(ξ)
ys = stretch.(ξ)

# ----- sanity check --------------------------------------------------------

Δx = diff(xs)
println("cells per axis      : $M")
println("min cell width      : $(round(minimum(Δx), digits=5))")
println("max cell width      : $(round(maximum(Δx), digits=5))")
println("max/min ratio (edge/centre) : $(round(maximum(Δx)/minimum(Δx), digits=3))")

# ----- plot ----------------------------------------------------------------

fig = Figure(; size=(700, 700))
ax  = Axis(fig[1, 1];
           aspect=DataAspect(),
           title="$(M)×$(M) quad mesh — edge/centre resolution ratio $(target_ratio)",
           xlabel="x", ylabel="y")

# Mesh lines: verticals at constant x, horizontals at constant y.
for i in eachindex(xs)
    lines!(ax, [xs[i], xs[i]], [ys[begin], ys[end]];
           color=(:steelblue, 0.7), linewidth=0.8)
end
for j in eachindex(ys)
    lines!(ax, [xs[begin], xs[end]], [ys[j], ys[j]];
           color=(:steelblue, 0.7), linewidth=0.8)
end

# Vertices.
vx = vec([xs[i] for i in eachindex(xs), _ in eachindex(ys)])
vy = vec([ys[j] for _ in eachindex(xs), j in eachindex(ys)])
scatter!(ax, vx, vy; markersize=3, color=:black)

display(fig)

nothing
