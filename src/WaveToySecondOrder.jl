module WaveToySecondOrder

using FastGaussQuadrature
using LinearAlgebra
using PolynomialBases: LobattoLegendre
using StaticArrays

# Discrete SBP-SAT operators on a single element, plus the SAT increment
# primitive `_sat_increment` used by both the 1D and 3D kernels.
include("operators.jl")

# Per-element and global 1D kernels (`initialize!`, `apply_laplacian!`,
# `rhs!`, plus the diagnostic `build_global_laplacian`).
include("kernels1d.jl")

# Per-element and global 3D kernels (`initialize3d!`, `apply_laplacian3d!`,
# `rhs3d!`, plus axis abstractions and face-data helpers).
include("kernels3d.jl")

end
