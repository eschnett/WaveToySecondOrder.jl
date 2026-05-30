module WaveToySecondOrder

# Three-layer package stack:
#
#   HexMeshes  → topology, parametric geometry, host-side queries.
#   HexSBPSAT  → SBP-SAT operators, per-node MeshGeometry, the 3D
#                discrete Laplacian `apply_laplacian!`, and diagnostics.
#   WaveToy    → the wave equation itself: IC, eigenmodes, Sommerfeld
#                radiative BC, recommended_dt, time-evolution driver.
#
# All three live in the same family of packages and import their lower-
# level neighbours explicitly. The kernel hot path is generic-enough that
# the wave-specific layer here is just a thin wrapper around
# `apply_laplacian!` with one extra dissipative kernel pass.

using HexMeshes
using HexMeshes: Mesh, MeshConnectivity,
                 PatchDesc, PatchKind, Cubic, Wedge, Inflation, Shell,
                 make_uniform_line, make_uniform_quad, make_uniform_hex,
                 make_cubed_square_mesh, make_inflated_square_mesh,
                 make_cubed_cube_mesh, make_inflated_cube_mesh,
                 nv, npatches, element_vertices,
                 locate_point, invert_element_map, interpolate_field,
                 trilinear_shape, trilinear_dshape,
                 trilinear_map, trilinear_jacobian,
                 lagrange_basis, tensor_interp
using HexSBPSAT
using HexSBPSAT: SBPOps, MeshGeometry, MeshWorkspace,
                 make_element, make_domain, make_operators, make_geometry,
                 make_workspace,
                 element_coords, to_device,
                 apply_laplacian!,
                 build_global_laplacian, discrete_laplacian,
                 spectral_radius_estimate,
                 discrete_inner_product, discrete_l2_norm,
                 physical_mass_diagonal
# Internal helpers from HexSBPSAT used by `wave.jl`'s 3D Sommerfeld pass.
using HexSBPSAT: _face_axis_idx, _face_row, _face_sign, _cross_sign,
                 _tangent_axes, _ijk_from_li
# Internal helpers used by `wave2d.jl`'s 2D Sommerfeld pass.
using HexSBPSAT: _face_axis_idx_2d, _face_row_2d, _face_sign_2d,
                 _cross_sign_2d, _tangent_axis_2d, _ij_from_li

using FastGaussQuadrature: gausslegendre
using KernelAbstractions
using KernelAbstractions: @kernel, @index, @Const, get_backend, CPU
using LinearAlgebra: eigvals
using OrdinaryDiffEqSymplecticRK
using ProgressMeter
using SpecialFunctions: besselj, sphericalbesselj
using StaticArrays

# `wave.jl`: 1D + 3D wave-equation layer. 1D side: `Params1d`,
# `initialize!`, `rhs_wave1d!`, `recommended_dt(::NamedTuple)`. 3D
# side: `Params3d`, `initialize3d!`, eigenmodes, `rhs_wave3d!`
# (apply_laplacian! + Sommerfeld dissipative pass), and the 3D
# `recommended_dt`.
include("wave.jl")

# `wave2d.jl`: 2D wave-equation layer (`Params2d`, `initialize2d!`,
# `eigenmode_cartesian_2d!`, `eigenmode_radial_2d!`, `rhs_wave2d!`,
# and the 2D `recommended_dt`).
include("wave2d.jl")

# `evolve.jl`: high-level drivers `evolve1d`, `evolve2d`, `evolve3d`.
# Each builds the geometry, runs a symplectic integration, samples a
# spacetime slice + L² error, and returns a NamedTuple consumed by
# the matching `bin/waveplot{1,2,3}d.jl` plot script.
include("evolve.jl")

# Re-export the operator-layer symbols that downstream `bin/` scripts
# and tests are used to seeing at the WaveToy level. Keeps the existing
# `using WaveToySecondOrder` call sites working unchanged for both the
# mesh-layer and operator-layer types.
export
    # Re-exports from HexMeshes
    Mesh, MeshConnectivity,
    PatchDesc, PatchKind, Cubic, Wedge, Inflation, Shell,
    make_uniform_line, make_uniform_quad, make_uniform_hex,
    make_cubed_square_mesh, make_inflated_square_mesh,
    make_cubed_cube_mesh, make_inflated_cube_mesh,
    nv, npatches, element_vertices, locate_point, invert_element_map,
    interpolate_field,
    # Re-exports from HexSBPSAT
    SBPOps, MeshGeometry, MeshWorkspace,
    make_element, make_domain, make_operators, make_geometry,
    make_workspace,
    element_coords, to_device,
    apply_laplacian!,
    build_global_laplacian, discrete_laplacian,
    spectral_radius_estimate,
    discrete_inner_product, discrete_l2_norm,
    physical_mass_diagonal,
    # Wave-equation layer — 1D
    Params1d, initialize!, rhs_wave1d!,
    # Wave-equation layer — 2D
    Params2d, initialize2d!,
    eigenmode_cartesian_2d!, eigenmode_radial_2d!,
    outgoing_pulse_2d!, outgoing_pulse_2d_cache, GaussianPulse2dCache,
    rhs_wave2d!,
    # Wave-equation layer — 3D
    Params3d, initialize3d!,
    eigenmode_cartesian!, eigenmode_radial!, eigenmode_quadrupole!,
    outgoing_pulse!,
    rhs_wave3d!,
    # Dimension-generic timestep limit (dispatches on dom / MeshGeometry)
    recommended_dt,
    # High-level drivers (move out of `bin/waveplot{1,2,3}d.jl`)
    evolve1d, evolve2d, evolve3d,
    pick_integrator,
    # Tags
    SOMMERFELD_BDRY_TAG

end
