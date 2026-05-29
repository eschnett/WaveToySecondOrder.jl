module WaveToySecondOrder

# Three-layer package stack:
#
#   HexMeshes  → topology, parametric geometry, host-side queries.
#   HexSBPSAT  → SBP-SAT operators, per-node MeshGeometry, the 3D
#                discrete Laplacian `apply_laplacian3d!`, and diagnostics.
#   WaveToy    → the wave equation itself: IC, eigenmodes, Sommerfeld
#                radiative BC, recommended_dt, time-evolution driver.
#
# All three live in the same family of packages and import their lower-
# level neighbours explicitly. The kernel hot path is generic-enough that
# the wave-specific layer here is just a thin wrapper around
# `apply_laplacian3d!` with one extra dissipative kernel pass.

using HexMeshes
using HexMeshes: Mesh, MeshConnectivity,
                 PatchDesc, PatchKind, Cubic, Wedge, Inflation, Shell,
                 make_uniform_hex, make_cubed_cube_mesh, make_inflated_cube_mesh,
                 nv, npatches, element_vertices,
                 locate_point, invert_element_map, interpolate_field,
                 trilinear_shape, trilinear_dshape,
                 trilinear_map, trilinear_jacobian,
                 lagrange_basis, tensor_interp
using HexSBPSAT
using HexSBPSAT: SBPOps, MeshGeometry,
                 make_element, make_domain, make_operators, make_geometry,
                 element_coords, to_device,
                 apply_laplacian!, apply_laplacian3d!,
                 build_global_laplacian, discrete_laplacian,
                 spectral_radius_estimate,
                 discrete_inner_product, discrete_l2_norm,
                 physical_mass_diagonal
# Internal helpers from HexSBPSAT used by `wave3d.jl`'s Sommerfeld pass.
using HexSBPSAT: _face_axis_idx, _face_row, _face_sign, _cross_sign,
                 _tangent_axes, _ijk_from_li

using KernelAbstractions
using KernelAbstractions: @kernel, @index, @Const, get_backend
using SpecialFunctions: sphericalbesselj
using StaticArrays

# `wave.jl`: 1D + 3D wave-equation ICs (`initialize!`, `initialize3d!`,
# eigenmodes), `Params3d`, `rhs_wave3d!` (apply_laplacian3d! + Sommerfeld
# dissipative pass), and the wave-equation timestep limit
# `recommended_dt`.
include("wave.jl")

# Re-export the operator-layer symbols that downstream `bin/` scripts
# and tests are used to seeing at the WaveToy level. Keeps the existing
# `using WaveToySecondOrder` call sites working unchanged for both the
# mesh-layer and operator-layer types.
export
    # Re-exports from HexMeshes
    Mesh, MeshConnectivity,
    PatchDesc, PatchKind, Cubic, Wedge, Inflation, Shell,
    make_uniform_hex, make_cubed_cube_mesh, make_inflated_cube_mesh,
    nv, npatches, element_vertices, locate_point, invert_element_map,
    interpolate_field,
    # Re-exports from HexSBPSAT
    SBPOps, MeshGeometry,
    make_element, make_domain, make_operators, make_geometry,
    element_coords, to_device,
    apply_laplacian!, apply_laplacian3d!,
    build_global_laplacian, discrete_laplacian,
    spectral_radius_estimate,
    discrete_inner_product, discrete_l2_norm,
    physical_mass_diagonal,
    # Wave-equation layer
    Params3d, initialize!, initialize3d!,
    eigenmode_cartesian!, eigenmode_radial!, eigenmode_quadrupole!,
    outgoing_pulse!,
    rhs_wave3d!, recommended_dt,
    SOMMERFELD_BDRY_TAG

end
