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

import SpacetimeMetrics
using FastGaussQuadrature: gausslegendre
using KernelAbstractions
using KernelAbstractions: @kernel, @index, @Const, get_backend, CPU
using LinearAlgebra: eigvals
# Symplectic partitioned RK for the 2D/3D `SecondOrderODEProblem`
# drivers; explicit RK (RK4 / Tsit5 / Vern7) for the first-order 1D
# ADM system, which is not Hamiltonian for variable shift.
using OrdinaryDiffEqSymplecticRK
using OrdinaryDiffEqLowOrderRK   # RK4; reexports ODEProblem etc.
using OrdinaryDiffEqTsit5        # Tsit5
using OrdinaryDiffEqVerner       # Vern7
using RecursiveArrayTools: ArrayPartition
using ProgressMeter
using SpecialFunctions: besselj, sphericalbesselj
using StaticArrays

# `wave.jl`: 3D wave-equation layer — `Params3d`, `initialize3d!`,
# eigenmodes, `rhs_wave3d!` (apply_laplacian! + Sommerfeld dissipative
# pass), and the 3D `recommended_dt`. (The 1D layer lives in
# `wave1d.jl`.)
include("wave.jl")

# `wave2d.jl`: 2D wave-equation layer (`Params2d`, `initialize2d!`,
# `eigenmode_cartesian_2d!`, `eigenmode_radial_2d!`, `rhs_wave2d!`,
# and the 2D `recommended_dt`).
include("wave2d.jl")

# `wave_lap_strong.jl`: alternative single-element scalar Laplacian
# in strong form (SBP G applied twice with centred-flux SAT). Lives
# next to `apply_laplacian!` (SIPG) for direct comparison; not used
# by the evolution drivers yet.
include("wave_lap_strong.jl")

# `wave_curved_rhs.jl`: scalar wave equation on a prescribed 4-metric
# in fully second-order form, mirroring `gh_rhs_element!`. Scalar
# testbed for the GH spatial discretisation.
include("wave_curved_rhs.jl")

# `wave_strong_rhs.jl`: curvilinear strong-form scalar wave RHS on a
# `MeshGeometry`. Supports cubed-cube / inflated-cube meshes with
# Bayliss–Turkel Sommerfeld at outer faces (tag 7).
include("wave_strong_rhs.jl")

# `evolve.jl`: high-level drivers `evolve1d`, `evolve2d`, `evolve3d`.
# Each builds the geometry, runs a symplectic integration, samples a
# spacetime slice + L² error, and returns a NamedTuple consumed by
# the matching `bin/waveplot{1,2,3}d.jl` plot script.
include("evolve.jl")

# `wave1d.jl`: conservative-form 1D scalar wave on a 1+1 ADM
# background with arbitrary lapse α(t,x), shift β(t,x), and spatial
# metric γ_xx(t,x), discretised with `HexMeshes.Mesh{1}` +
# `HexSBPSAT.apply_D!`, plus the `Background1D` sampling layer.
include("wave1d.jl")

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
    # Wave-equation layer — 1D: conservative-form scalar wave on a
    # 1+1 ADM background with arbitrary α(t,x), β(t,x), γ_xx(t,x) on
    # HexMeshes/HexSBPSAT (`wave1d.jl`); densitised
    # Π := (√γ/α)(∂_t Φ − β ∂_x Φ).
    Wave1DWorkspace, make_wave1d_workspace,
    wave1d_curved_rhs!, wave1d_energy,
    # ADM background sampling (analytic closures or SpacetimeMetrics).
    Background1D, AnalyticBackground1D, MetricBackground1D,
    sample_background!,
    # Re-export the connectivity-driven 1D derivative from HexSBPSAT.
    apply_D!,
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
    # Strong-form single-element scalar Laplacian (alternative to SIPG
    # `apply_laplacian!`). Two variants — `_element!` is the centred-flux
    # form (stable but not H-symmetric); `_conservative_element!` adds
    # Mattsson–Nordström SAT pair so `H · L` is symmetric and energy is
    # conserved by the semidiscrete equations.
    wave_lap_strong_element!,
    wave_lap_strong_conservative_element!,
    wave_lap_strong_conservative_mesh!,
    # Scalar wave on a prescribed 4-metric (`wave_curved_rhs.jl`).
    wave_curved_rhs_element!,
    wave_curved_rhs_mesh!,
    wave_curved_rhs_conservative_element!,
    wave_curved_rhs_conservative_mesh!,
    eval_curved_background!,
    # Curvilinear strong-form RHS on MeshGeometry with Sommerfeld
    # outer BC (`wave_strong_rhs.jl`).
    wave_strong_rhs_element!,
    wave_strong_rhs_mesh!,
    # Dimension-generic timestep limit (dispatches on dom / MeshGeometry)
    recommended_dt,
    # High-level drivers (consumed by `bin/wave1d.jl`,
    # `bin/waveplot{2,3}d.jl`)
    evolve1d, evolve2d, evolve3d,
    pick_integrator, pick_integrator_first_order,
    # Tags
    SOMMERFELD_BDRY_TAG

end
