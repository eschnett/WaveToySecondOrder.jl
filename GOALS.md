# Goals for the package WaveToySecondOrder

This is a Julia package implementing the scalar wave equation.

This file describes my goals for this package. Do not modify this
file, but you can suggest changes to correct errors or add missing
information.

## Overall goal

One of the main goals is to be a testbed for a GPU-efficient
implementation of the Einstein equations. Apart from "the usual"
conditions, "GPU-efficient" means to reduce storage and memory
bandwidth as much as possible.

## Formulation

Implement the scalar wave equation for an arbitray (given) background
metric. This include superluminal shift vectors. Use a
second-order-in-space formulation. A second-order-in-time formulation
is not necessary.

Provide 1d, 2d, and 3d implementations.

Use the package SpacetimeMetrics for background metrics. Assume the
ADM variables (lapse, shift, three-metric) are given on the
collocation points.

## Discretization

Use unstructured conforming hexahedral meshes, provided by
HexMeshes. Use SBP-SAT operators provided by HexSBPSAT. These
packages may be extended or updated if necessary. Elements with N=4 or
N=8 grid points seem particularly suitable for GPUs.

On curvilinear meshes the conservative first-derivative operator uses
metric terms computed *discretely* from the nodal coordinates (so the
discrete metric identities hold and a constant state is preserved —
free-stream preservation) and the split (skew-symmetric) form (so the
gradient/divergence pair is energy-stable). In 2D the metric
identities hold automatically; the 3D case needs the harder
conservative-curl metric form. Implemented for 2D cubed-square; 3D
curvilinear is future work.

The inter-element and outer boundary conditions are to be decided. The
current implementation is making choices that either need to be
confirmed, corrected, or consolidated. I prefer SBP/SAT operators
because they provide an energy estimate that helps establishing
stability and because they decouple elements.

## Simplifications

Things may not work out of the box. If so, consider simplifications of
the system to simplify debugging.

The ultimate goal is a cubed-sphere mesh in 3d. When implementing,
start with 1d, then move to 2d and 3d.

For each dimension, stepping stones are (in order of increasing
complexity): a single element, a cubical mesh, a cubed-cube mesh, then
the cubed-sphere mesh.

Start with periodic boundaries, then Dirichlet boundaries, then
excision boundaries, then Sommerfeld boundaries.

Boundary conditions are chosen per face from the local characteristic
*speeds* (inflow/outflow classification): superluminal outflow faces
use excision, superluminal inflow faces use full-state Dirichlet, and
subluminal faces use Dirichlet or radiative (Sommerfeld) conditions.
Radiative boundaries use a characteristic-free *field-radiation*
condition (∂_t Φ + (α/√γ)·n̂·∂_x Φ = 0, applied to the field via its
evolution equation), valid for small shift (|β| ≲ 0.1); this is the
choice that ports field-by-field to the Einstein equations. The
characteristic *eigenvector* decomposition is deliberately avoided for
boundaries — it would be needed only for perfectly-absorbing or
constraint-preserving conditions, which are out of scope. Only the
characteristic *speeds* (eigenvalues) are used, for face classification
and penalty magnitudes.

## Tests

Tests should include:
- A stability analysis of the RHS operator, constructing the operator
  explicitly and studying its eigenvalues. (This can be skipped for
  large systems with too many grid points.)
- A robust stability test, i.e. evolving noise.
- Convergence tests, comparing against analytic solutions.
- Monitoring the total energy for these tests.

Tests should finish in a relatively short time. (A rule of thumb is
less than 30 seconds each.)

## Examples and visualizations

Provide apps that evolve the scalar wave equation in 1d, 2d, and 3d
(one app per dimension). Use command line parameters to choose mesh
structure, mesh parameters, element order, initial conditions, and
boundary conditions. (Use analytic solutions as initial conditions.)
Also allow choosing the floating-point type and accelerator backend.

These apps serve both as confirmation that long-term evolutions are
working (although you are not expected to run these apps yourself) and
serve as examples to users of this package.

Include visualizations via CairoMakie and FileIO (png) / SixelTerm
(terminal) output. For each case, show figures showing the grid
structure, the initial and the final solution, the total energy vs.
time, and the L2 norm of the error vs. time. For 3d simulations plot a
slice of the domain.

## Documentation

Keep a file `METHODS.md` up-to-date with the numeric choices and
details you make.

## Implementation

Use HexMeshes and HexSBPSAT.

Use SpacetimeMetrics for background metrics.

Use DifferentialEquations or its subpackages for time integration.
Choose appropriate time integrators with a matching order of accuracy.

Write code that works on CPUs as well as GPUs using
KernelAbstractions. The local system supports Metal. An indirectly
available test system supports CUDA. Include tests for Metal and CUDA,
and automatically skip these tests if the hardware is not available.

Make the kernels type-agnostic. They should work at least with
Float64, Float32, and the types Float64x2 or Float32x2 from
MultiFloats.
