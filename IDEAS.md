# Short-term ideas

- more mesh connectivity test cases; count V E F?
- show energy change in plots
- Metal backend broke because Float64s snuck in
- waveplot3d progress bar too long (exclude all the parameters, print them first separately)
- add/fix CI scripts for packages
- add/fix README.md for packages
- add N=2 (or N=3?) test case(s)
- fix all waveplot3d examples
- sommerfeld bc are broken
- figure titles are too long and/or wrong
- show maxabs during evolution
- move most of waveplot3d script and friends into the package proper
- test HexMeshes with MultiFloats
- sommerfeld for 1d and 2d
- exact solution for 2d
- why no wave1d?
- test with Float32x2, Float64x2
- avoid constant 1/100 in _j2_over_x2; look at eps(T) instead
- change `main2d(; mesh_kind = :inflated_square, ic_kind = :outgoing, outer_bc = :sommerfeld, t1 = 1.5, N = 4, M = 8)` id so that t1=1 suffices
- periodic boundaries for cubed-* meshes
- GH: codegen for gauge and physical constraints
- helper function for walking grid, inter-element boundaries?
- helper function for reductions
- test shift > 1
- GH: test with GPU
- ensure there are wavetoy tests with the curvilinear mesh
- add figure with the curvilinear mesh to readme
- check docs for HexMeshes
- new release for HexMeshes
- enums for boundary tags
- rename "inflated cube" to "cubed sphere"?
- proper sommerfeld boundary conditions for cubed sphere
- inhomogenous dirichlet boundary conditions for cubed sphere
- use these in ks example
- set up all apples-with-apples tests
- visualization helper functions in HexMeshes for distorted meshes (curved mesh lines)
- what second-order integrators are suitable?
- test converting the scripts to julia apps

# Long-term ideas

- optimize non-distorted meshes; a factor 5 seems possible
- use adaptive time step sizes
- Rational, DoubleFloat
- remove outer `@testset` to get more progress output; remove `_section`
- integrate time integrator into rhs calculation. probabaly use custom
  integrator? this would combine all kernels inte one (rhs, step, face
  buffers).
- run on H200
- low-level benchmark with ncu
- Array indexing: fixed strides?
- CUDA: warp shuffles?
- CUDA: tensor cores?
- implement GH
- add proper I/O. which package? ADIOS2? HDF5? what metadata? conduit? XDMF/HDF5? VTKHDF?
- run large test, check convergence
- implement electrodynamics
- mesh for two black holes
- initial conditions, superposed kerr-schild, effective Tμν, i⁰

# Bad ideas

- [NOT] equal angles in outermost shell in inflated cube
- [NOT] use Meshes.jl



# Tests

using Revise
using Metal
include("bin/waveplot1d.jl")
include("bin/waveplot2d.jl")
include("bin/waveplot3d.jl")

main1d(; N = 4, M = 32)
main1d(; T = Float32, backend = MetalBackend(), N = 4, M = 32)

main2d(; N = 4, M = 8)
main2d(; mesh_kind = :cubed_square, N = 4, M = 8)
main2d(; mesh_kind = :inflated_square, ic_kind = :radial, N = 4, M = 8)
main2d(; mesh_kind = :inflated_square, ic_kind = :outgoing, outer_bc = :sommerfeld, t1 = 1.5, N = 4, M = 8)
main2d(; T = Float32, backend = MetalBackend(), mesh_kind = :inflated_square, ic_kind = :outgoing, outer_bc = :sommerfeld, t1 = 1.5, N = 4, M = 8)

main3d(; N = 4, M = 8)
main3d(; mesh_kind = :cubed_cube, N = 4, M = 8)
main3d(; mesh_kind = :inflated_cube, ic_kind = :radial, N = 4, M = 8)
main3d(; mesh_kind = :inflated_cube, ic_kind = :outgoing, outer_bc = :sommerfeld, N = 4, M = 8)
main3d(; T = Float32, backend = MetalBackend(), mesh_kind = :inflated_cube, ic_kind = :outgoing, outer_bc = :sommerfeld, N = 4, M = 8)
