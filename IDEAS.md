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
- ensure each test finishes in <30s, ideally <10s
- 1d/1d meshes: check names, check test cases
- meshes: clean up backward compatibility layers
- move most of waveplot3d script and friends into the package proper

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

# Bad ideas

- [NOT] equal angles in outermost shell in inflated cube
- [NOT] use Meshes.jl



# Tests

- main(; mesh_kind = :cubical, ic_kind = :cartesian, outer_bc = :dirichlet, N = 4, M = 8)
- main(; mesh_kind = :cubed_cube, ic_kind = :cartesian, outer_bc = :dirichlet, N = 4, M = 8, R = 0.1)
- main(; mesh_kind = :inflated_cube, ic_kind = :radial, outer_bc = :dirichlet, N = 4, M = 8, L = 0.1, R1 = 0.3, R2 = 1.0)
- main(; mesh_kind = :inflated_cube, ic_kind = :outgoing, outer_bc = :sommerfeld, N = 4, M = 8, L = 0.1, R1 = 0.3, R2 = 1.0)



# Tusculum

`find_patch` / `find_element_in_patch`
- why Newton?
- nan for outside.
- efficient on gpu?
- two functions, inverses: patch+local -> global, global -> patch+local
- overall design clean?
- skeleton visible in mesh? (should it?)



use enum for patch kinds
- remove symbols



Take a step back and review the whole package HexMeshes with a fresh mind:
- Are the datatypes well designed? Is there duplication? Could a redesign simplify their implementation?
- Is the external API well designed? Is there duplication? Does it prefer three dimensions?
- Are the functions well implemented? Is there duplication? Should they be split or refactored?
- Are there tests well-rounded? Are there tests for all three dimensions? Do the tests finish quickly?
- Is the package type-agnostic -- does it support Float32, Float64?
  arbitrary precision types? rational numbers, where this makes sense?
- Are there any issues with float-point precision?
- Does it run efficiently on GPUs (e.g. the point location functions)?
- Should the package make use of other, existing Julia packages?
- Would it make sense to add functionality to support visualization?
- Do you have any other observations? Are other clean-ups or redesigns possible or worthwhile?
Assume this is a preparation for a major redesign, i.e. breaking the API is allowed.



  Suggested phasing:
  1. Tolerance & precision fixes + round-trip tests (small, immediate value).
  2. FaceLink.kind Symbol → union type; rename uniform builders consistently.
  3. PatchSpec{D,T} / SkeletonMesh{D,T} unification + drop _2d suffixes.
  4. Split inflated_*.jl monoliths; documented orientation-group abstraction.
  5. Sibling viz/IO packages if/when wanted.
