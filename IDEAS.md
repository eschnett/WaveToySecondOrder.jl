# Short-term ideas

- Array indexing: fixed strides?

# Long-term ideas

- spherical outer boundary
- optimize non-distorted meshes; a factor 5 seems possible
- use adaptive time step sizes
- Rational, DoubleFloat
- remove outer `@testset` to get more progress output; remove `_section`
- integrate time integrator into rhs calculation. probabaly use custom
  integrator? this would combine all kernels inte one (rhs, step, face
  buffers).

# workgroups

  7. Migration sketch (what changes, in what order)

  ┌──────┬─────────────────────────────────────────────────────────────────────────────────────┬──────────────────────────────────────┐
  │ Step │                                       Change                                        │             Touch points             │
  ├──────┼─────────────────────────────────────────────────────────────────────────────────────┼──────────────────────────────────────┤
  │      │ Split rhs3d! into rhs3d_trace! + rhs3d_volume!. Add face_trace global buffer (sized │ src/kernels3d.jl, src/mesh.jl        │
  │ 1    │  at problem setup).                                                                 │ (MeshGeometry gets a face_trace::A   │
  │      │                                                                                     │ field)                               │
  ├──────┼─────────────────────────────────────────────────────────────────────────────────────┼──────────────────────────────────────┤
  │ 2    │ Rewrite rhs3d_volume! kernel as workgroup-per-element with @localmem for u, Du, W,  │ src/kernels3d.jl                     │
  │      │ ü_local. Add intra-workgroup loop over (li, lj, lk).                                │                                      │
  ├──────┼─────────────────────────────────────────────────────────────────────────────────────┼──────────────────────────────────────┤
  │ 3    │ Verify bit-identity vs current implementation on the existing test suite.           │ (tests don't change)                 │
  ├──────┼─────────────────────────────────────────────────────────────────────────────────────┼──────────────────────────────────────┤
  │ 4    │ Bench on Metal + measure. Verify no regression on CPU.                              │ bin/bench3d.jl                       │
  ├──────┼─────────────────────────────────────────────────────────────────────────────────────┼──────────────────────────────────────┤
  │      │ When CUDA hardware is available: add ext/WaveToySecondOrderCUDAExt.jl. Inside,      │                                      │
  │ 5    │ override _rhs3d_volume_kernel! with a CUDA.jl @cuda kernel that uses WMMA for the   │ ext/, Project.toml                   │
  │      │ stencil pass.                                                                       │                                      │
  ├──────┼─────────────────────────────────────────────────────────────────────────────────────┼──────────────────────────────────────┤
  │ 6    │ For more variables, generalise the kernel to take a Val(V) or struct-of-arrays      │ substantive refactor                 │
  │      │ state. Bench variable-blocking strategies.                                          │                                      │
  └──────┴─────────────────────────────────────────────────────────────────────────────────────┴──────────────────────────────────────┘
