# Short-term ideas

- GPUs
- KernelAbstractions
- face buffers

# Long-term ideas

- spherical outer boundary
- optimize non-distorted meshes; a factor 5 seems possible
- use adaptive time step sizes
- Rational, DoubleFloat
- remove outer `@testset` to get more progress output; remove `_section`
- integrate time integrator into rhs calculation. probabaly use custom
  integrator? this would combine all kernels inte one (rhs, step, face
  buffers).



# GPU migration

Migration order I'd suggest

1. Add KernelAbstractions dep; rewrite rhs3d! for the KA CPU() backend; verify all existing tests still pass and check the perf cost vs
Polyester. This is the riskiest step. If KA-CPU is much slower than Polyester, we know it before touching the GPU.
2. Loosen MeshGeometry to abstract arrays; add to_device.
3. Verify OrdinaryDiffEqSymplecticRK runs end-to-end with CuArray / MtlArray state.
4. Add the extension modules and a one-liner integration test per backend.
5. Migrate reductions (discrete_inner_product etc.) to mapreduce form.
