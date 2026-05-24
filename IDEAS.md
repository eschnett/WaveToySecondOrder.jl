# Short-term ideas

# Long-term ideas

- spherical outer boundary
- optimize non-distorted meshes; a factor 5 seems possible
- use adaptive time step sizes
- Rational, DoubleFloat
- remove outer `@testset` to get more progress output; remove `_section`
- integrate time integrator into rhs calculation. probabaly use custom
  integrator? this would combine all kernels inte one (rhs, step, face
  buffers).
