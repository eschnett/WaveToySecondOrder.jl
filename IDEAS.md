# Short-term ideas

- more mesh connectivity test cases; count V E F?
- use enum for patch kinds
- better initial conditions for inflated cube
- equal angles in outermost shell in inflated cube
- different \tau for different elements?
  - choose \tau for fiducial element, then scale?
- use actual eigenvalue solver in `spectral_radius_estimate`

# Long-term ideas

- spherical outer boundary
- optimize non-distorted meshes; a factor 5 seems possible
- use adaptive time step sizes
- Rational, DoubleFloat
- remove outer `@testset` to get more progress output; remove `_section`
- integrate time integrator into rhs calculation. probabaly use custom
  integrator? this would combine all kernels inte one (rhs, step, face
  buffers).
- run on H200
- low-level benchmark with ncu
- radiative outer boundaries
- Array indexing: fixed strides?
- CUDA: warp shuffles?
- CUDA: tensor cores?
