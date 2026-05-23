# Ideas

- apply inter-element boundary conditions on u and dt u, not u and dx u u
- add test cases for distorted meshes. test convergence order?

# Plans

- GPUs
- multi-threading
- spherical outer boundary
- benchmark and improve performance again
- optimize non-distorted meshes; a factor 5 seems possible
- use adaptive time step sizes
- maybe "Switch to Polyester.@batch for lower-overhead threading on small loops — would make the nthreads=1 overhead vanish."
- add robust stability test
