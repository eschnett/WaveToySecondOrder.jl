⏺ Structural review

  664 lines, 24 functions, four "domains" (1D element / 1D global / 3D element / 3D global) interleaved. Here's what I'd cut, what I'd
  add, and what's actually fine.

  Things to remove (dead or quasi-dead code)

  add_dirichlet_penalties! — once Phase 2/3 introduced _sat_increment, this mutating wrapper became unused. Every kernel now goes through
  the pure-functional _sat_increment returning an SVector. Delete it.

  evolve and evolve3d — these hard-code initial conditions (A, kx, ky, kz), the integrator (KahanLi8), the timestep, the time interval.
  They're really driver scripts hiding inside the module. Move them to bin/. The module should export the building blocks; the driver
  decides the experiment. (You already have bin/waveplot.jl and bin/waveplot3d.jl that call them — just inline the body into those
  scripts.)

  build_global_laplacian — only used by tests for symmetry / null-space checks. Either move into the test file, or keep here but mark as a
   diagnostic.

  add_axis_laplacian3d! is on the edge — it's called only by apply_laplacian3d! and could be inlined into it. Three near-identical loops,
  one per axis. Worth keeping factored, but the docstring should make that clear.

  Things to add

  FaceData{N, T} struct. The signature of apply_laplacian3d! is the biggest pain point:

  apply_laplacian3d!(Lu, u,
      ux_m, ux_p, Gux_m, Gux_p, αx_m, αx_p,
      uy_m, uy_p, Guy_m, Guy_p, αy_m, αy_p,
      uz_m, uz_p, Guz_m, Guz_p, αz_m, αz_p; ops, τ)

  That's the implementation leaking into the interface. The algorithm cares about three things: a 3D field, an operator, and per-face
  data. Wrap the six per-face fields in:

  struct FaceData{N, T}
      u_minus  :: SMatrix{N, N, T}
      u_plus   :: SMatrix{N, N, T}
      Gu_minus :: SMatrix{N, N, T}
      Gu_plus  :: SMatrix{N, N, T}
      α_minus  :: T
      α_plus   :: T
  end

  Then:

  apply_laplacian3d!(Lu, u, facex::FaceData, facey::FaceData, facez::FaceData; ops, τ)

  FaceData is isbits (all fields are concretely-typed SMatrix / scalars), so it's stack-allocated and won't reintroduce the NamedTuple
  allocation problem. This is the single most impactful readability change in the file.

  You could even build this once per-element-per-axis inside rhs3d!:
  facex = FaceData(u_xm, u_xp, Gu_xm, Gu_xp, αx_m, αx_p)
  making each of the six face blocks end in a single line.

  For 1D you could mirror this with FaceData1D{T} (scalars instead of SMatrices), but it's less urgent — the 8-arg 1D signature is
  borderline OK.

  Things where the algorithm and interface don't quite line up

  Outer Dirichlet asymmetry between 1D and 3D RHS.
  - 1D: rhs!(ü, u, u̇, bL, bR; …) — explicit boundary values.
  - 3D: rhs3d!(ü, u, u̇; …) — implicit homogeneous Dirichlet.

  The algorithms are identical; the interfaces aren't. Either:
  - (a) drop bL, bR from 1D (implicit zero), or
  - (b) add a bdry arg to 3D (six scalar values or six matrices).

  (b) is more general; (a) is simpler. I'd do (a) for now and only add (b) when you need inhomogeneous BCs (e.g., GR with non-trivial
  spacetime metric on the boundary).

  apply_laplacian! (1D) vs _apply_laplacian (pure) — the second is the algorithm; the first is the bang-bridge to mutable arrays. Worth
  renaming the bang version to apply_laplacian (no bang) since it accepts an AbstractVector output, or removing it entirely and having
  callers do Lu .= _apply_laplacian(SVector{N}(u), …). The latter is one line at each call site.

  Things that are good as-is

  - Two-method _sat_increment dispatched on Hmat. Algorithm cleanly factored: the Rational and GLL branches differ only in the H matrix
  structure, and the dispatch makes that explicit at the type level.
  - _node, _fiber_view, _face_smatrix, _face_gradient — these are the axis abstraction. Each is the right size (1-3 lines for helpers, ~12
   for the kernels). They map directly onto the math: "give me the i-th node of the D-axis fiber" / "give me the face slice along D".
  - SBPOps{N, T, NN, Hmat} as a typed bundle of operators. This is the right abstraction — it's the discrete operator family on one
  element. Don't change.
  - Six explicit if mx==1 / elseif mx==Mx / else face blocks in rhs3d!. Yes, repetitive, but the indexing differs per axis and per side in
   non-trivial ways. A Val{D, side}-dispatched helper would add more dispatch boilerplate than it saves. Keep.

  Proposed file organization

  Currently 1D and 3D code is interleaved. Reorder into clear sections:

  1. Operators & domain
     - make_element, make_domain
     - SBPOps, _make_operators

  2. Initial conditions
     - initialize! (Vector, Matrix)
     - initialize3d! (Array{T,3}, Array{T,6})

  3. Axis abstractions (used by both 1D and 3D where it makes sense)
     - _fiber_view, _node
     - _face_smatrix, _face_gradient

  4. 1D kernels
     - _sat_increment (both Hmat methods)
     - _apply_laplacian, apply_laplacian!
     - rhs! (Matrix, global)

  5. 3D kernels
     - FaceData, add_axis_laplacian3d!, apply_laplacian3d!
     - rhs3d!

  6. Diagnostics / test utilities
     - build_global_laplacian

  If you go one step further, splitting into operators.jl, kernels1d.jl, kernels3d.jl included from a thin top-level file is reasonable at
   this size. ~250 lines per file, each with a clear purpose. Trixi.jl is structured this way.

  Things I would not do

  Unify 3d-suffixed names with the 1D versions via dispatch. Tempting (initialize! overloaded on dim) but in your case the 1D
  Vector/Matrix and 3D Array{T,3}/Array{T,6} types are not parallel — Matrix doesn't mean "1D global" in any obvious way, just "a 2D
  array". Keeping 3d in the names makes the surface explicit.

  Replace _fiber_view (3D-Array views) with direct indexing. We tried something similar with _node for the 6D case. For the 3D-block case
  the view is going through one less level of nesting, and the compiler handles it well. Don't fix what works.

  Add an Element{N,T} type bundling ops + scratch. Tempting but premature: the only "scratch" left after Phase 3 is the constant Z =
  zero(SMatrix{N,N,T}), which is now type-inferred from ops and free. Wait until you have something concrete to bundle.

  Recommended order

  1. Delete add_dirichlet_penalties! (5 minutes).
  2. Move evolve / evolve3d to bin/ (10 minutes). Strip the hardcoded params, push them up into the scripts.
  3. Add FaceData struct, refactor apply_laplacian3d! and rhs3d! (1 hour). Run tests + benchmark to verify no allocation regression.
  4. Reorder the file into sections (30 min). Or split into 3 files if you want to commit harder.
  5. Decide on Dirichlet API consistency between 1D and 3D RHS (15 min).

  Want me to start with any of these? My pick would be #1 + #3 + #4 in one pass — they together would compress the file by ~100 lines and
  make the interface match the math better.
