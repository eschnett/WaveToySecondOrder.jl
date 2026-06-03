claude --resume 472e4092-91f1-4d9b-a9c2-5ff07b6cb80e



 Proposed corrections and additions to GOALS.md

 Context

 You asked me to read GOALS.md and suggest corrections / missing
 details. Below are my observations, grouped by section. I have not
 edited GOALS.md — these are recommendations for you to apply.
 Items are tagged [correction] (likely-wrong as written),
 [clarify] (ambiguous), or [missing] (worth adding).

 ---
 § Overall goal

 - [clarify] "Testbed for a GPU-efficient implementation of the
 Einstein equations" — worth adding one sentence that says the
 scalar wave is the simplest model that exercises the data-flow
 pattern (state + auxiliary momentum, evaluated on hex-mesh-of-
 elements with SBP-SAT), so the kernel structure and memory-
 layout choices transfer to GH/BSSN later. Right now the
 motivation reads as "Einstein in the future" without explaining
 what the scalar wave actually buys you.
 - [missing] Performance targets: "GPU-efficient" is unmeasured.
 Pin one or two numbers (e.g. ≥ 50 % of peak DRAM bandwidth at
 N=4, M~32³ on Metal/CUDA) so a future change can be evaluated.
 Without a target, "efficient" is unfalsifiable.

 § Formulation

 - [correction] "Use a second-order-in-space formulation. A
 second-order-in-time formulation is likely not possible …" —
 this conflates two axes. The package's actual choice is
 second-order PDE in space (one Laplacian-like spatial term)
 combined with a first-order ADM split in time with two state
 variables (Φ, Π). That distinction matters: a "second-order in
 time" form (Φ̈ = …) is perfectly fine for constant β (see the
 3D wave_curved_rhs! kernel that already does it) and works as
 long as the metric isn't pathological — what made it unworkable
 for the 1D sonic-horizon investigation was specifically variable-
 β plus the centred-flux SAT, not the order-in-time per se.
 Suggested rewrite:

 ▎ Solve the wave equation as a second-order spatial PDE
 ▎ using a first-order ADM split in time with state (Φ, Π),
 ▎ Π := √γ · (∂_t Φ − β^i ∂_i Φ). This handles arbitrary shift
 ▎ (including superluminal) and arbitrary lapse uniformly. A
 ▎ pure second-order-in-time form is retained where it's already
 ▎ working (e.g. 3D constant-shift demos) but isn't the canonical
 ▎ path for variable β.
 - [missing] Lapse α isn't mentioned. The current 1D kernel
 fixes α = 1. For Schwarzschild Painlevé–Gullstrand or BSSN-
 gauge problems, α varies in space and time. State explicitly
 that the kernel is expected to evolve on backgrounds with
 arbitrary (α, β^i, γ_{ij}).
 - [missing] Sonic horizons (|β·n̂| = 1). The package
 intends to support |β| > 1; in 1D this means crossing the
 sonic horizon is possible (a foliation-degeneracy point). Spell
 out the policy: the kernel should not blow up, but exact
 convergence near the horizon isn't expected. Or, the package
 requires |β| < 1 everywhere.
 - [clarify] "Use SpacetimeMetrics.jl for background metrics."
 Worth adding what the kernel consumes — full 4-metric g_{μν},
 or pre-computed ADM (α, β^i, γ_{ij}) plus Christoffels at GLL
 nodes. The current 3D wave_curved_rhs_* and 1D curved1d
 paths differ here.

 § Discretization

 - [clarify] "Elements of order 3 and 7 (with N=4 or N=8 grid
 points)". Use a single convention. The HexSBPSAT convention is
 N = number of GLL nodes per element axis = polynomial degree
       i. "Order 3" almost certainly means polynomial degree 3
 (= N = 4 GLL nodes). Suggested phrasing: "GLL elements with
 N ∈ {4, 8} nodes per axis (polynomial degree 3 and 7)". You
 could also note expected accuracy: ≈ N+1 for smooth solutions
 on constant coefficients, ≈ N−1 for variable coefficients —
 matches what the gauge-wave test (~3rd order at N=4) and the
 plane-wave test (~5th order at N=4) show.
 - [correction] "Inter-element and outer boundary conditions are
 to be decided." — partly decided already. The 1D code uses
 centred-flux SAT which gives an exactly skew H·D (energy-
 conservative for α=γ=1, β=const). For variable γ the kernel
 is conservative by construction (flux-divergence form) but the
 quadratic energy is not exactly conserved. State the choice
 rather than leaving it open. If alternatives are still being
 considered (Mattsson–Nordström, SIPG), say so explicitly.
 - [missing] Kreiss–Oliger dissipation. The 1D kernel uses
 ε · h^{2p+1} · D^{2p+2} artificial dissipation as a stability
 device for high-frequency noise on the marginally-stable
 imaginary-axis spectrum. KO is part of the discretization
 story; mention it as a tunable, with the default and the
 scaling that preserves formal order (O(h^{2p−1}) for smooth
 solutions).
 - [clarify] SBP energy estimate vs SAT decoupling — these two
 sentences are slightly in tension. SAT patches an energy
 estimate at every face; without the SAT pair you wouldn't get
 the H-norm energy theorem. Worth rephrasing the "I prefer SBP …
 and SAT" pair as a single design rationale, not two independent
 reasons.

 § Simplifications

 - [correction] Internal inconsistency. The §Formulation section
 says "1d, 2d, and 3d"; here you say "Start with 1d, then move
 to 3d." Pick one. If 2D is in scope (e.g. as a stepping stone
 for cubed-square meshes, which the existing
 make_cubed_square_mesh builds), add it. If not, remove the
 2D mention earlier.
 - [clarify] "For each dimension, stepping stones are … a single
 element, a cubical mesh, a cubed-cube mesh, then the cubed-
 sphere mesh." — the cubed-cube and cubed-sphere are inherently
 multi-D constructs (HexMeshes has make_cubed_cube_mesh,
 make_inflated_cube_mesh). In 1D the stepping stones collapse
 to "single periodic element → multi-element periodic →
 Dirichlet → Sommerfeld". State the 1D vs 2D/3D stepping-stone
 lists separately.
 - [missing] Boundary-condition stepping stones — currently
 periodic and Dirichlet are tested in 1D and 3D; Sommerfeld 3D
 is implemented but the 1D Sommerfeld is open (see IDEAS.md:
 "sommerfeld for 1d and 2d"). Make that visible in the goals.

 § Tests

 - [correction] "A robust stability test … corresponds to using
 the power method to find the largest eigenvalue." — not quite.
 The power method on a non-normal operator converges to the
 eigenvalue with largest |λ|, while what stability cares about
 is max Re(λ). For a skew operator the two are unrelated (all
 λ are imaginary; |λ| is the highest discrete wavenumber).
 Noise evolution actually detects: (i) any Re(λ) > 0 that
 shows up, and (ii) Jordan-block-driven polynomial growth that
 the eigenvalue scan can miss. Re-phrase to reflect that.
 - [missing] Two distinct stability checks are wanted:
 (a) spectrum of the linearised RHS operator (max Re(λ),
 cond(eigvecs) for defectiveness),
 (b) time evolution of √eps noise over many light-crossings
 with growth-factor bound.
 These catch different failure modes. Worth saying both are
 required and what each pins down.
 - [clarify] "Monitoring the total energy" — for curved
 backgrounds "total energy" needs disambiguation. The continuous
 conserved quantity is the inertial-frame energy when the
 spacetime has a timelike Killing vector; the ADM energy
 (H-norm of Π plus H-norm of D Φ weighted by γ) is not
 conserved for time-varying γ. The 1D curved1d testset uses
 the inertial energy. State which one in the goals.
 - [missing] Pass thresholds are not specified. Convergence
 tests should pin a minimum rate (e.g. "≥ N+1 − 1 for constant
 coefficients, ≥ 2 for variable coefficients"). Energy drift
 should pin a tolerance (e.g. "< 10⁻³ at the finest M"). Without
 thresholds, regressions can pass silently.
 - [clarify] "Tests should finish in a relatively short time
 (~30 s each)" — currently the full Pkg.test() is ~5 minutes;
 individual 1D testsets are ~5–10 s, but some 3D testsets
 (wave_strong_rhs_mesh!, the cubed-sphere Dirichlet test) are
 ▎ 1 minute each. State whether the 30 s budget is per-testset
 ▎ or per-@testset block, and whether it applies to 3D.

 § Examples and visualizations

 - [missing] Output format. For 3D, "the final solution" can't
 be a single figure — say which slice or projection (e.g.
 z = 0 plane, or radial profile for cubed-sphere). The 2D and
 3D waveplot{2,3}d.jl apps already make some of these choices;
 reference them so the goal doesn't ask for something different.
 - [missing] Whether the apps should write any persistent output
 (HDF5 snapshots, JLD2 checkpoints) or only PNG frames. Matters
 for the "long-term evolution" claim.

 § Documentation

 - [missing] METHODS.md doesn't exist yet (only IDEAS.md,
 NOTES.md, README.md). Either commit a stub now to fix the
 reference, or change the goal to "create and maintain
 METHODS.md". Worth sketching its scope: discretization
 choices (SBP-SAT variant, KO, integrator), state-vector
 conventions (densitised Π, sign of β), boundary-condition
 catalogue, and known limitations (sonic horizon, GPU
 unsupported types).

 § Implementation

 - [correction] "Use DifferentialEquations.jl or its subpackages
 for time integration." — the 1D testsuite currently uses
 inline RK4 (in test_curved1d_wave.jl); only evolve1d/2d/3d
 (the high-level demos) uses OrdinaryDiffEqSymplecticRK. Note
 also that for first-order ADM with variable β, the system
 is not Hamiltonian in general — symplectic integrators are
 not the right default. State the preferred integrator family
 (e.g. explicit RK4/RK8 for evolution; symplectic only for
 cases where it applies).
 - [clarify] Type-agnostic kernels — pin a concrete extended-
 precision package. Float64x2 etc. come from MultiFloats.jl
 (per IDEAS.md). Naming it removes ambiguity.
 - [missing] GPU strategy — the codebase already uses
 KernelAbstractions.jl (CPU/Metal/CUDA via the same kernel).
 Add one line that names KA as the GPU abstraction layer, and
 note that Metal is local-dev and CUDA is CI/test-system.
 - [missing] What [ε_KO] is the default? Currently 0.1 is
 the kernel default but tests routinely use 0.0 (no KO) or
 1e-4. Settle on a recommended default and document the CFL
 constraint it implies (the KO term has its own RK4 stability
 region).

 Cross-cutting / process

 - [missing] Current-state snapshot. GOALS.md reads as a
 fresh roadmap, but the package already has substantial 3D
 infrastructure (cubed-sphere meshes, Sommerfeld BC, GPU
 evolution, conservative 1D shift wave). One paragraph saying
 "Here is what's already in tree; here is what's still open"
 would orient readers (and future-you) faster.
 - [missing] 3D ADM-form roadmap. The clean 1D conservative
 kernel wave1d_curved1d_rhs! was just landed. The 3D path
 presently uses the second-order-in-time wave_curved_rhs_*.
 Will those be migrated to the first-order ADM form, or kept
 side by side? Stating the intent now avoids "two parallel
 worlds" diverging.
 - [missing] Versioning / release criteria. What counts as
 "v1.0"? E.g.: "3D curved-metric scalar wave converges on
 cubed-sphere with Sommerfeld outer BC, runs on Metal at ≥ N
 GFLOP/s, energy conserved to 10⁻³ over 100 light-crossings."
 Concrete criteria make "done" detectable.

 ---
 Suggested action

 Apply the highest-priority items first:
 1. [correction] second-order-in-space vs first-order-in-time
 distinction (§Formulation).
 2. [correction] power-method vs spectrum framing (§Tests).
 3. [correction] integrator: DifferentialEquations + symplectic
 only where applicable (§Implementation).
 4. [missing] lapse α; sonic-horizon policy (§Formulation).
 5. [missing] energy definition (§Tests); concrete pass
 thresholds.
 6. Resolve the 1D-vs-2D-vs-3D inconsistency (§Formulation vs
 §Simplifications).

 Other items are clarifying or expanding; pick what you find most
 useful.
