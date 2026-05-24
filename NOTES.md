⏺ Discovery summary. Metal supports @private after all — the issue was us, not Metal's KA integration.

  What I dug up:

  1. Metal.jl issue tracker: no open issue for the Scratchpad/MArray story. The address-space enum in Metal/SVvf5/src/device/pointer.jl
  has Thread = 4 # Ends up same as Device? — the maintainers themselves aren't sure how the thread address space is currently lowered. So
  the formal mapping isn't pinned down at the LLVM level.
  2. Metal.jl/src/MetalKernels.jl:186: the Scratchpad override is indeed MArray{__size(Dims), T}(undef) — just an MArray, no native
  private-memory intrinsic. Their SharedMemory directly above does emit threadgroup_memory via LLVM IR; for thread memory the equivalent
  doesn't exist.
  3. CUDA.jl#2313 (closed): documented this exact failure pattern for CUDA at one point. Resolution was not a Scratchpad rewrite — it was
  getting the @inbounds-everywhere discipline right, plus a GPUCompiler optimization-pass update. Tim Besard's comment is precise: "the
  core issue here is that the boundserror for MArray captures the object. Causing it to escape. In KA I am very cautious about this using
  @inbounds."
  4. KA/src/cpu.jl:212: KA's own CPU code defines aview (using Base.unsafe_view) with this comment: "Base.view creates a boundscheck which
   captures A — JuliaLang/julia#39308". They use it internally because they know Base.view poisons escape analysis.

  Verification. I wrote a 12-line standalone Metal kernel using @private Float32 (192,) at workgroupsize = 64 with @inbounds-everywhere →
  it compiles and runs cleanly. So Metal's @private does work; what killed us before was that our kernel had ue = @view u[:, :, :, e] and
  a half-dozen conn.X[f, e] reads outside @inbounds, all dropping back into the boundscheck path that captured Du / W and prevented MArray
   promotion.

  Fix applied: restored @private for scratch, eliminated the global-memory Du_buf / W_buf allocation in rhs3d!. Wrapped the view creations
   and the four conn.*[f, e] reads in _add_face_sat! in @inbounds. Net result:

  ┌────────────────────────┬───────────────────────────┬─────────────────────────────────────────────┐
  │                        │  Before (global scratch)  │              After (@private)               │
  ├────────────────────────┼───────────────────────────┼─────────────────────────────────────────────┤
  │ CPU per-element        │ 530–650 ns                │ 550–675 ns                                  │
  ├────────────────────────┼───────────────────────────┼─────────────────────────────────────────────┤
  │ CPU per-call allocs    │ 48, 9152 B                │ 48, 9152 B (no change — KA launch overhead) │
  ├────────────────────────┼───────────────────────────┼─────────────────────────────────────────────┤
  │ CPU tests              │ 445/445 in 34 s           │ 445/445 in 35.5 s                           │
  ├────────────────────────┼───────────────────────────┼─────────────────────────────────────────────┤
  │ Metal single-call      │ bit-identical             │ bit-identical                               │
  ├────────────────────────┼───────────────────────────┼─────────────────────────────────────────────┤
  │ Metal CandyRoz4 evolve │ bit-identical             │ bit-identical                               │
  ├────────────────────────┼───────────────────────────┼─────────────────────────────────────────────┤
  │ Device memory per call │ 2·3·N³·Ne·sizeof(T) extra │ zero extra                                  │
  └────────────────────────┴───────────────────────────┴─────────────────────────────────────────────┘

  CPU got marginally slower (~5%), GPU gets the proper register-resident scratch path (no DRAM bounce per element) and saves several
  KB/element of device memory on large problems.

  Takeaway for the project. For any future kernel that uses @private/MArray scratch, the rule is: every array access inside the kernel —
  including @view, scalar index, getfield-into-array, anything — must be inside @inbounds. If the compiler can find one slow path that
  retains a reference to the MArray, it will refuse to promote it, and on GPU that turns into gpu_gc_pool_alloc. The diagnostic is the
  stack trace bottoming out at Scratchpad → MArray → gpu_gc_pool_alloc — that's the signal that escape analysis failed, not that Metal
  lacks private memory.

  I did not file a Metal.jl issue because there isn't really a Metal-side bug to file — the existing Scratchpad works once the user code
  respects the boundscheck discipline. Filing one suggesting a real emit_thread_memory intrinsic might still be worthwhile (it would skip
  the optimizer dependence), but it's an optimization not a correctness fix.

  Sources:
  - JuliaGPU/Metal.jl/src/MetalKernels.jl (Scratchpad override)
  - JuliaGPU/Metal.jl/src/device/pointer.jl (address space enum)
  - JuliaGPU/KernelAbstractions.jl/src/cpu.jl (aview comment)
  - JuliaGPU/CUDA.jl#2313 — Kernel using StaticArray compiles in 1.9 but not 1.10
  - JuliaLang/julia#39308 — view creates a boundscheck which captures A
  - JuliaGPU/GPUCompiler.jl#559 — Update New PassManager Pipeline
