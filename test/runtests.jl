# Parallel test driver (ReTestItems). Each `*_tests.jl` file is a single
# `@testitem`; ReTestItems runs them across worker processes — the real
# speed-up for this compile-bound suite, since Julia's codegen lock means
# OS threads don't parallelise compilation, and the Metal GPU items
# aren't thread-safe. Per-item output is buffered and printed as a
# coherent block (out of order across items, never interleaved).

using Pkg

# Conditional GPU backend: on Apple Silicon add Metal so the worker
# processes inherit it (the `:gpu`-tagged items load it via a guarded
# `using Metal`; they no-op without functional hardware). Don't add it
# unconditionally — Metal.jl only instantiates on apple/aarch64.
if Sys.isapple() && Sys.ARCH === :aarch64
    Pkg.add("Metal")
end

using ReTestItems
using WaveToySecondOrder

# One worker process per core (capped at 8); one thread each — the work
# is compile- and BLAS-bound, not thread-parallel, so extra worker
# threads only oversubscribe the cores.
runtests(WaveToySecondOrder;
         nworkers = min(Sys.CPU_THREADS, 8),
         nworker_threads = 1)
