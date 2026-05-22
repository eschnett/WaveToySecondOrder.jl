# Tests for `src/operators.jl`: element / domain construction, `SBPOps`
# typing, and the SBP-property identities. (The polynomial-exactness
# assertions inside `_make_operators` already verify these in the Rational
# branch at construction time; here we add explicit external checks.)

using WaveToySecondOrder
using WaveToySecondOrder: make_element, make_domain, make_operators, SBPOps
using LinearAlgebra
using StaticArrays
using Test

@testset "operators" begin

    @testset "make_element / make_domain" begin
        # Rational branch: equispaced "vertex-centred" nodes on [0, 1].
        e = make_element(Rational{Int64}, 5)
        @test e.xs == 0//1 : 1//4 : 1//1
        @test e.h  == 1//4

        # Float64 branch: GLL collocation, h = minimum diff between nodes.
        ef = make_element(Float64, 5)
        @test length(ef.xs) == 5
        @test ef.xs[1]    ≈ 0.0
        @test ef.xs[end]  ≈ 1.0
        @test ef.h        ≈ minimum(diff(ef.xs))
        # GLL nodes for N=5 cluster near the endpoints — first interior
        # gap is smaller than the central gap.
        @test diff(ef.xs)[1] < diff(ef.xs)[2]

        # Domain: N cell-centred elements covering [x0, x1].
        d = make_domain(Float64, 8, 0.0, 1.0)
        @test d.N == 8
        @test d.h ≈ 1/8
        @test length(d.xs) == 8
    end

    @testset "SBPOps types and structure (Rational)" begin
        elem = make_element(Rational{Int64}, 5)
        ops  = make_operators(elem)
        @test ops isa SBPOps{5, Rational{Int64}, 25}
        # Rational H comes from the Vandermonde construction → dense SMatrix.
        @test ops.H isa SMatrix{5, 5, Rational{Int64}, 25}
        @test ops.G isa SMatrix{5, 5, Rational{Int64}, 25}
        @test ops.L isa SMatrix{5, 5, Rational{Int64}, 25}
    end

    @testset "SBPOps types and structure (Float64 GLL)" begin
        elem = make_element(Float64, 5)
        ops  = make_operators(elem)
        @test ops isa SBPOps{5, Float64, 25}
        # GLL H is diagonal — stored as a `Diagonal{Float64, SVector{5, Float64}}`.
        @test ops.H    isa Diagonal{Float64, SVector{5, Float64}}
        @test ops.Hinv isa Diagonal{Float64, SVector{5, Float64}}
    end

    @testset "SBP property H·D + (H·G)ᵀ = B (Rational, exact)" begin
        elem = make_element(Rational{Int64}, 5)
        ops  = make_operators(elem)
        lhs = Matrix(ops.H * ops.D) + Matrix(ops.H * ops.G)'
        @test lhs == Matrix(ops.B)
    end

    @testset "SBP property H·D + (H·G)ᵀ ≈ B (Float64 GLL)" begin
        for N in (3, 5, 9, 13, 17)
            elem = make_element(Float64, N)
            ops  = make_operators(elem)
            lhs = Matrix(ops.H * ops.D) + Matrix(ops.H * ops.G)'
            @test maximum(abs.(lhs - Matrix(ops.B))) < 1e-12
        end
    end

    @testset "L = D·G (Laplacian factorisation)" begin
        for T in (Rational{Int64}, Float64)
            elem = make_element(T, 5)
            ops  = make_operators(elem)
            @test ops.L == ops.D * ops.G
        end
    end

end
