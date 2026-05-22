# Tests for `src/mesh.jl`: `HexMesh`, `make_cubical_mesh`.

using WaveToySecondOrder
using WaveToySecondOrder: HexMesh, make_cubical_mesh, make_inflated_cube_mesh, nv
using Test

count_zero_neighbours(m::HexMesh) = count(==(0), m.neighbour)

@testset "mesh" begin

    @testset "make_cubical_mesh: shapes" begin
        m = make_cubical_mesh(Float64, 2, 3, 4, 0.0, 1.0)
        @test m isa HexMesh{Float64}
        @test m.Ne == 2 * 3 * 4
        @test size(m.neighbour)     == (6, m.Ne)
        @test size(m.orientation)   == (6, m.Ne)
        @test size(m.bdry)          == (6, m.Ne)
        @test size(m.vertex_coords) == (3, (2+1)*(3+1)*(4+1))
        @test size(m.vertex_idx)    == (8, m.Ne)
        @test nv(m) == (2+1)*(3+1)*(4+1)
    end

    @testset "shared vertices: adjacent elements reuse the same vertex ID" begin
        # For an `Mx × My × Mz` mesh, the total number of vertices is
        # (Mx+1)(My+1)(Mz+1), much less than 8·Ne — sharing must be exact.
        m = make_cubical_mesh(Float64, 3, 3, 3, 0.0, 1.0)
        @test nv(m) == 4 * 4 * 4
        @test nv(m) < 8 * m.Ne                # i.e. real sharing happened

        # Element 1 (mx=my=mz=1) shares its +x face (vertices 2,3,6,7 of
        # element 1) with element 2 (mx=2,my=1,mz=1), which on its −x face
        # owns vertices 1,4,5,8. The matching pairs must reference the
        # SAME global vertex IDs.
        @test m.vertex_idx[2, 1] == m.vertex_idx[1, 2]
        @test m.vertex_idx[3, 1] == m.vertex_idx[4, 2]
        @test m.vertex_idx[6, 1] == m.vertex_idx[5, 2]
        @test m.vertex_idx[7, 1] == m.vertex_idx[8, 2]
    end

    @testset "make_cubical_mesh: orientation is always 0 (axis-aligned)" begin
        m = make_cubical_mesh(Float64, 4, 0.0, 1.0)
        @test all(m.orientation .== 0)
    end

    @testset "make_cubical_mesh: boundary tags only on outer faces" begin
        Mx, My, Mz = 3, 2, 4
        m = make_cubical_mesh(Float64, Mx, My, Mz, 0.0, 1.0)
        # Helper: linear index from (mx, my, mz).
        lidx(mx, my, mz) = mx + (my-1)*Mx + (mz-1)*Mx*My
        # Inspect every element.
        for mz in 1:Mz, my in 1:My, mx in 1:Mx
            e = lidx(mx, my, mz)
            # On each face, bdry ≠ 0 iff neighbour == 0 — the two together
            # exactly partition the six face slots.
            for f in 1:6
                @test (m.bdry[f, e] ≠ 0) == (m.neighbour[f, e] == 0)
            end
            # The outer-tag values match the face index for the cubical mesh.
            mx == 1  && @test m.bdry[1, e] == 1
            mx == Mx && @test m.bdry[2, e] == 2
            my == 1  && @test m.bdry[3, e] == 3
            my == My && @test m.bdry[4, e] == 4
            mz == 1  && @test m.bdry[5, e] == 5
            mz == Mz && @test m.bdry[6, e] == 6
        end
    end

    @testset "make_cubical_mesh: neighbour pointers are symmetric" begin
        # If element A says "my +x neighbour is B", then B must say
        # "my −x neighbour is A". Same for ±y and ±z.
        m = make_cubical_mesh(Float64, 3, 0.0, 1.0)
        opposite = (2, 1, 4, 3, 6, 5)   # opposite face for each of 1..6
        for e in 1:m.Ne
            for f in 1:6
                n = m.neighbour[f, e]
                n == 0 && continue
                @test m.neighbour[opposite[f], n] == e
            end
        end
    end

    @testset "make_cubical_mesh: vertex coordinates" begin
        # A 2×2×2 cubical mesh of [0, 2]³ should have element 1 (mx=my=mz=1)
        # occupying [0,1]³ with vertices in Gmsh-canonical ordering.
        m = make_cubical_mesh(Float64, 2, 0.0, 2.0)

        # Look up element 1's eight corner positions via the shared table.
        corner(e, v) = m.vertex_coords[:, m.vertex_idx[v, e]]
        @test corner(1, 1) == [0.0, 0.0, 0.0]
        @test corner(1, 2) == [1.0, 0.0, 0.0]
        @test corner(1, 3) == [1.0, 1.0, 0.0]
        @test corner(1, 4) == [0.0, 1.0, 0.0]
        @test corner(1, 5) == [0.0, 0.0, 1.0]
        @test corner(1, 6) == [1.0, 0.0, 1.0]
        @test corner(1, 7) == [1.0, 1.0, 1.0]
        @test corner(1, 8) == [0.0, 1.0, 1.0]

        # The element diagonally opposite (mx=my=mz=2) should occupy [1,2]³.
        e_far = 2 + 1*2 + 1*4     # = 8
        @test corner(e_far, 1) == [1.0, 1.0, 1.0]
        @test corner(e_far, 7) == [2.0, 2.0, 2.0]
    end

    @testset "make_inflated_cube_mesh: neighbour count by topology" begin
        # Build the 7-patch inflated cube. The outer surface is the cube
        # [-1, 1]³ tiled by 6·N² quads, one per outermost-shell element of
        # the six outer patches. The four "side" faces of each outer patch
        # are shared with the four adjacent outer patches along the cube
        # edges, so each outer-shell element loses exactly one neighbour
        # (the radially outward one). All other elements (the inner cube
        # and the non-outermost outer-patch layers) are fully surrounded.
        #
        # Per-element neighbour counts therefore split into just two bins:
        #   outer-shell (one outer face) → 5
        #   everything else              → 6
        N, R = 4, 0.1
        m = make_inflated_cube_mesh(Float64, N, R)

        # Each element bumps each of its neighbours' counters. By neighbour
        # symmetry this equals the per-element count of non-zero neighbour
        # slots, but we follow the prompt and accumulate explicitly.
        count = zeros(Int, m.Ne)
        for e in 1:m.Ne, f in 1:6
            n = m.neighbour[f, e]
            n == 0 && continue
            count[n] += 1
        end

        # Histogram and sanity bounds.
        @test all(3 .≤ count .≤ 6)
        n3 = sum(count .== 3)
        n4 = sum(count .== 4)
        n5 = sum(count .== 5)
        n6 = sum(count .== 6)
        @test n3 + n4 + n5 + n6 == m.Ne

        @test n3 == 0
        @test n4 == 0
        @test n5 == 6 * N^2
        @test n6 == m.Ne - n5

        # Cross-check: total outer-face slots equals tally of missing neighbours.
        outer_slots = 3*n3 + 2*n4 + 1*n5
        @test outer_slots == 6 * N^2
        @test outer_slots == count_zero_neighbours(m)
    end

end
