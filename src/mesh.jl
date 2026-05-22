# Mesh topology + geometry for a conforming hexahedral element mesh.
#
# The mesh is the data layer that decouples the 3D kernels from any
# particular grid arrangement: instead of indexing neighbours via
# `(mx ± 1, my, mz)` tuples on a 3D lattice, the per-element loop walks
# a 1D list of elements and asks the mesh for each element's six
# neighbours and the orientation of each face. This is the prerequisite
# for unstructured meshes (cubed sphere, multi-block topologies, future
# adaptive refinement).
#
# Vertex storage follows the standard finite-element / Gmsh convention:
# one shared coordinate table `vertex_coords` of shape `(3, Nv)` over the
# `Nv` distinct mesh vertices, plus a per-element connectivity table
# `vertex_idx` of shape `(8, Ne)` giving the index into `vertex_coords`
# of each of the eight corners of each hex. Shared vertices on common
# faces of adjacent elements are stored once, not duplicated.
#
# Face index convention (used throughout for `neighbour`, `orientation`,
# `bdry`):
#
#     1 → −x face        2 → +x face
#     3 → −y face        4 → +y face
#     5 → −z face        6 → +z face
#
# Vertex index convention (8 corners of each hex, Gmsh-canonical ordering):
#
#     1: (−x, −y, −z)    2: (+x, −y, −z)
#     3: (+x, +y, −z)    4: (−x, +y, −z)
#     5: (−x, −y, +z)    6: (+x, −y, +z)
#     7: (+x, +y, +z)    8: (−x, +y, +z)

"""
    HexMesh{T}

Connectivity + geometry of a conforming hexahedral mesh.

# Fields

* `Ne :: Int` — number of elements.
* `neighbour :: Matrix{Int}` of shape `(6, Ne)` — element ID of the
  neighbour across each of the six faces (face ordering as above). `0`
  marks an outer-boundary face.
* `orientation :: Matrix{Int8}` of shape `(6, Ne)` — `0..7` encoding the
  rotation + reflection that maps this face's local `(p, q)` coordinates
  to the neighbour's. `0` everywhere on axis-aligned meshes.
* `bdry :: Matrix{Int8}` of shape `(6, Ne)` — boundary-condition tag,
  nonzero only on outer faces.
* `vertex_coords :: Matrix{T}` of shape `(3, Nv)` — Cartesian coordinates
  of every distinct vertex in the mesh. Shared between adjacent elements.
* `vertex_idx :: Matrix{Int}` of shape `(8, Ne)` — for each element, the
  indices into `vertex_coords` of its eight corners (in the canonical
  vertex ordering above).
"""
struct HexMesh{T}
    Ne            :: Int
    neighbour     :: Matrix{Int}
    orientation   :: Matrix{Int8}
    bdry          :: Matrix{Int8}
    vertex_coords :: Matrix{T}
    vertex_idx    :: Matrix{Int}
end

"""
    nv(mesh::HexMesh) → Int

Number of distinct mesh vertices.
"""
nv(mesh::HexMesh) = size(mesh.vertex_coords, 2)

"""
    make_cubical_mesh(::Type{T}, Mx, My, Mz, x0, x1) → HexMesh{T}
    make_cubical_mesh(::Type{T}, M, x0, x1)         → HexMesh{T}

Axis-aligned conforming hex mesh of the cuboid `[x0, x1]³` with
`Mx × My × Mz` (or `M × M × M`) elements.

Element ordering is column-major over `(mx, my, mz)`:

    e(mx, my, mz) = mx + (my-1)·Mx + (mz-1)·Mx·My

Vertex ordering is column-major over `(vx, vy, vz)` with `Mx+1`,
`My+1`, `Mz+1` vertices along each axis:

    v(vx, vy, vz) = vx + (vy-1)·(Mx+1) + (vz-1)·(Mx+1)·(My+1)

Inter-element faces are axis-aligned, so `orientation` is identically
zero. Outer faces are tagged by axis/direction:

    −x → 1     +x → 2     −y → 3     +y → 4     −z → 5     +z → 6
"""
function make_cubical_mesh(::Type{T}, Mx::Int, My::Int, Mz::Int, x0, x1) where {T}
    @assert Mx ≥ 1 && My ≥ 1 && Mz ≥ 1
    Ne = Mx * My * Mz
    hx = (x1 - x0) / Mx
    hy = (x1 - x0) / My
    hz = (x1 - x0) / Mz

    # Number of distinct vertices along each axis (one more than elements).
    Nvx, Nvy, Nvz = Mx + 1, My + 1, Mz + 1
    Nv = Nvx * Nvy * Nvz

    neighbour     = zeros(Int,  6, Ne)
    orientation   = zeros(Int8, 6, Ne)
    bdry          = zeros(Int8, 6, Ne)
    vertex_coords = Matrix{T}(undef, 3, Nv)
    vertex_idx    = Matrix{Int}(undef, 8, Ne)

    # --- shared vertex grid ------------------------------------------------
    vidx(vx, vy, vz) = vx + (vy - 1) * Nvx + (vz - 1) * Nvx * Nvy
    for vz in 1:Nvz, vy in 1:Nvy, vx in 1:Nvx
        v = vidx(vx, vy, vz)
        vertex_coords[1, v] = T(x0) + (vx - 1) * T(hx)
        vertex_coords[2, v] = T(x0) + (vy - 1) * T(hy)
        vertex_coords[3, v] = T(x0) + (vz - 1) * T(hz)
    end

    # --- per-element connectivity + neighbour table ------------------------
    lidx(mx, my, mz) = mx + (my - 1) * Mx + (mz - 1) * Mx * My

    for mz in 1:Mz, my in 1:My, mx in 1:Mx
        e = lidx(mx, my, mz)

        # Neighbours and boundary tags
        if mx == 1;   bdry[1, e] = 1;  else  neighbour[1, e] = lidx(mx-1, my, mz);  end
        if mx == Mx;  bdry[2, e] = 2;  else  neighbour[2, e] = lidx(mx+1, my, mz);  end
        if my == 1;   bdry[3, e] = 3;  else  neighbour[3, e] = lidx(mx, my-1, mz);  end
        if my == My;  bdry[4, e] = 4;  else  neighbour[4, e] = lidx(mx, my+1, mz);  end
        if mz == 1;   bdry[5, e] = 5;  else  neighbour[5, e] = lidx(mx, my, mz-1);  end
        if mz == Mz;  bdry[6, e] = 6;  else  neighbour[6, e] = lidx(mx, my, mz+1);  end

        # Eight corner indices in canonical (Gmsh) ordering.
        vertex_idx[1, e] = vidx(mx,   my,   mz  )
        vertex_idx[2, e] = vidx(mx+1, my,   mz  )
        vertex_idx[3, e] = vidx(mx+1, my+1, mz  )
        vertex_idx[4, e] = vidx(mx,   my+1, mz  )
        vertex_idx[5, e] = vidx(mx,   my,   mz+1)
        vertex_idx[6, e] = vidx(mx+1, my,   mz+1)
        vertex_idx[7, e] = vidx(mx+1, my+1, mz+1)
        vertex_idx[8, e] = vidx(mx,   my+1, mz+1)
    end

    return HexMesh{T}(Ne, neighbour, orientation, bdry, vertex_coords, vertex_idx)
end

# Cubic convenience: equal element count in each direction.
make_cubical_mesh(::Type{T}, M::Int, x0, x1) where {T} =
    make_cubical_mesh(T, M, M, M, x0, x1)

"""
    element_coords(mesh, elem) → Array{T, 5}   # shape (3, N, N, N, Ne)

Cartesian coordinates of every GLL node of every element, obtained by
trilinear interpolation between the eight corner vertices of each hex
using the reference-element nodes `elem.xs` ∈ [0, 1] as the local
parametric coordinates.

For axis-aligned hexes (e.g. those produced by `make_cubical_mesh`) the
result is separable in the three reference axes; for curved hexes
(future: cubed-sphere blocks) the same formula gives the genuine
trilinear map without any special-casing.
"""
function element_coords(mesh::HexMesh{T}, elem) where {T}
    N = elem.N
    ξs = elem.xs
    coords = Array{T, 5}(undef, 3, N, N, N, mesh.Ne)
    @inbounds for e in 1:mesh.Ne
        # Look up the eight corner coordinates via the shared table.
        vi = ntuple(v -> mesh.vertex_idx[v, e], 8)
        vx = ntuple(v -> mesh.vertex_coords[1, vi[v]], 8)
        vy = ntuple(v -> mesh.vertex_coords[2, vi[v]], 8)
        vz = ntuple(v -> mesh.vertex_coords[3, vi[v]], 8)
        for k in 1:N, j in 1:N, i in 1:N
            ξ, η, ζ = ξs[i], ξs[j], ξs[k]
            mξ, mη, mζ = 1 - ξ, 1 - η, 1 - ζ
            # Trilinear shape functions for the eight corners.
            n1 = mξ * mη * mζ;  n2 = ξ * mη * mζ
            n3 = ξ  *  η * mζ;  n4 = mξ *  η * mζ
            n5 = mξ * mη *  ζ;  n6 = ξ * mη *  ζ
            n7 = ξ  *  η *  ζ;  n8 = mξ *  η *  ζ
            coords[1, i, j, k, e] = n1*vx[1] + n2*vx[2] + n3*vx[3] + n4*vx[4] +
                                    n5*vx[5] + n6*vx[6] + n7*vx[7] + n8*vx[8]
            coords[2, i, j, k, e] = n1*vy[1] + n2*vy[2] + n3*vy[3] + n4*vy[4] +
                                    n5*vy[5] + n6*vy[6] + n7*vy[7] + n8*vy[8]
            coords[3, i, j, k, e] = n1*vz[1] + n2*vz[2] + n3*vz[3] + n4*vz[4] +
                                    n5*vz[5] + n6*vz[6] + n7*vz[7] + n8*vz[8]
        end
    end
    return coords
end
