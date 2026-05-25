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
* `neighbour :: Matrix{Int32}` of shape `(6, Ne)` — element ID of the
  neighbour across each of the six faces (face ordering as above). `0`
  marks an outer-boundary face. Stored as `Int32` rather than `Int` so
  that `nbr` reads from inside the GPU kernel stay 32-bit (Apple
  Silicon GPUs emulate 64-bit integer arithmetic, which costs ~4× on
  every 4-D array offset computation in `_add_face_sat!`). 2^31 ≫
  any realistic element count, so the narrower type is safe.
* `neighbour_face :: Matrix{Int8}` of shape `(6, Ne)` — face index of
  the neighbour element that abuts face `f` of `e`. For an axis-aligned
  cubical mesh this is just the opposite face index along the same axis
  (`(2,1,4,3,6,5)[f]`); for multi-patch meshes where the two sides have
  different orthogonal-axis conventions it can be any of the six values.
  `0` on outer-boundary faces (where `neighbour == 0`).
* `orientation :: Matrix{Int8}` of shape `(6, Ne)` — `0..7` encoding of
  the D₄ transform that maps this face's local face-quadrature `(p, q)`
  coordinates to the matching face on the neighbour. `0` is the identity
  (used everywhere on axis-aligned meshes and on the cubed-cube mesh
  by construction). The transform table is documented in `_neigh_pq`
  below.
* `bdry :: Matrix{Int8}` of shape `(6, Ne)` — boundary-condition tag,
  nonzero only on outer faces.
* `vertex_coords :: Matrix{T}` of shape `(3, Nv)` — Cartesian coordinates
  of every distinct vertex in the mesh. Shared between adjacent elements.
* `vertex_idx :: Matrix{Int}` of shape `(8, Ne)` — for each element, the
  indices into `vertex_coords` of its eight corners (in the canonical
  vertex ordering above).
"""
# `MeshConnectivity` bundles only what the kernel reads from a mesh: the
# four connectivity matrices, indexed by `(face, element)`. Whoever
# launches the kernel sees this as one bitstype-friendly argument, so
# GPU adaptation can replace its arrays with `MtlDeviceMatrix` (etc.)
# in one shot. `HexMesh` (below) embeds a `MeshConnectivity` plus the
# host-only vertex metadata that the kernel never touches.
struct MeshConnectivity{MI, MI8}
    neighbour      :: MI
    neighbour_face :: MI8
    orientation    :: MI8
    bdry           :: MI8
end

# `HexMesh{T}` is parametrised on the *concrete* storage types of the
# kernel-read connectivity matrices (via `conn::MeshConnectivity`) so
# that they may live on a GPU as `CuArray`, `MtlArray`, `ROCArray`, etc.
# Host-only fields (`vertex_coords`, `vertex_idx`) stay concrete
# `Matrix` — they are read by plotting and diagnostics, never by the
# kernel, so there is no benefit to migrating them onto a device.
# A `Base.getproperty` forwarder below preserves `mesh.neighbour` /
# `mesh.bdry` / etc. so existing host code does not need updating.
struct HexMesh{T, MI, MI8}
    Ne            :: Int
    conn          :: MeshConnectivity{MI, MI8}
    vertex_coords :: Matrix{T}
    vertex_idx    :: Matrix{Int}

    function HexMesh{T}(Ne::Int,
                        conn::MeshConnectivity{MI, MI8},
                        vertex_coords::Matrix{T},
                        vertex_idx::Matrix{Int}) where {T, MI, MI8}
        new{T, MI, MI8}(Ne, conn, vertex_coords, vertex_idx)
    end

    # Back-compat constructor matching the old flat-field signature.
    function HexMesh{T}(Ne::Int,
                        neighbour::MI, neighbour_face::MI8,
                        orientation::MI8, bdry::MI8,
                        vertex_coords::Matrix{T},
                        vertex_idx::Matrix{Int}) where {T, MI, MI8}
        new{T, MI, MI8}(Ne,
                        MeshConnectivity{MI, MI8}(neighbour, neighbour_face,
                                                  orientation, bdry),
                        vertex_coords, vertex_idx)
    end
end

# `mesh.neighbour` etc. forward to `mesh.conn.*` so existing call sites
# (mesh-build code, tests, diagnostics) keep working without churn.
@inline function Base.getproperty(m::HexMesh, name::Symbol)
    if name === :neighbour || name === :neighbour_face ||
       name === :orientation || name === :bdry
        getfield(getfield(m, :conn), name)
    else
        getfield(m, name)
    end
end
Base.propertynames(m::HexMesh) = (:Ne, :conn, :vertex_coords, :vertex_idx,
                                  :neighbour, :neighbour_face, :orientation, :bdry)

"""
    nv(mesh::HexMesh) → Int

Number of distinct mesh vertices.
"""
nv(mesh::HexMesh) = size(mesh.vertex_coords, 2)

# ----------------------------------------------------------------------
# Skeleton-based mesh construction
#
# Two-stage mesh build that uses *integer-only* dedup for vertex
# identification. Stage 1 builds a `SkeletonMesh` — a small structure
# carrying the patch list and inter-patch face connectivity at the
# combinatorial level (no floating-point coordinates). Stage 2
# (`_skeleton_to_mesh`) enumerates per-patch vertices as integer
# 4-tuples `(p, i, j, k)`, unifies face-shared ids via union-find,
# assigns dense canonical ids, and only then evaluates the family-
# specific parametric map to produce coordinates.
#
# This replaces the earlier position-keyed `Dict{NTuple{3, T}, Int}`
# dedup, which broke at M=4 inflated cube because cube and patch
# vertex positions computed via different floating-point expressions
# rounded to 1-ULP-different values for non-power-of-2 divisions
# (e.g. `0.05000000000000002` vs `0.05` for `L = 0.1`). Integer
# dedup eliminates that class of bug entirely.
#
# Step 1 of the cleanup wires `make_cubical_mesh` through this path;
# the cubed cube and inflated cube builders follow in later steps.

"""
    PatchSpec{T}

Combinatorial + parametric description of one patch in a multi-block
hex mesh.

# Fields

* `Ma, Mb, Mc :: Int` — element counts along the patch's three local
  axes. The patch contains `Ma·Mb·Mc` elements and
  `(Ma+1)·(Mb+1)·(Mc+1)` pre-dedup vertices.
* `family :: Symbol` — selects the parametric vertex map used at
  coordinate-assignment time. Currently supported:
    * `:cubical` — axis-aligned `[a_lo, a_hi] × [b_lo, b_hi] × [c_lo, c_hi]`.
  Curvilinear families (`:inflation_*`, `:shell_*`) will be added when
  `make_cubed_cube_mesh` and `make_inflated_cube_mesh` are rewired.
* `a_lo, a_hi, b_lo, b_hi, c_lo, c_hi :: T` — affine ranges in the
  patch's local `(a, b, c)` parameter space. The reference cube
  `[0, 1]³` is mapped to these before the family-specific transform.
* `L, R1, R2 :: T` — analytic constants used by curvilinear families;
  zero for `:cubical`. Stored on every `PatchSpec` so each patch
  carries everything it needs to evaluate its own vertices.
"""
struct PatchSpec{T}
    Ma     :: Int
    Mb     :: Int
    Mc     :: Int
    family :: Symbol
    a_lo   :: T
    a_hi   :: T
    b_lo   :: T
    b_hi   :: T
    c_lo   :: T
    c_hi   :: T
    L      :: T
    R1     :: T
    R2     :: T
end

"""
    FaceLink

One entry in a `SkeletonMesh`'s 6×n_patches face-connectivity table.
Two flavours, selected by `kind`:

* `kind = :interior` — face is shared with another patch face. Carries
  `(neigh_patch, neigh_face, orientation ∈ 0..7)`.
* `kind = :boundary` — face is on the domain boundary. Carries only
  `boundary_tag ∈ 1..127`.

Use `interior_link(np, nf, o)` / `boundary_link(tag)` to construct.
"""
struct FaceLink
    kind         :: Symbol
    neigh_patch  :: Int
    neigh_face   :: Int
    orientation  :: Int8
    boundary_tag :: Int8
end

interior_link(np::Integer, nf::Integer, o::Integer) =
    FaceLink(:interior, Int(np), Int(nf), Int8(o), Int8(0))
boundary_link(tag::Integer) =
    FaceLink(:boundary, 0, 0, Int8(0), Int8(tag))

"""
    SkeletonMesh{T}

Patch list + 6×n_patches face-link table. `_skeleton_to_mesh(skel)`
instantiates the full `HexMesh{T}` from this skeleton.
"""
struct SkeletonMesh{T}
    patches :: Vector{PatchSpec{T}}
    faces   :: Matrix{FaceLink}
end

# ----- Per-face index helpers (skeleton scope) ------------------------

# Element counts along the two tangent axes of face `f` in patch `ps`.
@inline function _face_tangent_counts(f::Integer, ps::PatchSpec)
    if f == 1 || f == 2
        return (ps.Mb, ps.Mc)
    elseif f == 3 || f == 4
        return (ps.Ma, ps.Mc)
    else                       # f == 5 or 6
        return (ps.Ma, ps.Mb)
    end
end

# Face-local 0-based vertex `(pp, qq)` → patch vertex `(i, j, k)`.
@inline function _face_vert_to_ijk(f::Integer, pp::Integer, qq::Integer,
                                     ps::PatchSpec)
    if f == 1
        return (0,     pp,    qq   )
    elseif f == 2
        return (ps.Ma, pp,    qq   )
    elseif f == 3
        return (pp,    0,     qq   )
    elseif f == 4
        return (pp,    ps.Mb, qq   )
    elseif f == 5
        return (pp,    qq,    0    )
    else
        return (pp,    qq,    ps.Mc)
    end
end

# Element `(a, b, c)` projected onto face `f` → face-cell `(p_cell, q_cell)`.
@inline function _face_cell_to_pq(f::Integer, a::Integer, b::Integer, c::Integer)
    if f == 1 || f == 2
        return (b, c)
    elseif f == 3 || f == 4
        return (a, c)
    else
        return (a, b)
    end
end

# Face-cell `(p_cell, q_cell)` on face `f` → element `(a, b, c)`.
@inline function _face_cell_to_abc(f::Integer, p_cell::Integer, q_cell::Integer,
                                    ps::PatchSpec)
    if f == 1
        return (1,      p_cell, q_cell)
    elseif f == 2
        return (ps.Ma,  p_cell, q_cell)
    elseif f == 3
        return (p_cell, 1,      q_cell)
    elseif f == 4
        return (p_cell, ps.Mb,  q_cell)
    elseif f == 5
        return (p_cell, q_cell, 1     )
    else
        return (p_cell, q_cell, ps.Mc )
    end
end

# D₄ transform on 0-indexed face-vertex coordinates `(p, q) ∈ 0..Mt1 × 0..Mt2`.
# Even `o` preserves the (p, q) dim ordering, odd `o` swaps it
# (so the neighbour's tangent counts come out as `(Mt2, Mt1)`).
@inline function _neigh_pq_vertex(o::Integer, p::Integer, q::Integer,
                                    Mt1::Integer, Mt2::Integer)
    if     o == 0;  return (p,        q       )
    elseif o == 1;  return (q,        Mt1 - p )
    elseif o == 2;  return (Mt1 - p,  Mt2 - q )
    elseif o == 3;  return (Mt2 - q,  p       )
    elseif o == 4;  return (Mt1 - p,  q       )
    elseif o == 5;  return (q,        p       )
    elseif o == 6;  return (p,        Mt2 - q )
    else            return (Mt2 - q,  Mt1 - p )
    end
end

# Same D₄ transform on 1-indexed face cells, `(b, c) ∈ 1..Mt1 × 1..Mt2`,
# used to identify the neighbour element across a cross-patch face link.
@inline function _neigh_pq_cell(o::Integer, b::Integer, c::Integer,
                                  Mt1::Integer, Mt2::Integer)
    if     o == 0;  return (b,             c            )
    elseif o == 1;  return (c,             Mt1 + 1 - b  )
    elseif o == 2;  return (Mt1 + 1 - b,   Mt2 + 1 - c  )
    elseif o == 3;  return (Mt2 + 1 - c,   b            )
    elseif o == 4;  return (Mt1 + 1 - b,   c            )
    elseif o == 5;  return (c,             b            )
    elseif o == 6;  return (b,             Mt2 + 1 - c  )
    else            return (Mt2 + 1 - c,   Mt1 + 1 - b  )
    end
end

# Family-dispatched coordinate map: given integer vertex
# `(i, j, k) ∈ 0..Ma × 0..Mb × 0..Mc` of patch `ps`, return the
# physical `(x, y, z)`. Each family interprets `(a_lo, a_hi, b_lo,
# b_hi, c_lo, c_hi)` and `(L, R1, R2)` according to its own
# parameterisation.
function _patch_vertex_position(ps::PatchSpec{T}, i::Integer,
                                  j::Integer, k::Integer) where {T}
    if ps.family === :cubical
        ξ = T(i) / T(ps.Ma)
        η = T(j) / T(ps.Mb)
        ζ = T(k) / T(ps.Mc)
        return (ps.a_lo + (ps.a_hi - ps.a_lo) * ξ,
                ps.b_lo + (ps.b_hi - ps.b_lo) * η,
                ps.c_lo + (ps.c_hi - ps.c_lo) * ζ)
    elseif ps.family === :wedge_pos_x || ps.family === :wedge_neg_x ||
           ps.family === :wedge_pos_y || ps.family === :wedge_neg_y ||
           ps.family === :wedge_pos_z || ps.family === :wedge_neg_z
        # Radial wedge (cubed-cube outer patch). Geometric radial
        # spacing `r(a) = R1·(R2/R1)^a` so each cell stays roughly
        # cubical; angular axes `(b, c) ∈ [-1, 1]²`.
        a = ps.a_lo + (ps.a_hi - ps.a_lo) * (T(i) / T(ps.Ma))
        b = ps.b_lo + (ps.b_hi - ps.b_lo) * (T(j) / T(ps.Mb))
        c = ps.c_lo + (ps.c_hi - ps.c_lo) * (T(k) / T(ps.Mc))
        r = ps.R1 * (ps.R2 / ps.R1)^a
        if ps.family === :wedge_pos_x
            return ( r,   b*r, c*r)
        elseif ps.family === :wedge_neg_x
            return (-r,   b*r, c*r)
        elseif ps.family === :wedge_pos_y
            return (b*r,  r,   c*r)
        elseif ps.family === :wedge_neg_y
            return (b*r, -r,   c*r)
        elseif ps.family === :wedge_pos_z
            return (b*r,  c*r,  r)
        else  # :wedge_neg_z
            return (b*r,  c*r, -r)
        end
    else
        # Inflation / shell families (13-patch inflated cube).
        # `kind ∈ 1..6` → inflation in +x/-x/+y/-y/+z/-z;
        # `kind ∈ 7..12` → spherical-shell in the same direction order.
        # Uses the right-handed `_patch_direction_vec` (with axis swaps
        # for `-x, +y, -z`) — the corresponding non-trivial D₄
        # orientations are encoded in the skeleton's face-link table.
        kind = _family_to_kind(ps.family)
        kind in Int8(1):Int8(12) || error("unknown patch family: $(ps.family)")
        a = ps.a_lo + (ps.a_hi - ps.a_lo) * (T(i) / T(ps.Ma))
        b = ps.b_lo + (ps.b_hi - ps.b_lo) * (T(j) / T(ps.Mb))
        c = ps.c_lo + (ps.c_hi - ps.c_lo) * (T(k) / T(ps.Mc))
        Q = sqrt(one(T) + (b * b + c * c))
        dir = ((kind - Int8(1)) % Int8(6)) + Int8(1)
        vx, vy, vz = _patch_direction_vec(dir, b, c)
        if kind ≥ Int8(7)  # shell
            r = (one(T) - a) * ps.R1 + a * ps.R2
            f = r / Q
        else               # inflation
            f = (one(T) - a) * ps.L + a * ps.R1 / Q
        end
        return (f * vx, f * vy, f * vz)
    end
end

"""
    _family_to_kind(family::Symbol) → Int8

Map a `PatchSpec.family` symbol to the integer `kind` tag used by
`PatchInfo` (which the analytic-Jacobian `make_geometry(::InflatedCubeMesh)`
dispatches on):

* `:cubical`               → `0`
* `:inflation_{pos,neg}_{x,y,z}` → `1..6`
* `:shell_{pos,neg}_{x,y,z}`     → `7..12`
* anything else            → `-1` (treated as "no PatchInfo needed")
"""
function _family_to_kind(family::Symbol)
    family === :cubical          ? Int8(0)  :
    family === :inflation_pos_x  ? Int8(1)  :
    family === :inflation_neg_x  ? Int8(2)  :
    family === :inflation_pos_y  ? Int8(3)  :
    family === :inflation_neg_y  ? Int8(4)  :
    family === :inflation_pos_z  ? Int8(5)  :
    family === :inflation_neg_z  ? Int8(6)  :
    family === :shell_pos_x      ? Int8(7)  :
    family === :shell_neg_x      ? Int8(8)  :
    family === :shell_pos_y      ? Int8(9)  :
    family === :shell_neg_y      ? Int8(10) :
    family === :shell_pos_z      ? Int8(11) :
    family === :shell_neg_z      ? Int8(12) :
    Int8(-1)
end

"""
    _skeleton_to_mesh(skel::SkeletonMesh{T}) → HexMesh{T}

Instantiate the full element-level `HexMesh{T}` from a `SkeletonMesh`:

1. Per-patch pre-dedup vertex ids `(p, i, j, k)`.
2. Union-find over face-shared ids using the skeleton's interior
   `FaceLink`s and the integer D₄ orientation transform.
3. Dense canonical vertex ids `1..Nv`.
4. Coordinates from `_patch_vertex_position` at one representative
   per canonical id.
5. Per-element `vertex_idx`, `neighbour`, `neighbour_face`,
   `orientation`, `bdry` tables — within-patch faces use trivial
   sibling-element connectivity; cross-patch faces inherit
   `(neighbour_face, orientation)` from the skeleton.

No floating-point comparison is ever load-bearing.
"""
function _skeleton_to_mesh(skel::SkeletonMesh{T}) where {T}
    n_patches = length(skel.patches)
    @assert size(skel.faces) == (6, n_patches)

    # Pre-dedup vertex / element offsets per patch.
    vert_offs = Vector{Int}(undef, n_patches + 1)
    elem_offs = Vector{Int}(undef, n_patches + 1)
    vert_offs[1] = 0
    elem_offs[1] = 0
    for p in 1:n_patches
        ps = skel.patches[p]
        vert_offs[p+1] = vert_offs[p] + (ps.Ma + 1) * (ps.Mb + 1) * (ps.Mc + 1)
        elem_offs[p+1] = elem_offs[p] + ps.Ma * ps.Mb * ps.Mc
    end
    Nv_pre = vert_offs[end]
    Ne     = elem_offs[end]

    # Pre-dedup vertex id for `(p, i, j, k)` and element id for `(p, a, b, c)`.
    @inline function vid(p, i, j, k)
        ps = skel.patches[p]
        return vert_offs[p] + 1 + i + (ps.Ma + 1) * (j + (ps.Mb + 1) * k)
    end
    @inline function eid(p, a, b, c)
        ps = skel.patches[p]
        return elem_offs[p] + a + ps.Ma * ((b - 1) + ps.Mb * (c - 1))
    end

    # --- Union-find (path-compression + rank) -------------------------
    parent = collect(1:Nv_pre)
    rank   = zeros(Int, Nv_pre)
    function uf_find(x)
        while parent[x] != x
            parent[x] = parent[parent[x]]
            x = parent[x]
        end
        return x
    end
    function uf_union!(x, y)
        rx, ry = uf_find(x), uf_find(y)
        rx == ry && return
        if     rank[rx] < rank[ry]; parent[rx] = ry
        elseif rank[rx] > rank[ry]; parent[ry] = rx
        else;                        parent[ry] = rx;  rank[rx] += 1
        end
        return
    end

    # Walk every interior face link once. The `(p, f) < (p2, f2)` guard
    # processes each pair exactly once even when the skeleton lists both
    # halves (which it normally does, for symmetry).
    for p in 1:n_patches, f in 1:6
        link = skel.faces[f, p]
        link.kind === :interior || continue
        p2 = link.neigh_patch
        f2 = link.neigh_face
        ((p, f) < (p2, f2)) || continue
        o   = Int(link.orientation)
        ps  = skel.patches[p]
        ps2 = skel.patches[p2]
        Mt1, Mt2 = _face_tangent_counts(f, ps)
        for qq in 0:Mt2, pp in 0:Mt1
            i1, j1, k1 = _face_vert_to_ijk(f, pp, qq, ps)
            pp2, qq2 = _neigh_pq_vertex(o, pp, qq, Mt1, Mt2)
            i2, j2, k2 = _face_vert_to_ijk(f2, pp2, qq2, ps2)
            uf_union!(vid(p, i1, j1, k1), vid(p2, i2, j2, k2))
        end
    end

    # --- Canonical dense ids -----------------------------------------
    Nv = 0
    canon = zeros(Int, Nv_pre)
    for x in 1:Nv_pre
        if uf_find(x) == x
            Nv += 1
            canon[x] = Nv
        end
    end
    final = Vector{Int}(undef, Nv_pre)
    for x in 1:Nv_pre
        final[x] = canon[uf_find(x)]
    end

    # --- Coordinates (one evaluation per canonical id) ---------------
    vertex_coords = Matrix{T}(undef, 3, Nv)
    written = falses(Nv)
    for p in 1:n_patches
        ps = skel.patches[p]
        for k in 0:ps.Mc, j in 0:ps.Mb, i in 0:ps.Ma
            id = final[vid(p, i, j, k)]
            written[id] && continue
            x, y, z = _patch_vertex_position(ps, i, j, k)
            vertex_coords[1, id] = x
            vertex_coords[2, id] = y
            vertex_coords[3, id] = z
            written[id] = true
        end
    end
    @assert all(written)

    # --- Per-element tables ------------------------------------------
    vertex_idx     = Matrix{Int}(undef, 8, Ne)
    neighbour      = zeros(Int32, 6, Ne)
    neighbour_face = zeros(Int8,  6, Ne)
    orientation    = zeros(Int8,  6, Ne)
    bdry           = zeros(Int8,  6, Ne)

    # Opposite face along the same axis: 1↔2 (-x/+x), 3↔4 (-y/+y), 5↔6 (-z/+z).
    OPP = (Int8(2), Int8(1), Int8(4), Int8(3), Int8(6), Int8(5))

    for p in 1:n_patches
        ps = skel.patches[p]
        for c in 1:ps.Mc, b in 1:ps.Mb, a in 1:ps.Ma
            e = eid(p, a, b, c)
            # Gmsh-canonical 8 corner vertex ids.
            vertex_idx[1, e] = final[vid(p, a - 1, b - 1, c - 1)]
            vertex_idx[2, e] = final[vid(p, a,     b - 1, c - 1)]
            vertex_idx[3, e] = final[vid(p, a,     b,     c - 1)]
            vertex_idx[4, e] = final[vid(p, a - 1, b,     c - 1)]
            vertex_idx[5, e] = final[vid(p, a - 1, b - 1, c    )]
            vertex_idx[6, e] = final[vid(p, a,     b - 1, c    )]
            vertex_idx[7, e] = final[vid(p, a,     b,     c    )]
            vertex_idx[8, e] = final[vid(p, a - 1, b,     c    )]

            # Per-face neighbour / orientation / bdry.
            for (f, na, nb, nc, at_patch_boundary) in (
                    (1, a - 1, b,     c,     a == 1     ),
                    (2, a + 1, b,     c,     a == ps.Ma ),
                    (3, a,     b - 1, c,     b == 1     ),
                    (4, a,     b + 1, c,     b == ps.Mb ),
                    (5, a,     b,     c - 1, c == 1     ),
                    (6, a,     b,     c + 1, c == ps.Mc ),
                )
                if !at_patch_boundary
                    neighbour[f, e]      = eid(p, na, nb, nc)
                    neighbour_face[f, e] = OPP[f]
                    orientation[f, e]    = Int8(0)
                else
                    link = skel.faces[f, p]
                    if link.kind === :boundary
                        bdry[f, e] = link.boundary_tag
                    else
                        p2 = link.neigh_patch
                        f2 = link.neigh_face
                        o  = Int(link.orientation)
                        ps2 = skel.patches[p2]
                        Mt1, Mt2 = _face_tangent_counts(f, ps)
                        p_cell, q_cell = _face_cell_to_pq(f, a, b, c)
                        p2c, q2c = _neigh_pq_cell(o, p_cell, q_cell, Mt1, Mt2)
                        a2, b2, c2 = _face_cell_to_abc(f2, p2c, q2c, ps2)
                        neighbour[f, e]      = eid(p2, a2, b2, c2)
                        neighbour_face[f, e] = Int8(f2)
                        orientation[f, e]    = Int8(o)
                    end
                end
            end
        end
    end

    return HexMesh{T}(Ne, neighbour, neighbour_face, orientation, bdry,
                      vertex_coords, vertex_idx)
end

"""
    make_cubical_mesh(::Type{T}, Mx, My, Mz, x0, x1) → HexMesh{T}
    make_cubical_mesh(::Type{T}, M, x0, x1)         → HexMesh{T}

Axis-aligned conforming hex mesh of the cuboid `[x0, x1]³` with
`Mx × My × Mz` (or `M × M × M`) elements. Backed by the skeleton-based
build (`_skeleton_to_mesh`) over a single `:cubical` patch with all
six faces tagged as domain boundary `1..6` (matching the face index
ordering, as before).

Element ordering remains column-major over `(mx, my, mz)`:

    e(mx, my, mz) = mx + (my-1)·Mx + (mz-1)·Mx·My

`orientation` is identically zero (axis-aligned, single patch).
"""
function make_cubical_mesh(::Type{T}, Mx::Int, My::Int, Mz::Int, x0, x1) where {T}
    @assert Mx ≥ 1 && My ≥ 1 && Mz ≥ 1
    z = zero(T)
    patch = PatchSpec{T}(Mx, My, Mz, :cubical,
                          T(x0), T(x1), T(x0), T(x1), T(x0), T(x1),
                          z, z, z)
    faces = Matrix{FaceLink}(undef, 6, 1)
    for f in 1:6
        faces[f, 1] = boundary_link(f)
    end
    skel = SkeletonMesh{T}([patch], faces)
    return _skeleton_to_mesh(skel)
end

# Cubic convenience: equal element count in each direction.
make_cubical_mesh(::Type{T}, M::Int, x0, x1) where {T} =
    make_cubical_mesh(T, M, M, M, x0, x1)

# D₄ orientation transform: maps self's face-local `(p, q)` into the
# neighbour's `(p, q)` using 1-indexed coordinates in `1..N`. Resolves
# the eight rotations + reflections of the unit square. Used by the
# kernel `_face_sat_compute!` to read across an inter-element face
# when the two sides' tangent axes don't line up directly.
#
# This is the 1-indexed cousin of `_neigh_pq_vertex` (0-indexed,
# 0..M) and `_neigh_pq_cell` (1-indexed cells, 1..M); the three are
# the same group operation re-expressed for the index range each
# caller works in.
@inline function _neigh_pq(o::Integer, p::Integer, q::Integer, N::Integer)
    # Mirror in whatever integer type `p, q, N` are passed as. Callers
    # in the kernel pass `Int32` to keep the index chain off the
    # 64-bit slow path on NVIDIA / Apple Silicon. The `one(N)` keeps
    # the "+1" in the same type as `N`.
    np1 = N + one(N)
    if     o == 0;  return (p,        q       )
    elseif o == 1;  return (q,        np1 - p)
    elseif o == 2;  return (np1 - p,  np1 - q)
    elseif o == 3;  return (np1 - q,  p       )
    elseif o == 4;  return (np1 - p,  q       )
    elseif o == 5;  return (q,        p       )
    elseif o == 6;  return (p,        np1 - q)
    else            return (np1 - q,  np1 - p)
    end
end

"""
    make_cubed_cube_mesh(::Type{T}, M::Int, R::Real) → HexMesh{T}

Conforming hex mesh of the cube `[-1, 1]³` built from a "cubed-sphere"
block topology applied to a cubic domain: one central cubic patch
`[-R, R]³` plus six radial-wedge patches connecting it to the six outer
cube faces. All outer faces of the global domain are flat (the overall
shape is still a cube), so the *outer* mesh boundary is `[-1, 1]³`
exactly — the geometry has cubed-sphere topology but a cube codomain.
(The name `inflated_cube` is reserved for a future variant in which the
outer boundary is curved to a sphere.)

`M` is the mesh-resolution parameter: each of the seven patches is
subdivided into `M` cells along each non-radial axis (so the inner patch
has `M³` cells and each outer patch has `L·M²`).

# Geometry

For the +x patch (the other five are obtained by axis permutation /
reflection), with local indices `i ∈ 0..L`, `j ∈ 0..M`, `k ∈ 0..M` and
`s_j = -1 + 2j/M`, `t_k = -1 + 2k/M`:

    r          = R · α^i        (radial coordinate)
    (x, y, z)  = (r,  s_j·r,  t_k·r)

so the cross-section at radial level `i` is the square `[-r, r]²`, which
matches the inner cube's `[-R, R]²` face at `i = 0` and the outer cube's
`[-1, 1]²` face at `i = L`.

# Element count

`M³` cells in the inner patch + `6·L·M²` cells in the six outer patches.

# `L` and radial spacing (step 2 of the construction)

We want each outer-patch cell to be roughly cubical: angular width
`2r/M` should match the radial width `r_{i+1} − r_i`. With geometric
spacing `r_i = R·α^i` the cell aspect is constant in `α`:

    r_{i+1} - r_i = r_i · (α - 1),  angular size = 2 r_i / M

so isotropic ⇒ `α - 1 ≈ 2/M`. The radial endpoint constraint `r_L = 1`
fixes `α = (1/R)^(1/L)`, so we pick

    L = round( log(1/R) / log(1 + 2/M) )

and use the resulting `α`. For `M = 5`, `R = 0.1` this gives `L = 7`,
`α ≈ 1.389`.

# Orientation

By construction, every patch's local axes are oriented so that, at any
shared face, the (p, q) face-node coordinates on the two sides match
directly — `orientation[f, e] = 0` everywhere.
"""
# Tangential-face connectivity for the six radial directions of a
# cubed-cube topology. Indexed by direction `dir ∈ 1..6` and face
# `f ∈ 3..6`:
#
#   `_WEDGE_NEIGHBOUR[dir][f - 2] = (neigh_dir, neigh_face)`
#
# All twelve cube-edge interfaces have orientation 0 because every
# wedge's tangent (p, q) local axes point in the same physical
# (x, y, z) direction at the shared edge (cubed-cube wedges use the
# uniform no-axis-swap convention `v = (±1, b, c)` etc.).
const _WEDGE_NEIGHBOUR = (
    ((4, 4), (3, 4), (6, 4), (5, 4)),   # +x: faces 3,4,5,6 → -y +y -z +z
    ((4, 3), (3, 3), (6, 3), (5, 3)),   # -x:                → -y +y -z +z
    ((2, 4), (1, 4), (6, 6), (5, 6)),   # +y:                → -x +x -z +z
    ((2, 3), (1, 3), (6, 5), (5, 5)),   # -y:                → -x +x -z +z
    ((2, 6), (1, 6), (4, 6), (3, 6)),   # +z:                → -x +x -y +y
    ((2, 5), (1, 5), (4, 5), (3, 5)),   # -z:                → -x +x -y +y
)

# Tangential-face connectivity for the inflation / shell patches of an
# inflated-cube mesh. Same indexing scheme as `_WEDGE_NEIGHBOUR` but
# DIFFERENT entries: the inflated cube's `_patch_direction_vec` is
# right-handed with axis-swaps for the negative-leading-axis directions
# (`-x: v = (-1, c, b)`, `+y: v = (c, 1, b)`, `-z: v = (c, b, -1)`),
# which changes which face of each neighbour patch meets a given cube
# edge. Derived once on paper from the patch parameterisations; all
# twelve cube-edge orientations remain 0 (verified by Gmsh vertex
# correspondence at each edge).
const _INFLATION_NEIGHBOUR = (
    ((4, 4), (3, 6), (6, 6), (5, 4)),   # +x: faces 3,4,5,6 → -y4 +y6 -z6 +z4
    ((6, 5), (5, 3), (4, 3), (3, 5)),   # -x:                → -z5 +z3 -y3 +y5
    ((6, 4), (5, 6), (2, 6), (1, 4)),   # +y:                → -z4 +z6 -x6 +x4
    ((2, 5), (1, 3), (6, 3), (5, 5)),   # -y:                → -x5 +x3 -z3 +z5
    ((2, 4), (1, 6), (4, 6), (3, 4)),   # +z:                → -x4 +x6 -y6 +y4
    ((4, 5), (3, 3), (2, 3), (1, 5)),   # -z:                → -y5 +y3 -x3 +x5
)

function _cubed_cube_skeleton(::Type{T}, M::Int, R::Real) where {T}
    @assert M ≥ 1
    @assert 0 < R < 1
    Rv = T(R)

    # Radial element count `L` per outer patch — same heuristic as the
    # pre-skeleton implementation: pick `L` such that the cell aspect
    # `(α - 1) ≈ 2/M` with `α = (1/R)^(1/L)`.
    L = max(1, round(Int, log(1/R) / log(1 + 2/M)))

    z = zero(T)
    o = one(T)

    # Patch 1: inner cube `[-R, R]³`.
    inner = PatchSpec{T}(M, M, M, :cubical,
                          -Rv, Rv, -Rv, Rv, -Rv, Rv,
                          z, z, z)

    # Patches 2..7: outer wedges in the dir order (+x, -x, +y, -y, +z, -z).
    # All wedges share the parameter ranges `a ∈ [0, 1]`, `b ∈ [-1, 1]`,
    # `c ∈ [-1, 1]`; the family selects the embedding direction. The
    # `(R1, R2)` slots hold the inner-cube half-edge and the outer-cube
    # half-edge (1), used by `_patch_vertex_position` to compute
    # `r(a) = R1 · (R2/R1)^a = R · (1/R)^a`.
    wedge_families = (:wedge_pos_x, :wedge_neg_x,
                      :wedge_pos_y, :wedge_neg_y,
                      :wedge_pos_z, :wedge_neg_z)
    patches = PatchSpec{T}[inner]
    for fam in wedge_families
        push!(patches, PatchSpec{T}(L, M, M, fam,
                                     z, o, -o, o, -o, o,
                                     z, Rv, o))
    end

    faces = Matrix{FaceLink}(undef, 6, length(patches))

    # ---- Cube ↔ wedge interfaces. ----
    # Inner cube face `f` → wedge whose direction matches that face,
    # connecting at the wedge's face 1 (`a = 0`, inner-radial face).
    # Vertex layout aligns so all six cube↔wedge interfaces have
    # orientation 0.
    #
    #   f=1 (-x) → -x wedge (patch 3)
    #   f=2 (+x) → +x wedge (patch 2)
    #   f=3 (-y) → -y wedge (patch 5)
    #   f=4 (+y) → +y wedge (patch 4)
    #   f=5 (-z) → -z wedge (patch 7)
    #   f=6 (+z) → +z wedge (patch 6)
    cube_face_to_wedge_patch = (3, 2, 5, 4, 7, 6)
    for f in 1:6
        wp = cube_face_to_wedge_patch[f]
        faces[f, 1]  = interior_link(wp, 1, 0)
        faces[1, wp] = interior_link(1, f, 0)
    end

    # ---- Outer-cube boundary tags on each wedge's face 2. ----
    # Match the original convention: -x=1, +x=2, -y=3, +y=4, -z=5, +z=6,
    # indexed by direction `dir ∈ 1..6` of the wedge.
    OUTER_TAG = (Int8(2), Int8(1), Int8(4), Int8(3), Int8(6), Int8(5))
    for dir in 1:6
        faces[2, dir + 1] = boundary_link(OUTER_TAG[dir])
    end

    # ---- Wedge ↔ wedge tangential faces. ----
    # See `_WEDGE_NEIGHBOUR` (module-level constant) for the table.
    for dir in 1:6
        wp = dir + 1
        for f in 3:6
            neigh_dir, neigh_face = _WEDGE_NEIGHBOUR[dir][f - 2]
            faces[f, wp] = interior_link(neigh_dir + 1, neigh_face, 0)
        end
    end

    return SkeletonMesh{T}(patches, faces)
end

"""
    make_cubed_cube_mesh(::Type{T}, M::Int, R::Real) → HexMesh{T}

Conforming hex mesh of the cube `[-1, 1]³` built from a "cubed-sphere"
block topology applied to a cubic domain: one central cubic patch
`[-R, R]³` plus six radial-wedge patches connecting it to the six outer
cube faces. All outer faces of the global domain are flat (the overall
shape is still a cube), so the *outer* mesh boundary is `[-1, 1]³`
exactly — the geometry has cubed-sphere topology but a cube codomain.

`M` is the mesh-resolution parameter: each of the seven patches is
subdivided into `M` cells along each non-radial axis (so the inner patch
has `M³` cells and each outer patch has `L·M²`).

# Geometry

For the +x patch (the other five are obtained by axis permutation /
reflection), with local indices `i ∈ 0..L`, `j ∈ 0..M`, `k ∈ 0..M`,
parametric coords `a = i/L`, `b = -1 + 2j/M`, `c = -1 + 2k/M`:

    r          = R · (1/R)^a        (radial coordinate)
    (x, y, z)  = (r,  b·r,  c·r)

so the cross-section at radial level `i` is the square `[-r, r]²`,
matching the inner cube's `[-R, R]²` face at `a = 0` and the outer
cube's `[-1, 1]²` face at `a = 1`.

# Element count

`M³ + 6·L·M²` total — `M³` in the inner cube + `L·M²` per outer wedge.

# `L` and radial spacing

To keep each outer-patch cell roughly cubical, set radial-to-tangential
aspect to 1: `α - 1 ≈ 2/M` with `α = (1/R)^(1/L)`. Solving for the
endpoint constraint `r_L = 1` gives

    L = round( log(1/R) / log(1 + 2/M) )

For `M = 5, R = 0.1` this gives `L = 7, α ≈ 1.389`.

# Orientation

By construction, every patch's local axes are oriented so that, at any
shared face, the `(p, q)` face-node coordinates on the two sides match
directly — `orientation[f, e] = 0` everywhere.

Built via the skeleton path (`_cubed_cube_skeleton` →
`_skeleton_to_mesh`), so vertex deduplication and orientations are
combinatorial / integer-keyed — no floating-point comparison is
load-bearing.
"""
make_cubed_cube_mesh(::Type{T}, M::Int, R::Real) where {T} =
    _skeleton_to_mesh(_cubed_cube_skeleton(T, M, R))

"""
    PatchInfo{T}

Per-element parametric description used by `make_geometry` for elements
of an `InflatedCubeMesh` that should be discretised with an analytic
curvilinear map instead of the default trilinear interpolation of the
eight corners.

`kind` selects the patch family:

* `0`     — trilinear (inner cube; the angular / radial fields are unused).
* `1..6`  — inflation patch in directions `(+x, -x, +y, -y, +z, -z)`:
  bridges the inner cube face at `r = L` (with `r = |x|/|y|/|z|`) to the
  inner sphere at radius `R₁`.
* `7..12` — spherical-shell patch in the same directional order:
  bridges the inner sphere at `R₁` to the outer sphere at `R₂`.

For curved patches, `(a_lo, a_hi)` is the radial parameter range
(`s ∈ [0, 1]` for inflation, `ρ ∈ [0, 1]` for shell), `(b_lo, b_hi)` and
`(c_lo, c_hi)` are the two tangent-angular ranges (each `⊂ [-1, 1]`).
"""
struct PatchInfo{T}
    kind :: Int8
    a_lo :: T
    a_hi :: T
    b_lo :: T
    b_hi :: T
    c_lo :: T
    c_hi :: T
end

"""
    InflatedCubeMesh{T}

A 13-patch conforming hex mesh:

* one axis-aligned inner cube `[-L, L]³` (`M³` elements),
* six inflation patches that bridge each cube face to the inner sphere
  at radius `R₁` (`M_i × M × M` elements each),
* six spherical-shell patches that bridge the inner sphere `R₁` to the
  outer sphere `R₂` (`M_s × M × M` elements each).

Total element count: `M³ + 6 · M² · (M_i + M_s)`.

# Fields

* `base :: HexMesh{T}` — connectivity + trilinear vertex info. Host-side
  queries (`element_vertices`, `nv`, `locate_point`, plotting) work
  through this just like a plain `HexMesh`.
* `patch_info :: Vector{PatchInfo{T}}` of length `base.Ne` — per-element
  parametric description used by `make_geometry` to evaluate the
  analytic Jacobian on the curved patches.
* `L, R1, R2 :: T` — inner-cube half-edge, inner-sphere radius, outer-
  sphere radius. Required: `0 < L · √3 < R1 < R2`.

The radial spacing is **exactly** constant on the spherical shells
(uniform `(R2 - R1)/M_s` along every radial ray); on the inflation
patches the radial spacing is constant on average (varies by a factor
between cube-face center and cube-face corner).
"""
struct InflatedCubeMesh{T, MI, MI8}
    base       :: HexMesh{T, MI, MI8}
    patch_info :: Vector{PatchInfo{T}}
    L          :: T
    R1         :: T
    R2         :: T
end

# Forward `HexMesh` accessors through the wrapper so existing host code
# (`mesh.Ne`, `mesh.bdry`, `mesh.vertex_coords`, …) keeps working unchanged.
@inline function Base.getproperty(m::InflatedCubeMesh, name::Symbol)
    if name === :base || name === :patch_info ||
       name === :L || name === :R1 || name === :R2
        return getfield(m, name)
    else
        return getproperty(getfield(m, :base), name)
    end
end
Base.propertynames(m::InflatedCubeMesh) =
    (:base, :patch_info, :L, :R1, :R2,
     :Ne, :conn, :vertex_coords, :vertex_idx,
     :neighbour, :neighbour_face, :orientation, :bdry)

nv(mesh::InflatedCubeMesh) = nv(mesh.base)

# Direction-dependent unit vector `v(b, c)` for the six face directions.
# For each direction `dir ∈ 1..6` (mapping `(+x, -x, +y, -y, +z, -z)`),
# the parameterisation places the physical point `P = f(a, b, c) · v(b, c)`
# (inflation) or `P = (r(a) / Q) · v(b, c)` (shell), where the local
# `(ξ, η, ζ)` frame is right-handed in physical space — picking the
# tangent-axis pair per direction so that `det J > 0`. The corresponding
# tables:
#
#   +x:  v = ( 1,  b,  c)            -x:  v = (-1,  c,  b)
#   +y:  v = ( c,  1,  b)            -y:  v = ( b, -1,  c)
#   +z:  v = ( b,  c,  1)            -z:  v = ( c,  b, -1)
@inline function _patch_direction_vec(dir::Int8, b::T, c::T) where {T}
    if dir == Int8(1)
        return (one(T), b, c)
    elseif dir == Int8(2)
        return (-one(T), c, b)
    elseif dir == Int8(3)
        return (c, one(T), b)
    elseif dir == Int8(4)
        return (b, -one(T), c)
    elseif dir == Int8(5)
        return (b, c, one(T))
    else
        return (c, b, -one(T))
    end
end

# Same as `_patch_direction_vec`, plus the constant partials `∂v/∂b` and
# `∂v/∂c` (each a 3-tuple; entries are `0` or `±1`). Used by the
# analytic-Jacobian path in `make_geometry`.
@inline function _patch_direction_vec_and_derivs(dir::Int8, b::T, c::T) where {T}
    z = zero(T); o = one(T)
    if dir == Int8(1)        # +x
        return (o, b, c,    z, o, z,    z, z, o)
    elseif dir == Int8(2)    # -x
        return (-o, c, b,   z, z, o,    z, o, z)
    elseif dir == Int8(3)    # +y
        return (c, o, b,    z, z, o,    o, z, z)
    elseif dir == Int8(4)    # -y
        return (b, -o, c,   o, z, z,    z, z, o)
    elseif dir == Int8(5)    # +z
        return (b, c, o,    o, z, z,    z, o, z)
    else                     # -z
        return (c, b, -o,   z, o, z,    o, z, z)
    end
end

"""
    make_inflated_cube_mesh(::Type{T}, L, R1, R2, M; M_i, M_s) → InflatedCubeMesh{T}

Build a 13-patch inflated cube mesh of the ball `|x| ≤ R2`:

* an axis-aligned inner cube `[-L, L]³` with `M × M × M` elements,
* six inflation patches that interpolate from each cube face to the
  inner sphere `r = R1`, with `M_i × M × M` elements each,
* six spherical-shell patches that interpolate from the inner sphere to
  the outer sphere `r = R2`, with `M_s × M × M` elements each.

`M_i` defaults to `round((R1 - (1 + √3)/2 · L) / h)` (average radial gap
between the cube and the inner sphere, expressed in cube-edge cells
`h = 2L/M`). `M_s` defaults to `round((R2 - R1) / h)`. Each defaults to
at least 1.

The shell patches use the parameterisation `r(ρ) = (1 - ρ)·R1 + ρ·R2`
along every radial ray, so they have exactly constant radial spacing
`(R2 - R1) / M_s` and exactly uniform angular sampling in
`(η, ζ) ∈ [-1, 1]²`. The inflation patches use
`r(s, η, ζ) = (1 - s)·L + s · R1 / √(1 + η² + ζ²)`, so their radial
spacing is constant on average (varies between cube-face center and
cube-face corner). The geometry matches conformally at every inter-
patch interface (cube → inflation at `r = L`; inflation → shell at
`r = R1`; adjacent inflation/shell patches along the shared cube
edges and great circles on the inner / outer spheres).

The outer boundary `r = R2` is tagged `bdry = 1` on every shell-patch
outer face.

`make_geometry(mesh, elem)` dispatches per element on `patch_info[e].kind`:
trilinear for the inner cube; analytic Jacobian on the curved patches.
"""
function _inflated_cube_skeleton(::Type{T}, L::Real, R1::Real, R2::Real, M::Int;
                                   M_i::Union{Nothing, Int}=nothing,
                                   M_s::Union{Nothing, Int}=nothing) where {T}
    @assert M ≥ 1
    @assert L > 0
    @assert L * sqrt(3) < R1 "inner sphere R1 must enclose the cube corner (L·√3)"
    @assert R1 < R2

    Lv  = T(L)
    R1v = T(R1)
    R2v = T(R2)
    h   = 2L / M

    Mi = M_i === nothing ?
         max(1, round(Int, (R1 - (1 + sqrt(3))/2 * L) / h)) :
         M_i
    Ms = M_s === nothing ?
         max(1, round(Int, (R2 - R1) / h)) :
         M_s
    @assert Mi ≥ 1
    @assert Ms ≥ 1

    z = zero(T)
    o = one(T)

    # Patch 1: inner cube `[-L, L]³`.
    inner = PatchSpec{T}(M, M, M, :cubical,
                          -Lv, Lv, -Lv, Lv, -Lv, Lv,
                          z, z, z)

    # Patches 2..7: inflation in directions (+x, -x, +y, -y, +z, -z),
    # parameter range `(a, b, c) ∈ [0, 1] × [-1, 1]²`.
    inflation_families = (:inflation_pos_x, :inflation_neg_x,
                          :inflation_pos_y, :inflation_neg_y,
                          :inflation_pos_z, :inflation_neg_z)
    # Patches 8..13: shells in the same direction order.
    shell_families     = (:shell_pos_x, :shell_neg_x,
                          :shell_pos_y, :shell_neg_y,
                          :shell_pos_z, :shell_neg_z)

    patches = PatchSpec{T}[inner]
    for fam in inflation_families
        push!(patches, PatchSpec{T}(Mi, M, M, fam,
                                     z, o, -o, o, -o, o,
                                     Lv, R1v, R2v))
    end
    for fam in shell_families
        push!(patches, PatchSpec{T}(Ms, M, M, fam,
                                     z, o, -o, o, -o, o,
                                     Lv, R1v, R2v))
    end
    @assert length(patches) == 13

    faces = Matrix{FaceLink}(undef, 6, length(patches))

    # ---- Cube ↔ inflation interfaces. ----
    # Inner cube face `f` connects to the inflation patch whose
    # direction matches that face, at the inflation's face 1 (inner-
    # radial, `a = 0`). With the right-handed `_patch_direction_vec`,
    # the conventions for `-x, +y, -z` swap the (b, c) → (η_phys, ζ_phys)
    # axis mapping relative to the cube, giving a D₄ transpose
    # (`o = 5`). The other three directions match directly (`o = 0`).
    # Both sides of an `o = 5` link carry `o = 5` (transpose is its
    # own inverse).
    CUBE_FACE_TO_DIR         = (2, 1, 4, 3, 6, 5)
    CUBE_FACE_ORIENTATION    = (Int8(5), Int8(0), Int8(0),
                                Int8(5), Int8(5), Int8(0))
    for f in 1:6
        d  = CUBE_FACE_TO_DIR[f]
        ip = d + 1                   # inflation patch id
        oo = CUBE_FACE_ORIENTATION[f]
        faces[f, 1]  = interior_link(ip, 1, oo)
        faces[1, ip] = interior_link(1,  f, oo)
    end

    # ---- Inflation tangential / radial-outer faces. ----
    # Tangential faces 3..6 connect to adjacent inflation patches via
    # `_INFLATION_NEIGHBOUR` (the cube-edge topology *for the right-handed
    # `_patch_direction_vec` convention*; not the same as the cubed-cube
    # `_WEDGE_NEIGHBOUR`). All twelve cube-edge orientations are still 0.
    # Face 2 (outer-radial, `a = 1`) connects to the same-direction shell
    # patch's face 1 with orientation 0.
    for d in 1:6
        ip = d + 1                   # inflation patch
        sp = d + 7                   # shell patch (same direction)
        faces[2, ip] = interior_link(sp, 1, 0)
        for f in 3:6
            neigh_dir, neigh_face = _INFLATION_NEIGHBOUR[d][f - 2]
            faces[f, ip] = interior_link(neigh_dir + 1, neigh_face, 0)
        end
    end

    # ---- Shell tangential / outer-sphere faces. ----
    # Face 1: → inflation patch face 2 (inner-radial side).
    # Face 2: outer sphere domain boundary, tag 1.
    # Faces 3..6: adjacent shell patches via `_INFLATION_NEIGHBOUR` (same
    # direction conventions, same connectivity).
    for d in 1:6
        sp = d + 7
        ip = d + 1
        faces[1, sp] = interior_link(ip, 2, 0)
        faces[2, sp] = boundary_link(1)
        for f in 3:6
            neigh_dir, neigh_face = _INFLATION_NEIGHBOUR[d][f - 2]
            faces[f, sp] = interior_link(neigh_dir + 7, neigh_face, 0)
        end
    end

    return SkeletonMesh{T}(patches, faces)
end

# Build the per-element `PatchInfo` table that
# `make_geometry(::InflatedCubeMesh)` reads to dispatch on patch kind
# and recover the parameter-space extent of each element. Order matches
# `_skeleton_to_mesh`'s element enumeration (column-major over `(a, b, c)`
# inside each patch, patches walked in order).
function _build_inflated_cube_patch_info(skel::SkeletonMesh{T}, Ne::Int) where {T}
    patch_info = Vector{PatchInfo{T}}(undef, Ne)
    e = 0
    zT = zero(T)
    for ps in skel.patches
        kind = _family_to_kind(ps.family)
        for c in 1:ps.Mc, b in 1:ps.Mb, a in 1:ps.Ma
            e += 1
            if kind == Int8(0)
                # Inner cube: trilinear path; parameter-extent slots unused.
                patch_info[e] = PatchInfo{T}(Int8(0), zT, zT, zT, zT, zT, zT)
            else
                a_lo = ps.a_lo + (ps.a_hi - ps.a_lo) * T(a - 1) / T(ps.Ma)
                a_hi = ps.a_lo + (ps.a_hi - ps.a_lo) * T(a)     / T(ps.Ma)
                b_lo = ps.b_lo + (ps.b_hi - ps.b_lo) * T(b - 1) / T(ps.Mb)
                b_hi = ps.b_lo + (ps.b_hi - ps.b_lo) * T(b)     / T(ps.Mb)
                c_lo = ps.c_lo + (ps.c_hi - ps.c_lo) * T(c - 1) / T(ps.Mc)
                c_hi = ps.c_lo + (ps.c_hi - ps.c_lo) * T(c)     / T(ps.Mc)
                patch_info[e] = PatchInfo{T}(kind, a_lo, a_hi, b_lo, b_hi, c_lo, c_hi)
            end
        end
    end
    @assert e == Ne
    return patch_info
end

function make_inflated_cube_mesh(::Type{T}, L::Real, R1::Real, R2::Real, M::Int;
                                  M_i::Union{Nothing, Int}=nothing,
                                  M_s::Union{Nothing, Int}=nothing) where {T}
    skel       = _inflated_cube_skeleton(T, L, R1, R2, M; M_i, M_s)
    base       = _skeleton_to_mesh(skel)
    patch_info = _build_inflated_cube_patch_info(skel, base.Ne)
    return InflatedCubeMesh(base, patch_info, T(L), T(R1), T(R2))
end

# ----------------------------------------------------------------------
# Per-element geometric map
#
# Each hex element is the image of the reference cube `[0, 1]³` under the
# trilinear map defined by its eight Gmsh-ordered corner vertices `v₁..v₈`.
# Writing `m_a = 1 - a` for `a ∈ {ξ, η, ζ}`, the eight Gmsh-ordered shape
# functions are
#
#     N₁ = m_ξ·m_η·m_ζ      (corner (−x, −y, −z))
#     N₂ =  ξ ·m_η·m_ζ      (corner (+x, −y, −z))
#     N₃ =  ξ · η ·m_ζ      (corner (+x, +y, −z))
#     N₄ = m_ξ· η ·m_ζ      (corner (−x, +y, −z))
#     N₅ = m_ξ·m_η· ζ       (corner (−x, −y, +z))
#     N₆ =  ξ ·m_η· ζ       (corner (+x, −y, +z))
#     N₇ =  ξ · η · ζ       (corner (+x, +y, +z))
#     N₈ = m_ξ· η · ζ       (corner (−x, +y, +z))
#
# and the map is `x(ξ, η, ζ) = Σ Nᵥ(ξ, η, ζ) · vᵥ`. The Jacobian matrix
# `J[i, a] = ∂xᵢ / ∂ξₐ` (with `ξ₁ = ξ`, `ξ₂ = η`, `ξ₃ = ζ`) drops the
# right-side fall-through but otherwise follows the same shape derivatives.
# For axis-aligned hexes `J` is diagonal-constant; for curved/distorted
# hexes (cubed-cube outer patches, future cubed-sphere blocks) `J` varies
# with position and must be inverted per node when applying operators.

# Trilinear shape functions at one reference point. Returns the 8-tuple
# of values in Gmsh corner order, fully stack-allocated.
@inline function trilinear_shape(ξ::T, η::T, ζ::T) where {T}
    mξ, mη, mζ = one(T) - ξ, one(T) - η, one(T) - ζ
    return (mξ * mη * mζ,    ξ * mη * mζ,
             ξ *  η * mζ,   mξ *  η * mζ,
            mξ * mη *  ζ,    ξ * mη *  ζ,
             ξ *  η *  ζ,   mξ *  η *  ζ)
end

# Partial derivatives of the eight shape functions at one reference
# point. Returns three 8-tuples for `∂/∂ξ`, `∂/∂η`, `∂/∂ζ`.
@inline function trilinear_dshape(ξ::T, η::T, ζ::T) where {T}
    mξ, mη, mζ = one(T) - ξ, one(T) - η, one(T) - ζ
    dξ = (-mη * mζ,  mη * mζ,
           η * mζ,  -η * mζ,
          -mη *  ζ,  mη *  ζ,
           η *  ζ,  -η *  ζ)
    dη = (-mξ * mζ, -ξ * mζ,
           ξ * mζ,  mξ * mζ,
          -mξ *  ζ, -ξ *  ζ,
           ξ *  ζ,  mξ *  ζ)
    dζ = (-mξ * mη, -ξ * mη,
          -ξ *  η, -mξ *  η,
           mξ * mη,  ξ * mη,
           ξ *  η,  mξ *  η)
    return dξ, dη, dζ
end

"""
    trilinear_map(verts, ξ, η, ζ) → SVector{3, T}

Image of the reference-cube point `(ξ, η, ζ) ∈ [0, 1]³` under the trilinear
map defined by the eight Gmsh-ordered corner vertices `verts`.
"""
@inline function trilinear_map(verts::NTuple{8, SVector{3, T}},
                                ξ, η, ζ) where {T}
    Nv = trilinear_shape(T(ξ), T(η), T(ζ))
    p = zero(SVector{3, T})
    @inbounds for v in 1:8
        p += Nv[v] * verts[v]
    end
    return p
end

"""
    trilinear_jacobian(verts, ξ, η, ζ) → SMatrix{3, 3, T, 9}

Jacobian of the trilinear element map at reference-point `(ξ, η, ζ)`:
column `a` is `∂x / ∂ξₐ` (with `ξ₁ = ξ`, `ξ₂ = η`, `ξ₃ = ζ`).
"""
@inline function trilinear_jacobian(verts::NTuple{8, SVector{3, T}},
                                     ξ, η, ζ) where {T}
    dξ, dη, dζ = trilinear_dshape(T(ξ), T(η), T(ζ))
    col1 = zero(SVector{3, T})
    col2 = zero(SVector{3, T})
    col3 = zero(SVector{3, T})
    @inbounds for v in 1:8
        col1 += dξ[v] * verts[v]
        col2 += dη[v] * verts[v]
        col3 += dζ[v] * verts[v]
    end
    return SMatrix{3, 3, T}(col1[1], col1[2], col1[3],
                            col2[1], col2[2], col2[3],
                            col3[1], col3[2], col3[3])
end

"""
    _patch_point_and_jac(pi, ξ, η_ref, ζ_ref, L, R1, R2) → (P, J)

Analytic element map for one node of an `InflatedCubeMesh` curved patch.
Given a per-element `PatchInfo` and a reference-cube coordinate
`(ξ, η_ref, ζ_ref) ∈ [0, 1]³`, returns the physical point `P` and the
`SMatrix{3, 3}` Jacobian `J[i, a] = ∂P_i / ∂ξₐ_ref`. The reference-to-
parameter affine map and the parameter-to-physical map are composed
analytically; no finite differencing is involved.

Used by `make_geometry(::InflatedCubeMesh)`. For `pi.kind == 0` (inner
cube), use the trilinear path instead — this routine assumes a curved
patch.
"""
@inline function _patch_point_and_jac(pi::PatchInfo{T},
                                       ξ::T, η_ref::T, ζ_ref::T,
                                       L::T, R1::T, R2::T) where {T}
    # Reference-cube [0, 1]³ → parameter-space (a, b, c).
    da = pi.a_hi - pi.a_lo
    db = pi.b_hi - pi.b_lo
    dc = pi.c_hi - pi.c_lo
    a  = pi.a_lo + da * ξ
    b  = pi.b_lo + db * η_ref
    c  = pi.c_lo + dc * ζ_ref

    Q  = sqrt(one(T) + (b*b + c*c))
    Q3 = Q * Q * Q
    is_shell = pi.kind ≥ Int8(7)
    dir      = ((pi.kind - Int8(1)) % Int8(6)) + Int8(1)

    # Scalar `f(a, b, c)` such that `P = f · v(b, c)`.
    if is_shell
        rval  = (one(T) - a) * R1 + a * R2
        f     = rval / Q
        df_da = (R2 - R1) / Q
        df_db = -rval * b / Q3
        df_dc = -rval * c / Q3
    else
        f     = (one(T) - a) * L + a * R1 / Q
        df_da = -L + R1 / Q
        df_db = -a * R1 * b / Q3
        df_dc = -a * R1 * c / Q3
    end

    vx, vy, vz, dvxb, dvyb, dvzb, dvxc, dvyc, dvzc =
        _patch_direction_vec_and_derivs(dir, b, c)

    Px = f * vx; Py = f * vy; Pz = f * vz

    dPa_x = df_da * vx;            dPa_y = df_da * vy;            dPa_z = df_da * vz
    dPb_x = df_db * vx + f * dvxb; dPb_y = df_db * vy + f * dvyb; dPb_z = df_db * vz + f * dvzb
    dPc_x = df_dc * vx + f * dvxc; dPc_y = df_dc * vy + f * dvyc; dPc_z = df_dc * vz + f * dvzc

    # Reference-cube Jacobian: column `a` is `∂P/∂ξ_ref_a`, scaled by the
    # affine ref→parameter derivative.
    J = SMatrix{3, 3, T}(
        dPa_x * da, dPa_y * da, dPa_z * da,
        dPb_x * db, dPb_y * db, dPb_z * db,
        dPc_x * dc, dPc_y * dc, dPc_z * dc)
    P = SVector{3, T}(Px, Py, Pz)
    return P, J
end

# Convenience: extract an element's eight corners from the shared
# `vertex_coords` table as an 8-tuple of `SVector{3, T}`.
@inline function element_vertices(mesh::HexMesh{T}, e::Integer) where {T}
    @inbounds ntuple(v -> begin
        vi = mesh.vertex_idx[v, e]
        SVector{3, T}(mesh.vertex_coords[1, vi],
                      mesh.vertex_coords[2, vi],
                      mesh.vertex_coords[3, vi])
    end, Val(8))
end

# Forward element-corner queries through the `InflatedCubeMesh` wrapper.
# The trilinear corners are only an approximation of the curved-patch
# geometry, but match exactly on the inner cube and are good enough for
# bounding-box reject (`locate_point`) and for plotting.
@inline function element_vertices(mesh::InflatedCubeMesh{T}, e::Integer) where {T}
    return element_vertices(mesh.base, e)
end

"""
    MeshGeometry{T, N}

Per-node geometric data for a `HexMesh`, evaluated at the GLL collocation
points of a 1D reference element with `N` nodes. Holds *only* what the
kernel reads — the underlying `HexMesh` topology (vertices and their
indices into the connectivity) is **not** carried here; keep your own
reference to it for host-side queries (`element_vertices`, plotting,
`locate_point`, etc.).

# Fields

* `Ne :: Int` — element count. Mirrors `mesh.Ne` of the originating
  `HexMesh` and is used as the kernel `ndrange`.
* `conn :: MeshConnectivity{MI, MI8}` — the four connectivity matrices
  copied across from `mesh.conn`. Kernel-resident; backed by `Array`
  on the host and by the appropriate device array on GPU backends.
* `coords :: Array{T, 5}` of shape `(3, N, N, N, Ne)` — physical (x, y, z)
  coordinate of every collocation point.
* `jac    :: Array{T, 6}` of shape `(3, 3, N, N, N, Ne)` — Jacobian
  matrix `J[a, b] = ∂xₐ / ∂ξ_b` of the element map at each node.
* `invjac :: Array{T, 6}` — inverse of `J` at each node; supplies
  `∂ξ / ∂x` to operators that need to pull physical gradients back to the
  reference cube.
* `detjac :: Array{T, 4}` of shape `(N, N, N, Ne)` — absolute value of
  `det J`, the per-node volume factor used by the integration weights.
* `Hphys :: Array{T, 4}` of shape `(N, N, N, Ne)` — the per-node
  physical mass `H_ref[i]·H_ref[j]·H_ref[k]·|det J|`. Precomputed
  here so that GPU-portable reductions (`discrete_inner_product`,
  `discrete_l2_norm`, `spectral_radius_estimate`) can run as a single
  `mapreduce` over device arrays without re-deriving the mass per
  node from the 1D quadrature weights and the Jacobian on each call.
* `face_trace :: Array{T, 5}` of shape `(4, N, N, 6, Ne)` — per-
  element face-trace staging buffer used by the two-pass `rhs3d!`
  implementation. Filled by pass 1 with `(u, ∂x u, ∂y u, ∂z u)` at
  each face quadrature node (physical gradient — the local element's
  `J⁻ᵀ` has already been applied), then read by pass 2 across the
  neighbour relation `mesh.conn.neighbour` to compute the face SAT
  contributions. This is workspace, not geometry: its values are
  overwritten on every `rhs3d!` call. Hard-coded for V=1 fields
  (the wave equation); supporting multi-component PDEs in the future
  will replace the leading `4` with `4·V` or grow a sixth axis.
* `handedness :: Vector{Int8}` of length `Ne` — `±1`, the sign of
  `det J` on element `e`. A non-degenerate hex has uniform-sign Jacobian
  throughout, so a single scalar per element captures the handedness;
  `_add_face_sat!` reads this to pick the outward face normal direction
  without a per-face-node test.

The curvilinear-Laplacian kernel composes these with the 1D quadrature
weights from `ops.H` on the fly: per-node physical mass is
`Hphys = H_ref[i] H_ref[j] H_ref[k] · |det J|` and the weak-form stiffness
kernel is `Wmetric = Hphys · (J⁻¹ J⁻ᵀ)`.
"""
# `MeshGeometry{T, N}` is parametrised on the concrete storage types of
# every kernel-read field so it can be device-resident on any backend.
# All fields are bitstype-adaptable, so KA's launch-time recursive
# `adapt` migrates everything to device types in one shot — there is no
# special handling of any host-only field, because there is none.
struct MeshGeometry{T, N, MC, A5, A6, A4, V1}
    Ne         :: Int
    conn       :: MC
    coords     :: A5
    jac        :: A6
    invjac     :: A6
    detjac     :: A4
    Hphys      :: A4
    face_trace :: A5
    handedness :: V1

    function MeshGeometry{T, N}(Ne::Int, conn::MC,
                                coords::A5, jac::A6, invjac::A6,
                                detjac::A4, Hphys::A4,
                                face_trace::A5,
                                handedness::V1) where {T, N, MC, A5, A6, A4, V1}
        new{T, N, MC, A5, A6, A4, V1}(Ne, conn,
                                       coords, jac, invjac,
                                       detjac, Hphys, face_trace, handedness)
    end
end

"""
    make_geometry(mesh, elem) → MeshGeometry{T, N}

Evaluate the trilinear element map of every hex in `mesh` at the GLL
collocation points of the reference element `elem` (using `elem.xs ∈
[0, 1]` as reference coordinates), and bundle the resulting physical
coordinates, Jacobians, inverse Jacobians, and `|det J|` into a
`MeshGeometry`. The returned geometry copies `mesh.conn` by reference;
the caller retains ownership of the `HexMesh` (with its vertex data)
for host-side queries.
"""
function make_geometry(mesh::HexMesh{T}, elem) where {T}
    N  = elem.N
    ξs = elem.xs
    Ne = mesh.Ne

    # 1D GLL quadrature weights, used to build the per-node physical
    # mass `Hphys`. We pull them from a fresh `SBPOps` rather than
    # depending on the user to pass `ops` in — operator construction
    # is `O(N²)` and runs once at mesh setup, so the cost is invisible.
    ops_ref = make_operators(elem)
    H_1d    = SVector{N, T}(ntuple(i -> ops_ref.H[i, i], Val(N)))

    coords     = Array{T, 5}(undef, 3, N, N, N, Ne)
    jac        = Array{T, 6}(undef, 3, 3, N, N, N, Ne)
    invjac     = Array{T, 6}(undef, 3, 3, N, N, N, Ne)
    detjac     = Array{T, 4}(undef, N, N, N, Ne)
    Hphys      = Array{T, 4}(undef, N, N, N, Ne)
    face_trace = Array{T, 5}(undef, 4, N, N, 6, Ne)   # workspace, see struct doc
    handedness = Vector{Int8}(undef, Ne)
    @inbounds for e in 1:Ne
        verts = element_vertices(mesh, e)
        # Sign of det(J) at element corner (ξ = η = ζ = 0). For a
        # non-degenerate hex the sign is uniform throughout, so any
        # single sample point determines the element's handedness.
        J_corner   = trilinear_jacobian(verts, zero(T), zero(T), zero(T))
        handedness[e] = det(J_corner) ≥ 0 ? Int8(1) : Int8(-1)
        for k in 1:N, j in 1:N, i in 1:N
            ξ, η, ζ = ξs[i], ξs[j], ξs[k]
            p  = trilinear_map(verts, ξ, η, ζ)
            J  = trilinear_jacobian(verts, ξ, η, ζ)
            Ji = inv(J)
            dJ = abs(det(J))
            for a in 1:3
                coords[a, i, j, k, e] = p[a]
                for b in 1:3
                    jac[a, b, i, j, k, e]    = J[a, b]
                    invjac[a, b, i, j, k, e] = Ji[a, b]
                end
            end
            detjac[i, j, k, e] = dJ
            Hphys[i, j, k, e]  = H_1d[i] * H_1d[j] * H_1d[k] * dJ
        end
    end
    return MeshGeometry{T, N}(Ne, mesh.conn,
                              coords, jac, invjac, detjac, Hphys, face_trace, handedness)
end

"""
    make_geometry(mesh::InflatedCubeMesh, elem) → MeshGeometry{T, N}

Evaluate the per-element geometric map of every patch in `mesh` at the
GLL collocation points of the reference element `elem`. Dispatches per
element on `mesh.patch_info[e].kind`:

* `kind == 0` (inner cube): trilinear interpolation of the 8 corners —
  identical to `make_geometry(::HexMesh, elem)`.
* `kind == 1..6` (inflation patch): analytic Jacobian from
  `r(s, η, ζ) = (1 - s)·L + s · R₁ / √(1 + η² + ζ²)` evaluated through
  `_patch_point_and_jac`.
* `kind == 7..12` (shell patch): analytic Jacobian from
  `r(ρ) = (1 - ρ)·R₁ + ρ·R₂`, also through `_patch_point_and_jac`.

The returned `MeshGeometry` is interchangeable with one built from a
plain `HexMesh`; downstream kernels are agnostic to the underlying
mesh's curvature.
"""
function make_geometry(mesh::InflatedCubeMesh{T}, elem) where {T}
    N  = elem.N
    ξs = elem.xs
    Ne = mesh.Ne

    ops_ref = make_operators(elem)
    H_1d    = SVector{N, T}(ntuple(i -> ops_ref.H[i, i], Val(N)))

    coords     = Array{T, 5}(undef, 3, N, N, N, Ne)
    jac        = Array{T, 6}(undef, 3, 3, N, N, N, Ne)
    invjac     = Array{T, 6}(undef, 3, 3, N, N, N, Ne)
    detjac     = Array{T, 4}(undef, N, N, N, Ne)
    Hphys      = Array{T, 4}(undef, N, N, N, Ne)
    face_trace = Array{T, 5}(undef, 4, N, N, 6, Ne)
    handedness = Vector{Int8}(undef, Ne)

    Lv  = mesh.L
    R1v = mesh.R1
    R2v = mesh.R2

    @inbounds for e in 1:Ne
        pi = mesh.patch_info[e]
        if pi.kind == Int8(0)
            # Trilinear path — inner cube
            verts = element_vertices(mesh.base, e)
            J_c   = trilinear_jacobian(verts, zero(T), zero(T), zero(T))
            handedness[e] = det(J_c) ≥ 0 ? Int8(1) : Int8(-1)
            for k in 1:N, j in 1:N, i in 1:N
                ξ, η, ζ = ξs[i], ξs[j], ξs[k]
                p  = trilinear_map(verts, ξ, η, ζ)
                J  = trilinear_jacobian(verts, ξ, η, ζ)
                Ji = inv(J)
                dJ = abs(det(J))
                for a in 1:3
                    coords[a, i, j, k, e] = p[a]
                    for b in 1:3
                        jac[a, b, i, j, k, e]    = J[a, b]
                        invjac[a, b, i, j, k, e] = Ji[a, b]
                    end
                end
                detjac[i, j, k, e] = dJ
                Hphys[i, j, k, e]  = H_1d[i] * H_1d[j] * H_1d[k] * dJ
            end
        else
            # Analytic curvilinear path — inflation / shell patch
            _, J_c = _patch_point_and_jac(pi, T(0.5), T(0.5), T(0.5),
                                          Lv, R1v, R2v)
            handedness[e] = det(J_c) ≥ 0 ? Int8(1) : Int8(-1)
            for k in 1:N, j in 1:N, i in 1:N
                ξ, η, ζ = ξs[i], ξs[j], ξs[k]
                p, J = _patch_point_and_jac(pi, ξ, η, ζ, Lv, R1v, R2v)
                Ji = inv(J)
                dJ = abs(det(J))
                for a in 1:3
                    coords[a, i, j, k, e] = p[a]
                    for b in 1:3
                        jac[a, b, i, j, k, e]    = J[a, b]
                        invjac[a, b, i, j, k, e] = Ji[a, b]
                    end
                end
                detjac[i, j, k, e] = dJ
                Hphys[i, j, k, e]  = H_1d[i] * H_1d[j] * H_1d[k] * dJ
            end
        end
    end
    return MeshGeometry{T, N}(Ne, mesh.conn,
                              coords, jac, invjac, detjac, Hphys, face_trace, handedness)
end

################################################################################
# Device migration

"""
    to_device(mesh::HexMesh, backend) → HexMesh
    to_device(geom::MeshGeometry, backend) → MeshGeometry

Move every kernel-read array of `mesh` / `geom` onto `backend` (a
`KernelAbstractions.Backend` instance — `CPU()`, `CUDABackend()`,
`MetalBackend()`, `ROCBackend()`). For `mesh`, this migrates the four
connectivity matrices; the host-only `vertex_coords` / `vertex_idx`
are left as plain CPU `Matrix`. For `geom`, it migrates `coords`,
`jac`, `invjac`, `detjac`, `handedness`, and the embedded mesh.

The CPU → CPU case is a no-op-shaped copy: every allocation goes
through `KernelAbstractions.allocate(backend, …)` which on the CPU
backend just calls `Array{T}(undef, …)`. Round-tripping through
`to_device(g, CPU())` is therefore a valid smoke test that exercises
the migration path without requiring a GPU.
"""
function to_device(mesh::HexMesh{T}, backend) where {T}
    nb  = KernelAbstractions.allocate(backend, Int32, size(mesh.neighbour))
    nbf = KernelAbstractions.allocate(backend, Int8, size(mesh.neighbour_face))
    ori = KernelAbstractions.allocate(backend, Int8, size(mesh.orientation))
    bdr = KernelAbstractions.allocate(backend, Int8, size(mesh.bdry))
    copyto!(nb,  mesh.neighbour)
    copyto!(nbf, mesh.neighbour_face)
    copyto!(ori, mesh.orientation)
    copyto!(bdr, mesh.bdry)
    new_conn = MeshConnectivity(nb, nbf, ori, bdr)
    return HexMesh{T}(mesh.Ne, new_conn, mesh.vertex_coords, mesh.vertex_idx)
end

function to_device(mesh::InflatedCubeMesh{T}, backend) where {T}
    base_dev = to_device(mesh.base, backend)
    return InflatedCubeMesh(base_dev, mesh.patch_info, mesh.L, mesh.R1, mesh.R2)
end

function to_device(geom::MeshGeometry{T, N}, backend) where {T, N}
    conn_dev = to_device(geom.conn, backend)
    coords  = KernelAbstractions.allocate(backend, T,    size(geom.coords))
    jac     = KernelAbstractions.allocate(backend, T,    size(geom.jac))
    invjac  = KernelAbstractions.allocate(backend, T,    size(geom.invjac))
    detjac  = KernelAbstractions.allocate(backend, T,    size(geom.detjac))
    Hphys   = KernelAbstractions.allocate(backend, T,    size(geom.Hphys))
    ft      = KernelAbstractions.allocate(backend, T,    size(geom.face_trace))
    hand    = KernelAbstractions.allocate(backend, Int8, size(geom.handedness))
    copyto!(coords, geom.coords)
    copyto!(jac,    geom.jac)
    copyto!(invjac, geom.invjac)
    copyto!(detjac, geom.detjac)
    copyto!(Hphys,  geom.Hphys)
    # face_trace is workspace; no host data to copy. Its values are
    # overwritten on every `rhs3d!` call. We still allocate it on the
    # device so the kernels can write into it directly.
    copyto!(hand,   geom.handedness)
    return MeshGeometry{T, N}(geom.Ne, conn_dev,
                              coords, jac, invjac, detjac, Hphys, ft, hand)
end

# `MeshConnectivity` device migration — used both directly (when a
# caller migrates a HexMesh) and indirectly through `to_device(geom)`.
function to_device(conn::MeshConnectivity, backend)
    nb  = KernelAbstractions.allocate(backend, Int32, size(conn.neighbour))
    nbf = KernelAbstractions.allocate(backend, Int8, size(conn.neighbour_face))
    ori = KernelAbstractions.allocate(backend, Int8, size(conn.orientation))
    bdr = KernelAbstractions.allocate(backend, Int8, size(conn.bdry))
    copyto!(nb,  conn.neighbour)
    copyto!(nbf, conn.neighbour_face)
    copyto!(ori, conn.orientation)
    copyto!(bdr, conn.bdry)
    return MeshConnectivity(nb, nbf, ori, bdr)
end

# `Adapt.adapt_structure` rules. When KernelAbstractions launches a
# kernel on a GPU backend, it walks each argument with `Adapt.adapt(to,
# arg)` and replaces host arrays with their device representations
# (`CuArray` → `CuDeviceArray`, `MtlArray` → `MtlDeviceArray`, etc.).
# `MeshConnectivity` and `MeshGeometry` participate in that walk by
# recursively adapting every field. No special rule is needed for
# `HexMesh`: it is host-only and never crosses a kernel boundary.
Adapt.adapt_structure(to, c::MeshConnectivity) = MeshConnectivity(
    Adapt.adapt(to, c.neighbour),
    Adapt.adapt(to, c.neighbour_face),
    Adapt.adapt(to, c.orientation),
    Adapt.adapt(to, c.bdry))

Adapt.adapt_structure(to, geom::MeshGeometry{T, N}) where {T, N} =
    MeshGeometry{T, N}(
        geom.Ne,
        Adapt.adapt(to, geom.conn),
        Adapt.adapt(to, geom.coords),
        Adapt.adapt(to, geom.jac),
        Adapt.adapt(to, geom.invjac),
        Adapt.adapt(to, geom.detjac),
        Adapt.adapt(to, geom.Hphys),
        Adapt.adapt(to, geom.face_trace),
        Adapt.adapt(to, geom.handedness))

"""
    element_coords(mesh, elem) → Array{T, 5}

Thin wrapper that returns just the physical collocation coordinates from
`make_geometry(mesh, elem)`. Prefer `make_geometry` when you also need the
per-node Jacobian.
"""
element_coords(mesh::HexMesh, elem) = make_geometry(mesh, elem).coords
element_coords(mesh::InflatedCubeMesh, elem) = make_geometry(mesh, elem).coords

# ----------------------------------------------------------------------
# Interpolation from per-element data to arbitrary physical points
#
# Used for visualisation and diagnostics. Not optimised: the point-to-
# element search is `O(Ne)` per query and the inversion of the trilinear
# map runs a small Newton iteration. Plenty fast for plotting grids of a
# few thousand points; never call this from a hot inner loop.

"""
    invert_element_map(verts, p; tol, maxiter) → (ξ::SVector{3, T}, ok::Bool)

Solve `trilinear_map(verts, ξ, η, ζ) = p` for the reference coordinate
`(ξ, η, ζ)`. Returns the converged `ξ` and a flag indicating whether the
residual fell below `tol` within `maxiter` Newton steps.
"""
function invert_element_map(verts::NTuple{8, SVector{3, T}}, p::SVector{3, T};
                             tol = T(1e-12), maxiter::Int = 20) where {T}
    ξ = SVector{3, T}(one(T)/2, one(T)/2, one(T)/2)
    res_last = T(Inf)
    for _ in 1:maxiter
        x = trilinear_map(verts, ξ[1], ξ[2], ξ[3])
        r = x - p
        res_last = sqrt(r[1]^2 + r[2]^2 + r[3]^2)
        res_last < tol && return ξ, true
        J = trilinear_jacobian(verts, ξ[1], ξ[2], ξ[3])
        ξ = ξ - (J \ r)
    end
    return ξ, res_last < sqrt(tol)
end

"""
    locate_point(mesh, p; tol) → (e::Int, ξ::SVector{3, T})

Find which element of `mesh` contains the physical point `p` and return
its index plus the reference coordinate. Returns `(0, _)` if `p` is
outside every element. Brute-force search with a bounding-box reject;
intended only for visualisation.
"""
locate_point(mesh::InflatedCubeMesh, p; kwargs...) = locate_point(mesh.base, p; kwargs...)

function locate_point(mesh::HexMesh{T}, p::SVector{3, T};
                       tol = T(1e-8)) where {T}
    @inbounds for e in 1:mesh.Ne
        verts = element_vertices(mesh, e)
        # Cheap reject: skip elements whose bounding box does not contain p.
        xmin = min(verts[1][1], verts[2][1], verts[3][1], verts[4][1],
                   verts[5][1], verts[6][1], verts[7][1], verts[8][1])
        xmax = max(verts[1][1], verts[2][1], verts[3][1], verts[4][1],
                   verts[5][1], verts[6][1], verts[7][1], verts[8][1])
        (p[1] < xmin - tol || p[1] > xmax + tol) && continue
        ymin = min(verts[1][2], verts[2][2], verts[3][2], verts[4][2],
                   verts[5][2], verts[6][2], verts[7][2], verts[8][2])
        ymax = max(verts[1][2], verts[2][2], verts[3][2], verts[4][2],
                   verts[5][2], verts[6][2], verts[7][2], verts[8][2])
        (p[2] < ymin - tol || p[2] > ymax + tol) && continue
        zmin = min(verts[1][3], verts[2][3], verts[3][3], verts[4][3],
                   verts[5][3], verts[6][3], verts[7][3], verts[8][3])
        zmax = max(verts[1][3], verts[2][3], verts[3][3], verts[4][3],
                   verts[5][3], verts[6][3], verts[7][3], verts[8][3])
        (p[3] < zmin - tol || p[3] > zmax + tol) && continue
        ξ, ok = invert_element_map(verts, p)
        ok && all(-tol ≤ ξ[i] ≤ 1 + tol for i in 1:3) && return e, ξ
    end
    return 0, SVector{3, T}(zero(T), zero(T), zero(T))
end

# 1D Lagrange basis values at `ξ` for nodes `xs`. Length(xs) = N → returns
# `NTuple{N, T}`. Generic and unoptimised — `O(N²)` per call.
function lagrange_basis(xs, ξ::T) where {T}
    N = length(xs)
    out = ntuple(N) do i
        v = one(T)
        @inbounds for j in 1:N
            i == j && continue
            v *= (ξ - xs[j]) / (xs[i] - xs[j])
        end
        v
    end
    return out
end

# Tensor-product Lagrange interpolation of an `(N, N, N)` block at the
# reference point `(ξ, η, ζ)`. `xs` are the 1D GLL nodes on `[0, 1]`.
function tensor_interp(ue::AbstractArray{T, 3},
                        ξ::T, η::T, ζ::T, xs) where {T}
    N = length(xs)
    ℓξ = lagrange_basis(xs, ξ)
    ℓη = lagrange_basis(xs, η)
    ℓζ = lagrange_basis(xs, ζ)
    s = zero(T)
    @inbounds for k in 1:N, j in 1:N, i in 1:N
        s += ue[i, j, k] * ℓξ[i] * ℓη[j] * ℓζ[k]
    end
    return s
end

"""
    interpolate_field(mesh, elem, u, p; default) → T

Evaluate the per-element field `u` (shape `(N, N, N, Ne)`) at the
physical point `p` by locating the element of `mesh::HexMesh` that
contains `p`, inverting the trilinear element map, and applying
tensor-product Lagrange interpolation on the GLL nodes `elem.xs`.
Returns `default` if `p` lies outside the mesh. Brute-force, intended
for visualisation.
"""
function interpolate_field(mesh::HexMesh{T}, elem,
                            u::AbstractArray{T, 4},
                            p::SVector{3, T};
                            default = T(NaN)) where {T}
    e, ξ = locate_point(mesh, p)
    e == 0 && return default
    return tensor_interp(view(u, :, :, :, e), ξ[1], ξ[2], ξ[3], elem.xs)
end

interpolate_field(mesh::InflatedCubeMesh, elem, u, p; kwargs...) =
    interpolate_field(mesh.base, elem, u, p; kwargs...)

# Vectorised convenience: take any iterable of points and return an
# array of values with the same shape.
function interpolate_field(mesh::HexMesh{T}, elem,
                            u::AbstractArray{T, 4},
                            points::AbstractArray{<:SVector{3, T}};
                            default = T(NaN)) where {T}
    out = similar(points, T)
    for I in eachindex(points)
        out[I] = interpolate_field(mesh, elem, u, points[I]; default)
    end
    return out
end
