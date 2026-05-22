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

# Hex face-corner index conventions (Gmsh canonical):
#
#   face 1 (−x / −i):  vertices 1, 4, 5, 8
#   face 2 (+x / +i):  vertices 2, 3, 6, 7
#   face 3 (−y / −j):  vertices 1, 2, 5, 6
#   face 4 (+y / +j):  vertices 3, 4, 7, 8
#   face 5 (−z / −k):  vertices 1, 2, 3, 4
#   face 6 (+z / +k):  vertices 5, 6, 7, 8
const FACE_LOCAL_VERTICES = (
    (1, 4, 5, 8),
    (2, 3, 6, 7),
    (1, 2, 5, 6),
    (3, 4, 7, 8),
    (1, 2, 3, 4),
    (5, 6, 7, 8),
)

"""
    make_inflated_cube_mesh(::Type{T}, N::Int, R::Real) → HexMesh{T}

Conforming hex mesh of the cube `[-1, 1]³` whose topology is the
"inflated cube" / "cubed sphere": one central cubic patch `[-R, R]³`
plus six radial-wedge patches connecting it to the six outer cube faces.
All outer faces of the global domain are flat (the overall shape is
still a cube), so the *outer* mesh boundary is `[-1, 1]³` exactly.

# Geometry

For the +x patch (the other five are obtained by axis permutation /
reflection), with local indices `i ∈ 0..L`, `j ∈ 0..N`, `k ∈ 0..N` and
`s_j = -1 + 2j/N`, `t_k = -1 + 2k/N`:

    r          = R · α^i        (radial coordinate)
    (x, y, z)  = (r,  s_j·r,  t_k·r)

so the cross-section at radial level `i` is the square `[-r, r]²`, which
matches the inner cube's `[-R, R]²` face at `i = 0` and the outer cube's
`[-1, 1]²` face at `i = L`.

# Element count

`N^3` cells in the inner patch + `6·L·N²` cells in the six outer patches.

# `L` and radial spacing (step 2 of the construction)

We want each outer-patch cell to be roughly cubical: angular width
`2r/N` should match the radial width `r_{i+1} − r_i`. With geometric
spacing `r_i = R·α^i` the cell aspect is constant in `α`:

    r_{i+1} - r_i = r_i · (α - 1),  angular size = 2 r_i / N

so isotropic ⇒ `α - 1 ≈ 2/N`. The radial endpoint constraint `r_L = 1`
fixes `α = (1/R)^(1/L)`, so we pick

    L = round( log(1/R) / log(1 + 2/N) )

and use the resulting `α`. For `N = 5`, `R = 0.1` this gives `L = 7`,
`α ≈ 1.389`.

# Orientation

By construction, every patch's local axes are oriented so that, at any
shared face, the (p, q) face-node coordinates on the two sides match
directly — `orientation[f, e] = 0` everywhere.
"""
function make_inflated_cube_mesh(::Type{T}, N::Int, R::Real) where {T}
    @assert N ≥ 1
    @assert 0 < R < 1

    Rv = T(R)

    # --- Step 2: pick L and the radial node positions -------------------
    L = max(1, round(Int, log(1/R) / log(1 + 2/N)))
    α = (1/Rv)^(1/T(L))
    radial = [Rv * α^(j-1) for j in 1:L+1]
    radial[end] = one(T)        # snap the outer node to exactly 1

    # --- Step 3a: build the vertex coordinate table ---------------------
    # All 7 patches generate vertices on the same shared grid; vertices
    # that two patches both produce (faces, edges and corners of the
    # inner cube; edges and corners between adjacent outer patches) get
    # de-duplicated by exact-position `Dict` lookup. Inputs are computed
    # with identical floating-point ops on either side so they hash equal.
    vertex_dict = Dict{NTuple{3, T}, Int}()
    add_vertex!(x, y, z) = get!(vertex_dict, (x, y, z)) do
        length(vertex_dict) + 1
    end

    # Inner cube grid (i, j, k) ∈ 0..N × 0..N × 0..N → vertex index.
    cube_v = Array{Int, 3}(undef, N+1, N+1, N+1)
    for c in 0:N, b in 0:N, a in 0:N
        s = T(-1 + 2*a/N); t = T(-1 + 2*b/N); u = T(-1 + 2*c/N)
        cube_v[a+1, b+1, c+1] = add_vertex!(s*Rv, t*Rv, u*Rv)
    end

    # Outer-patch vertex grids: 6 grids of shape (L+1, N+1, N+1).
    function build_patch_vertices(dir::Int)
        v = Array{Int, 3}(undef, L+1, N+1, N+1)
        for c in 0:N, b in 0:N, a in 0:L
            r  = radial[a+1]
            sb = T(-1 + 2*b/N)
            tc = T(-1 + 2*c/N)
            x, y, z = if dir == 1       # +x
                ( r,    sb*r, tc*r)
            elseif dir == 2             # −x
                (-r,    sb*r, tc*r)
            elseif dir == 3             # +y
                (sb*r,  r,    tc*r)
            elseif dir == 4             # −y
                (sb*r, -r,    tc*r)
            elseif dir == 5             # +z
                (sb*r, tc*r,  r   )
            else                        # −z (dir == 6)
                (sb*r, tc*r, -r   )
            end
            v[a+1, b+1, c+1] = add_vertex!(x, y, z)
        end
        return v
    end
    patch_v = ntuple(d -> build_patch_vertices(d), 6)

    Nv = length(vertex_dict)
    vertex_coords = Matrix{T}(undef, 3, Nv)
    for (key, idx) in vertex_dict
        vertex_coords[1, idx] = key[1]
        vertex_coords[2, idx] = key[2]
        vertex_coords[3, idx] = key[3]
    end

    # --- Step 3b: build the per-element 8-corner connectivity -----------
    Ne = N^3 + 6 * L * N^2
    vertex_idx = Matrix{Int}(undef, 8, Ne)

    function fill_corners!(e, vg, i, j, k)
        vertex_idx[1, e] = vg[i,   j,   k  ]
        vertex_idx[2, e] = vg[i+1, j,   k  ]
        vertex_idx[3, e] = vg[i+1, j+1, k  ]
        vertex_idx[4, e] = vg[i,   j+1, k  ]
        vertex_idx[5, e] = vg[i,   j,   k+1]
        vertex_idx[6, e] = vg[i+1, j,   k+1]
        vertex_idx[7, e] = vg[i+1, j+1, k+1]
        vertex_idx[8, e] = vg[i,   j+1, k+1]
    end

    e = 0
    for k in 1:N, j in 1:N, i in 1:N
        e += 1; fill_corners!(e, cube_v, i, j, k)
    end
    for vg in patch_v
        for k in 1:N, j in 1:N, i in 1:L
            e += 1; fill_corners!(e, vg, i, j, k)
        end
    end
    @assert e == Ne

    # --- Step 3c: derive neighbour table from face-vertex signatures ----
    # Two faces match iff their *sets* of four global vertex IDs are equal;
    # we sort each face's vertex tuple to get a canonical hashable signature.
    neighbour = zeros(Int, 6, Ne)
    face_sig  = Dict{NTuple{4, Int}, Tuple{Int, Int}}()
    @inbounds for ee in 1:Ne, f in 1:6
        flv = FACE_LOCAL_VERTICES[f]
        face = (vertex_idx[flv[1], ee], vertex_idx[flv[2], ee],
                vertex_idx[flv[3], ee], vertex_idx[flv[4], ee])
        sig  = NTuple{4, Int}(sort!(collect(face)))
        prev = get(face_sig, sig, (0, 0))
        if prev[1] == 0
            face_sig[sig] = (ee, f)
        else
            neighbour[f,        ee     ] = prev[1]
            neighbour[prev[2],  prev[1]] = ee
        end
    end

    # --- Step 3d: tag outer-cube faces ----------------------------------
    bdry = zeros(Int8, 6, Ne)
    on(v, t) = abs(v - t) < T(1e-10)
    @inbounds for ee in 1:Ne, f in 1:6
        neighbour[f, ee] == 0 || continue
        flv = FACE_LOCAL_VERTICES[f]
        v1 = vertex_idx[flv[1], ee]; v2 = vertex_idx[flv[2], ee]
        v3 = vertex_idx[flv[3], ee]; v4 = vertex_idx[flv[4], ee]
        bdry[f, ee] =
            all(on(vertex_coords[1, v], -1) for v in (v1, v2, v3, v4)) ? 1 :
            all(on(vertex_coords[1, v], +1) for v in (v1, v2, v3, v4)) ? 2 :
            all(on(vertex_coords[2, v], -1) for v in (v1, v2, v3, v4)) ? 3 :
            all(on(vertex_coords[2, v], +1) for v in (v1, v2, v3, v4)) ? 4 :
            all(on(vertex_coords[3, v], -1) for v in (v1, v2, v3, v4)) ? 5 :
            all(on(vertex_coords[3, v], +1) for v in (v1, v2, v3, v4)) ? 6 :
            Int8(0)
    end

    # Orientation: zero by construction (see docstring).
    orientation = zeros(Int8, 6, Ne)

    return HexMesh{T}(Ne, neighbour, orientation, bdry, vertex_coords, vertex_idx)
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
# hexes (inflated-cube outer patches, future cubed-sphere blocks) `J` varies
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

"""
    MeshGeometry{T, N}

Per-node geometric data for a `HexMesh`, evaluated at the GLL collocation
points of a 1D reference element with `N` nodes. Built once at mesh setup
time and consumed by the kernels for IC evaluation, plotting, and
metric-aware operator application.

# Fields

* `mesh   :: HexMesh{T}` — the underlying topology + vertex coordinates.
* `coords :: Array{T, 5}` of shape `(3, N, N, N, Ne)` — physical (x, y, z)
  coordinate of every collocation point.
* `jac    :: Array{T, 6}` of shape `(3, 3, N, N, N, Ne)` — Jacobian
  matrix `J[a, b] = ∂xₐ / ∂ξ_b` of the element map at each node.
* `invjac :: Array{T, 6}` — inverse of `J` at each node; supplies
  `∂ξ / ∂x` to operators that need to pull physical gradients back to the
  reference cube.
* `detjac :: Array{T, 4}` of shape `(N, N, N, Ne)` — absolute value of
  `det J`, the per-node volume factor used by the integration weights.

The curvilinear-Laplacian kernel composes these with the 1D quadrature
weights from `ops.H` on the fly: per-node physical mass is
`Hphys = H_ref[i] H_ref[j] H_ref[k] · |det J|` and the weak-form stiffness
kernel is `Wmetric = Hphys · (J⁻¹ J⁻ᵀ)`.
"""
struct MeshGeometry{T, N}
    mesh   :: HexMesh{T}
    coords :: Array{T, 5}
    jac    :: Array{T, 6}
    invjac :: Array{T, 6}
    detjac :: Array{T, 4}
end

"""
    make_geometry(mesh, elem) → MeshGeometry{T, N}

Evaluate the trilinear element map of every hex in `mesh` at the GLL
collocation points of the reference element `elem` (using `elem.xs ∈
[0, 1]` as reference coordinates), and bundle the resulting physical
coordinates, Jacobians, inverse Jacobians, and `|det J|` into a
`MeshGeometry`.
"""
function make_geometry(mesh::HexMesh{T}, elem) where {T}
    N  = elem.N
    ξs = elem.xs
    Ne = mesh.Ne
    coords = Array{T, 5}(undef, 3, N, N, N, Ne)
    jac    = Array{T, 6}(undef, 3, 3, N, N, N, Ne)
    invjac = Array{T, 6}(undef, 3, 3, N, N, N, Ne)
    detjac = Array{T, 4}(undef, N, N, N, Ne)
    @inbounds for e in 1:Ne
        verts = element_vertices(mesh, e)
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
        end
    end
    return MeshGeometry{T, N}(mesh, coords, jac, invjac, detjac)
end

"""
    element_coords(mesh, elem) → Array{T, 5}

Thin wrapper that returns just the physical collocation coordinates from
`make_geometry(mesh, elem)`. Prefer `make_geometry` when you also need the
per-node Jacobian.
"""
element_coords(mesh::HexMesh, elem) = make_geometry(mesh, elem).coords

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
    interpolate_field(geom, elem, u, p; default) → T

Evaluate the per-element field `u` (shape `(N, N, N, Ne)`) at the
physical point `p` by locating the element containing `p`, inverting the
trilinear element map, and applying tensor-product Lagrange interpolation
on the GLL nodes `elem.xs`. Returns `default` if `p` lies outside the
mesh. Brute-force, intended for visualisation.
"""
function interpolate_field(geom::MeshGeometry{T, N}, elem,
                            u::AbstractArray{T, 4},
                            p::SVector{3, T};
                            default = T(NaN)) where {T, N}
    e, ξ = locate_point(geom.mesh, p)
    e == 0 && return default
    return tensor_interp(view(u, :, :, :, e), ξ[1], ξ[2], ξ[3], elem.xs)
end

# Vectorised convenience: take any iterable of points and return an
# array of values with the same shape.
function interpolate_field(geom::MeshGeometry{T}, elem,
                            u::AbstractArray{T, 4},
                            points::AbstractArray{<:SVector{3, T}};
                            default = T(NaN)) where {T}
    out = similar(points, T)
    for I in eachindex(points)
        out[I] = interpolate_field(geom, elem, u, points[I]; default)
    end
    return out
end
