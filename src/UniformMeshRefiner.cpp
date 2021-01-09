#include "UniformMeshRefiner.hpp"

#include <cassert>
#include <cstddef>
#include <cstring>

#include <algorithm>
#include <array>
#include <set>
#include <unordered_map>

#include "blas.hpp"

#include "utilities.hpp"

//! Find the node of a triangle not included in an edge.
//!
//!\param triangle_connectivity Nodes of the triangle. Must have length 3.
//!\param edge_connectivity Nodes of the edge. Must have length 2.
//!
//!\return Node opposite the edge.
static moab::EntityHandle find_node_opposite_edge(
    const mgard::PseudoArray<const moab::EntityHandle> triangle_connectivity,
    const mgard::PseudoArray<const moab::EntityHandle> edge_connectivity) {
  moab::EntityHandle const *const _begin = edge_connectivity.begin();
  moab::EntityHandle const *const _end = edge_connectivity.end();
  for (const moab::EntityHandle x : triangle_connectivity) {
    // Look for this node in the edge.
    moab::EntityHandle const *const p = std::find(_begin, _end, x);
    // If we don't find it, the node is opposite the edge.
    if (p == _end) {
      return x;
    }
  }
  throw std::invalid_argument("triangle nodes somehow contained in edge nodes");
}

using OrderedEdge = std::array<moab::EntityHandle, 2>;

// Sort a pair of endpoints.
static OrderedEdge make_ordered_edge(const moab::EntityHandle a,
                                     const moab::EntityHandle b) {
  OrderedEdge edge = {a, b};
  std::sort(edge.begin(), edge.end());
  return edge;
}

namespace mgard {

MeshLevel UniformMeshRefiner::do_operator_parentheses(const MeshLevel &mesh) {
  moab::ErrorCode ecode;

  moab::Range NODES;
  moab::Range EDGES;
  ecode = bisect_edges(mesh, NODES, EDGES);
  if (ecode != moab::MB_SUCCESS) {
    throw std::runtime_error("failed to bisect edges");
  }

  moab::Range ELEMENTS;
  switch (mesh.topological_dimension) {
  case 2:
    ecode = quadrisect_triangles(mesh, NODES, EDGES, ELEMENTS);
    if (ecode != moab::MB_SUCCESS) {
      throw std::runtime_error("failed to quadrisect triangles");
    }
    break;
  case 3:
    ecode = octasect_tetrahedra(mesh, NODES, EDGES, ELEMENTS);
    if (ecode != moab::MB_SUCCESS) {
      throw std::runtime_error("failed to octasect tetrahedra");
    }
    break;
  default:
    throw std::logic_error("unsupported topological dimension");
  }
  return MeshLevel(mesh.impl, NODES, EDGES, ELEMENTS);
}

moab::ErrorCode UniformMeshRefiner::bisect_edges(const MeshLevel &mesh,
                                                 moab::Range &NODES,
                                                 moab::Range &EDGES) {
  moab::ErrorCode ecode;
  assert(NODES.empty());
  assert(EDGES.empty());
  const moab::Range &nodes = mesh.entities[moab::MBVERTEX];
  const moab::Range &edges = mesh.entities[moab::MBEDGE];
  const std::size_t nnodes = mesh.ndof();
  const std::size_t nedges = edges.size();
  // This could be made cleaner by using something like
  //`std::vector<double [3]>`, but I don't want to spend a bunch of time
  // looking into initialization and contiguity now.
  // TODO: I believe it will be automatically initalized to zero. Check.
  std::vector<double> midpoint_coordinates(3 * nedges, 0);
  // Similarly.
  std::vector<double> endpoint_coordinates(3 * nnodes);
  ecode = mesh.impl.get_coords(nodes, endpoint_coordinates.data());

  moab::Range NEW_NODES;
  // We'll reset the coordinates below. Creating the nodes now lets us iterate
  // over the edges just once, finding the coordinates of the midpoints and
  // creating the new edges in the same loop.
  ecode =
      mesh.impl.create_vertices(midpoint_coordinates.data(), nedges, NEW_NODES);
  // From the MOAB documentation:
  //> Entities allocated in sequence will typically have contiguous handles
  // Checking that.
  assert(NEW_NODES.psize() == 1);
  assert(NEW_NODES.front() == nodes.back() + 1);
  assert(NEW_NODES.back() == NEW_NODES.front() + nedges - 1);

  // We'll check that the new edges have contiguous handles as we go and
  // construct the new edge range at the end. (We don't need `EDGES.front()` to
  // follow `edges.back()`, but it should and anyway it'll make the contiguity
  // check a little simpler.)
  moab::EntityHandle most_recent_edge = edges.back();
  for (std::size_t i = 0; i < nedges; ++i) {
    const moab::EntityHandle edge = edges[i];
    const moab::EntityHandle midpoint = NEW_NODES[i];
    moab::EntityHandle EDGE_CONNECTIVITY[2];
    EDGE_CONNECTIVITY[0] = midpoint;
    double *const p = midpoint_coordinates.data() + 3 * i;
    double *const q = endpoint_coordinates.data();
    for (const moab::EntityHandle endpoint : mesh.connectivity(edge)) {
      EDGE_CONNECTIVITY[1] = endpoint;
      moab::EntityHandle EDGE;
      ecode =
          mesh.impl.create_element(moab::MBEDGE, EDGE_CONNECTIVITY, 2, EDGE);
      ++most_recent_edge;
      assert(EDGE == most_recent_edge);
      blas::axpy(3, 1.0, q + 3 * mesh.index(endpoint), p);
    }
    blas::scal(3, 0.5, p);
  }
  ecode = mesh.impl.set_coords(NEW_NODES, midpoint_coordinates.data());
  MB_CHK_ERR(ecode);

  NODES.insert(nodes.front(), NEW_NODES.back());
  assert(NODES.psize() == 1);
  assert(NODES.size() == nnodes + nedges);

  EDGES.insert(edges.back() + 1, most_recent_edge);
  assert(EDGES.psize() == 1);
  assert(EDGES.size() == 2 * nedges);
  return moab::MB_SUCCESS;
}

moab::ErrorCode UniformMeshRefiner::quadrisect_triangles(
    const MeshLevel &mesh, const moab::Range &NODES, moab::Range &EDGES,
    moab::Range &ELEMENTS) {
  moab::ErrorCode ecode;
  assert(ELEMENTS.empty());
  assert(mesh.element_type == moab::MBTRI);
  const moab::Range &elements = mesh.entities[mesh.element_type];
  const std::size_t nnodes = mesh.ndof();
  [[maybe_unused]] const std::size_t nelements = elements.size();

  // We'll use this to check that elements are contiguous. (We don't need
  //`ELEMENTS.front()` to follow `elements.back()`, but it should and anyway
  // it'll make this check a little simpler.)
  moab::EntityHandle most_recent_element = elements.back();
  // Similarly for the edges.
  moab::EntityHandle most_recent_edge = EDGES.back();
  for (const moab::EntityHandle element : elements) {
    // See the note in `octasect_tetrahedra` about whether the refinement is
    // deterministic.
    const PseudoArray<const moab::EntityHandle> element_connectivity =
        mesh.connectivity(element);

    std::vector<moab::EntityHandle> edges;
    edges.reserve(3);
    ecode = mesh.impl.get_adjacencies(&element, 1, 1, false, edges);
    MB_CHK_ERR(ecode);
    assert(edges.size() == 3);

    // For each node in the original element, the midpoint of the opposite
    // edge. We use this to ensure that the new elements have the same
    // orientation as the original element.
    std::unordered_map<moab::EntityHandle, moab::EntityHandle> opposites;

    for (const moab::EntityHandle edge : edges) {
      opposites.insert(
          {find_node_opposite_edge(element_connectivity,
                                   mesh.connectivity(edge)),
           // The new nodes were created with the same ordering as the old
           // edges. The `i`th node is the midpoint of the `i`th edge.
           NODES[nnodes + mesh.index(edge)]});
    }

    // Add the elements containing an old node.
    moab::EntityHandle ELEMENT_CONNECTIVITY[3];
    for (std::size_t i = 0; i < 3; ++i) {
      const moab::EntityHandle node = element_connectivity[i];
      ELEMENT_CONNECTIVITY[0] = node;
      for (std::size_t j = 1; j < 3; ++j) {
        const std::size_t k = (i + (3 - j)) % 3;
        ELEMENT_CONNECTIVITY[j] = opposites.at(element_connectivity[k]);
      }
      moab::EntityHandle ELEMENT;
      ecode = mesh.impl.create_element(moab::MBTRI, ELEMENT_CONNECTIVITY, 3,
                                       ELEMENT);
      MB_CHK_ERR(ecode);
      ++most_recent_element;
      assert(ELEMENT == most_recent_element);
    }

    // Add the element containing only new nodes.
    for (std::size_t i = 0; i < 3; ++i) {
      ELEMENT_CONNECTIVITY[i] = opposites.at(element_connectivity[i]);
    }
    {
      moab::EntityHandle ELEMENT;
      ecode = mesh.impl.create_element(moab::MBTRI, ELEMENT_CONNECTIVITY, 3,
                                       ELEMENT);
      MB_CHK_ERR(ecode);
      ++most_recent_element;
      assert(ELEMENT == most_recent_element);
    }
    // Add the corresponding edges.
    moab::EntityHandle EDGE_CONNECTIVITY[2];
    for (std::size_t i = 0; i < 3; ++i) {
      EDGE_CONNECTIVITY[0] = ELEMENT_CONNECTIVITY[i];
      EDGE_CONNECTIVITY[1] = ELEMENT_CONNECTIVITY[(i + 1) % 3];
      moab::EntityHandle EDGE;
      ecode =
          mesh.impl.create_element(moab::MBEDGE, EDGE_CONNECTIVITY, 2, EDGE);
      MB_CHK_ERR(ecode);
      ++most_recent_edge;
      assert(EDGE == most_recent_edge);
    }
  }
  ELEMENTS.insert(elements.back() + 1, most_recent_element);
  assert(ELEMENTS.psize() == 1);
  assert(ELEMENTS.size() == 4 * nelements);

  EDGES.insert(EDGES.back() + 1, most_recent_edge);
  assert(EDGES.psize() == 1);
  return moab::MB_SUCCESS;
}

//! Function object used to find the local index of a midpoint of an edge.
class TetrahedralEdgeMidpointIndex {
public:
  //! Constructor.
  //!
  //!\param nodes 'Old' nodes of the 'old' tetrahedron.
  TetrahedralEdgeMidpointIndex(moab::EntityHandle const *const nodes)
      : begin(nodes), end(nodes + 4) {}

  //! Find the index of a midpoint of an edge.
  //!
  //!\param edge_connectivity Connectivity (endpoints) of the edge in question.
  std::size_t operator()(
      const PseudoArray<const moab::EntityHandle> edge_connectivity) const {
    assert(edge_connectivity.size == 2);
    const std::size_t I = index(edge_connectivity[0]);
    const std::size_t J = index(edge_connectivity[1]);
    const std::size_t i = std::min(I, J);
    const std::size_t j = std::max(I, J);
    // See Figure 16 (b) (as of this writing) of the MGARD unstructured paper.
    // If the 'old' nodes are indexed starting with 1, the index of a midpoint
    // is determined by the indices of the corresponding endpoints as follows.
    //     {1, 2} ↦ 5
    //     {1, 3} ↦ 6
    //     {1, 4} ↦ 7
    //     {2, 3} ↦ 8
    //     {2, 4} ↦ 9
    //     {3, 4} ↦ 10
    // I have no idea how to do this most efficiently, but the following
    // works and doesn't require any branching. `j - i` is added to an offset,
    // which is computed using a quadratic function in `i` (0 ↦ 3, 1 ↦ 6, and
    // 2 ↦ 8). I assume the cost is dominated by the calls to `std::find`.
    return (((5 - i) * i + 6) >> 1) + j;
  }

private:
  //! Beginning of the range of 'old' nodes.
  moab::EntityHandle const *const begin;

  //! End of the range of 'old' nodes.
  moab::EntityHandle const *const end;

  //! Find the index of an endpoint of an edge.
  //!
  //!\param node Endpoint in question.
  std::size_t index(const moab::EntityHandle node) const {
    return std::find(begin, end, node) - begin;
  }
};

moab::ErrorCode UniformMeshRefiner::octasect_tetrahedra(
    const MeshLevel &mesh, const moab::Range &NODES, moab::Range &EDGES,
    moab::Range &ELEMENTS) {
  moab::ErrorCode ecode;
  assert(ELEMENTS.empty());
  assert(mesh.element_type == moab::MBTET);
  const moab::Range &elements = mesh.entities[mesh.element_type];
  const std::size_t nnodes = mesh.ndof();
  [[maybe_unused]] const std::size_t nelements = elements.size();

  // We'll use this to check that elements are contiguous. (We don't need
  //`ELEMENTS.front()` to follow `elements.back()`, but it should and anyway
  // it'll make this check a little simpler.)
  moab::EntityHandle most_recent_element = elements.back();
  // Similarly for the edges.
  moab::EntityHandle most_recent_edge = EDGES.back();

  moab::EntityHandle ELEMENT;
  moab::EntityHandle ELEMENT_CONNECTIVITY[4];
  moab::EntityHandle EDGE;
  moab::EntityHandle EDGE_CONNECTIVITY[2];

#define CREATE_TETRAHEDRON(a, b, c, d)                                         \
  ELEMENT_CONNECTIVITY[0] = a;                                                 \
  ELEMENT_CONNECTIVITY[1] = b;                                                 \
  ELEMENT_CONNECTIVITY[2] = c;                                                 \
  ELEMENT_CONNECTIVITY[3] = d;                                                 \
  ecode =                                                                      \
      mesh.impl.create_element(moab::MBTET, ELEMENT_CONNECTIVITY, 4, ELEMENT); \
  MB_CHK_ERR(ecode);                                                           \
  ++most_recent_element;                                                       \
  assert(ELEMENT == most_recent_element);

#define CREATE_EDGE(a, b)                                                      \
  EDGE_CONNECTIVITY[0] = a;                                                    \
  EDGE_CONNECTIVITY[1] = b;                                                    \
  {                                                                            \
    const OrderedEdge edge = make_ordered_edge(a, b);                          \
    if (!inner_edges.count(edge)) {                                            \
      ecode =                                                                  \
          mesh.impl.create_element(moab::MBEDGE, EDGE_CONNECTIVITY, 2, EDGE);  \
      MB_CHK_ERR(ecode);                                                       \
      ++most_recent_edge;                                                      \
      assert(EDGE == most_recent_edge);                                        \
      inner_edges.insert(edge);                                                \
    }                                                                          \
  }

  // We need to keep track of the inner edges we create because, unlike in the
  // 2D case, neighboring elements can contribute the same inner edge. Perhaps
  // it would be better to index these inner edges by, for example, the vertices
  // of the face containing them to speed up the search.
  std::set<OrderedEdge> inner_edges;
  for (const moab::EntityHandle element : elements) {
    // `MeshLevel::connectivity` calls `moab::Interface::get_connectivity`.
    // From the documentation for that function:
    //
    // > The nodes in 'connectivity' are properly ordered according to the
    // > element's canonical ordering.
    //
    // This refinement procedure will be deterministic only if that ordering is
    // deterministic. I'm not looking into that now.
    const PseudoArray<const moab::EntityHandle> element_connectivity =
        mesh.connectivity(element);
    moab::EntityHandle nodes[10];
    std::memcpy(nodes, element_connectivity.data,
                element_connectivity.size * sizeof(moab::EntityHandle));
    const TetrahedralEdgeMidpointIndex midpoint_index(nodes);

    std::vector<moab::EntityHandle> edges;
    edges.reserve(6);
    ecode = mesh.impl.get_adjacencies(&element, 1, 1, false, edges);
    MB_CHK_SET_ERR(ecode, "failed to get edge adjacencies");
    assert(edges.size() == 6);

    // Possibly it would be more efficient to get all of the edge connectivities
    // at once. I'm not sure, though, and don't want to test now, whether we'd
    // get the nodes repeated, as we'd need.
    for (const moab::EntityHandle edge : edges) {
      // The new nodes were created with the same ordering as the old edges. The
      // `i`th node is the midpoint of the `i`th edge.
      nodes[midpoint_index(mesh.connectivity(edge))] =
          NODES[nnodes + mesh.index(edge)];
    }

    // The following block of code was generated by a script. See the first
    // paragraph (as of this writing) of the MGARD unstructured paper appendix.

    const moab::EntityHandle A = nodes[0];
    const moab::EntityHandle B = nodes[1];
    const moab::EntityHandle C = nodes[2];
    const moab::EntityHandle D = nodes[3];
    const moab::EntityHandle AB = nodes[4];
    const moab::EntityHandle AC = nodes[5];
    const moab::EntityHandle AD = nodes[6];
    const moab::EntityHandle BC = nodes[7];
    const moab::EntityHandle BD = nodes[8];
    const moab::EntityHandle CD = nodes[9];

    CREATE_TETRAHEDRON(A, AB, AC, AD)
    CREATE_TETRAHEDRON(B, AB, BC, BD)
    CREATE_TETRAHEDRON(C, AC, BC, CD)
    CREATE_TETRAHEDRON(D, AD, BD, CD)
    CREATE_TETRAHEDRON(AC, AD, CD, BC)
    CREATE_TETRAHEDRON(AD, CD, BD, BC)
    CREATE_TETRAHEDRON(AC, AB, AD, BC)
    CREATE_TETRAHEDRON(AB, AD, BC, BD)

    CREATE_EDGE(AB, AC)
    CREATE_EDGE(AB, AD)
    CREATE_EDGE(AB, BC)
    CREATE_EDGE(AB, BD)
    CREATE_EDGE(AC, AD)
    CREATE_EDGE(AC, BC)
    CREATE_EDGE(AC, CD)
    CREATE_EDGE(AD, BC)
    CREATE_EDGE(AD, BD)
    CREATE_EDGE(AD, CD)
    CREATE_EDGE(BC, BD)
    CREATE_EDGE(BC, CD)
    CREATE_EDGE(BD, CD)
  }

#undef CREATE_TETRAHEDRON

#undef CREATE_EDGE

  ELEMENTS.insert(elements.back() + 1, most_recent_element);
  assert(ELEMENTS.psize() == 1);
  assert(ELEMENTS.size() == 8 * nelements);

  EDGES.insert(EDGES.back() + 1, most_recent_edge);
  assert(EDGES.psize() == 1);
  return moab::MB_SUCCESS;
}

} // namespace mgard
