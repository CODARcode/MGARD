#include "MeshLevel.hpp"

#include <cassert>

#include <stdexcept>
#include <utility>

#include "measure.hpp"

#define IS_ELEMENT(type) (moab::MBTRI <= type && type <= moab::MBPOLYHEDRON)

//TODO: `{edge,tri,tet}_measure` work on blocks of elements.

double edge_measure(
    moab::Interface const * const impl,
    const moab::EntityHandle edge
) {
    moab::ErrorCode ecode;
    std::vector<moab::EntityHandle> connectivity;
    ecode = impl->get_connectivity(&edge, 1, connectivity);
    MB_CHK_ERR_RET_VAL(ecode, -1);
    const std::size_t N = connectivity.size();
    assert(N == 2);
    std::vector<double> coordinates(3 * N);
    ecode = impl->get_coords(connectivity.data(), N, coordinates.data());
    MB_CHK_ERR_RET_VAL(ecode, -1);
    return helpers::edge_measure(coordinates.data());
}

double tri_measure(
    moab::Interface * const impl,
    const moab::EntityHandle tri
) {
    moab::ErrorCode ecode;
    std::vector<moab::EntityHandle> connectivity;
    ecode = impl->get_connectivity(&tri, 1, connectivity);
    MB_CHK_ERR_RET_VAL(ecode, -1);
    const std::size_t N = connectivity.size();
    assert(N == 3);
    std::vector<double> coordinates(3 * N);
    ecode = impl->get_coords(connectivity.data(), N, coordinates.data());
    MB_CHK_ERR_RET_VAL(ecode, -1);
    return helpers::tri_measure(coordinates.data());
}

double tet_measure(
    moab::Interface * const impl,
    const moab::EntityHandle tet
) {
    moab::ErrorCode ecode;
    std::vector<moab::EntityHandle> connectivity;
    ecode = impl->get_connectivity(&tet, 1, connectivity);
    MB_CHK_ERR_RET_VAL(ecode, -1);
    const std::size_t N = connectivity.size();
    assert(N == 4);
    std::vector<double> coordinates(3 * N);
    ecode = impl->get_coords(connectivity.data(), N, coordinates.data());
    MB_CHK_ERR_RET_VAL(ecode, -1);
    return helpers::tet_measure(coordinates.data());
}

double element_measure(
    moab::Interface * const impl,
    const moab::EntityHandle element
) {
    const moab::EntityType type = impl->type_from_handle(element);
    if (type == moab::MBTRI) {
        return tri_measure(impl, element);
    } else if (type == moab::MBTET) {
        return tet_measure(impl, element);
    } else {
        throw std::invalid_argument(
            "can only find measures of triangles and tetrahedra"
        );
    }
}

namespace mgard {

MeshLevel::MeshLevel(
    moab::Interface * const impl,
    const moab::Range nodes,
    const moab::Range edges,
    const moab::Range elements
) :
    impl(impl),
    nodes(nodes),
    edges(edges),
    elements(elements) {
}

MeshLevel::MeshLevel(
    moab::Interface * const impl,
    const moab::EntityHandle mesh_set
) :
    impl(impl) {
        moab::ErrorCode ecode;

        ecode = impl->get_entities_by_type(mesh_set, moab::MBVERTEX, nodes);
        MB_CHK_ERR_RET(ecode);
        ecode = impl->get_entities_by_type(mesh_set, moab::MBEDGE, edges);
        MB_CHK_ERR_RET(ecode);
        ecode = impl->get_entities_by_type(mesh_set, moab::MBTRI, elements);
        MB_CHK_ERR_RET(ecode);
}

std::size_t MeshLevel::node_index(const moab::EntityHandle node) const {
    assert(impl->type_from_handle(node) == moab::MBVERTEX);
    assert(nodes.psize() == 1);
    return node - nodes.front();
}

std::size_t MeshLevel::edge_index(const moab::EntityHandle edge) const {
    assert(impl->type_from_handle(edge) == moab::MBEDGE);
    assert(edges.psize() == 1);
    return edge - edges.front();
}

std::size_t MeshLevel::element_index(const moab::EntityHandle element) const {
    const moab::EntityType type = impl->type_from_handle(element);
    assert(IS_ELEMENT(type));
    assert(elements.psize() == 1);
    return element - elements.front();
}

double MeshLevel::measure(const moab::EntityHandle handle) const {
    const::moab::EntityType type = impl->type_from_handle(handle);
    //Can't imagine ever needing the measure of a node.
    if (type == moab::MBEDGE) {
        //Can add an `edge_measures` member if we end up needing these a lot.
        return edge_measure(impl, handle);
    } else if (IS_ELEMENT(type)) {
        if (!element_measures.empty()) {
            return element_measures.at(element_index(handle));
        } else {
            return element_measure(impl, handle);
        }
    } else {
        throw std::invalid_argument(
            "can only find measures of edges and elements"
        );
    }
}

void MeshLevel::precompute_element_measures() {
    if (!element_measures.empty()) {
        return;
    }
    assert(element_measures.size() == 0);
    element_measures.reserve(elements.size());
    for (auto iter = elements.begin(); iter != elements.end(); ++iter) {
        element_measures.push_back(element_measure(impl, *iter));
    }
}

void MeshLevel::mass_matrix_matvec(double const * const v, double * const b) {
    moab::ErrorCode ecode;
    const std::size_t N = nodes.size();
    for (std::size_t i = 0; i < N; ++i) {
        b[i] = 0;
    }
    const int d = impl->dimension_from_handle(elements.front());
    assert(d == impl->dimension_from_handle(elements.back()));
    for (moab::EntityHandle element : elements) {
        const double measure_factor = measure(element) / ((d + 1) * (d + 2));
        moab::EntityHandle const *connectivity;
        int n;
        ecode = impl->get_connectivity(element, connectivity, n);
        MB_CHK_ERR_RET(ecode);
        //Pairs `(i, u)` where `i` is the global index of a node and `u` is
        //`v[i]`, the value of the function there.
        std::vector<std::pair<std::size_t, double>> nodal_pairs(n);
        double nodal_values_sum = 0;
        for (int i = 0; i < n; ++i) {
            std::pair<std::size_t, double> &pair = nodal_pairs.at(i);
            nodal_values_sum += (
                pair.second = v[pair.first = node_index(connectivity[i])]
            );
        }
        for (std::pair<std::size_t, double> pair : nodal_pairs) {
            b[pair.first] += measure_factor * (nodal_values_sum + pair.second);
        }
    }
}

}
