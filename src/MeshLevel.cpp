#include "MeshLevel.hpp"

#include <cassert>

#include <stdexcept>
#include <utility>

#include "measure.hpp"

#define IS_ELEMENT(type) (moab::MBTRI <= type && type <= moab::MBPOLYHEDRON)


class EntityMeasureFunction {
    public:
        EntityMeasureFunction(const moab::EntityType type) {
            switch (type) {
                case (moab::MBEDGE):
                    f = helpers::edge_measure;
                    expected_connectivity_size = 2;
                    break;
                case (moab::MBTRI):
                    f = helpers::tri_measure;
                    expected_connectivity_size = 3;
                    break;
                case (moab::MBTET):
                    f = helpers::tet_measure;
                    expected_connectivity_size = 4;
                    break;
                default:
                    throw std::invalid_argument(
                        "can only find measures of edges, triangles, and "
                        "tetrahedra"
                    );
            }
        }

        moab::ErrorCode operator()(
            moab::Interface const * const impl,
            const moab::EntityHandle handle,
            double *measure
        ) const {
            moab::ErrorCode ecode;

            std::vector<moab::EntityHandle> connectivity;
            ecode = impl->get_connectivity(&handle, 1, connectivity);
            MB_CHK_ERR(ecode);
            const std::size_t N = connectivity.size();
            assert(N == expected_connectivity_size);
            std::vector<double> coordinates(3 * N);
            ecode = impl->get_coords(
                connectivity.data(), N, coordinates.data()
            );
            MB_CHK_ERR(ecode);
            *measure = f(coordinates.data());
            return moab::MB_SUCCESS;
        }

        //Can imagine rewriting this to fetch all the connectivity data and
        //vertex coordinates at once.
        moab::ErrorCode operator()(
            moab::Interface const * const impl,
            const moab::Range handles,
            double *measures
        ) const {
            for (auto iter = handles.begin(); iter != handles.end(); ++iter) {
                moab::ErrorCode ecode = this->operator()(
                    impl, *iter, measures++
                );
                MB_CHK_ERR_CONT(ecode);
            }
            return moab::MB_SUCCESS;
        }

    private:
        //Function to compute the measure from the coordinates of the vertices.
        double (*f)(double const * const);
        std::size_t expected_connectivity_size;
};

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
    if (IS_ELEMENT(type) && !element_measures.empty()) {
        //Can add an `edge_measures` member if we end up needing those a lot. In
        //that case, I think it'd be nicer to have something like
        //    std::map<moab::EntityType, std::vector<double>> measures;
        //or (probably more properly)
        //    std::vector<double> measures[moab::MBMAXTYPE];
        //In the same vein, could replace `nodes`/`edges`/`elements` members
        //with
        //    moab::Range entities[moab::MBMAXTYPE];
        //and combined `{node,edge,element}_index` into a single function.
        return element_measures.at(element_index(handle));
    } else {
        double measure;
        moab::ErrorCode ecode = EntityMeasureFunction(type)(
            impl, handle, &measure
        );
        MB_CHK_ERR_CONT(ecode);
        return measure;
    }
}

void MeshLevel::precompute_element_measures() {
    if (!element_measures.empty() || elements.empty()) {
        return;
    }
    assert(element_measures.empty());
    element_measures.resize(elements.size());
    //We checked above that `elements` is not empty.
    const moab::EntityType type = impl->type_from_handle(elements.front());
    if (type == impl->type_from_handle(elements.back())) {
        moab::ErrorCode ecode = EntityMeasureFunction(type)(
            impl, elements, element_measures.data()
        );
        MB_CHK_ERR_CONT(ecode);
        return;
    } {
        std::size_t i = 0;
        for (auto element : elements) {
            moab::ErrorCode ecode = EntityMeasureFunction(
                impl->type_from_handle(element)
            )(impl, element, &element_measures.at(i++));
            MB_CHK_ERR_CONT(ecode);
        }
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
