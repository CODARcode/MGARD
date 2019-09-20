#include "MeshLevel.hpp"

#include <cassert>

#include <stdexcept>
#include <utility>

#include "measure.hpp"

typedef double (*EntityMeasureFunction)(double const * const);

//For each entity type we get the number of vertices an entity of that type has
//(zero for default) and a pointer to a function that computes the measure of an
//entity of that type given a block of doubles with three coordinates for each
//vertex of the entity.
static std::pair<
    std::size_t, EntityMeasureFunction
> emf_data[moab::MBMAXTYPE] = {
    {0, NULL}, //MBVERTEX
    {2, helpers::edge_measure}, //MBEDGE
    {3, helpers::tri_measure}, //MBTRI
    {0, NULL}, //MBQUAD
    {0, NULL}, //MBPOLYGON
    {4, helpers::tet_measure}, //MBTET
    {0, NULL}, //MBPYRAMID
    {0, NULL}, //MBPRISM
    {0, NULL}, //MBKNIFE
    {0, NULL}, //MBHEX
    {0, NULL}, //MBPOLYHEDRON
    {0, NULL} //MBENTITYSET
  //MBMAXTYPE
};

namespace mgard {

//Public member functions.

MeshLevel::MeshLevel(
    moab::Interface * const impl,
    const moab::Range nodes,
    const moab::Range edges,
    const moab::Range elements
):
    impl(impl),
    element_type(moab::MBMAXTYPE)
{
    entities[moab::MBVERTEX] = nodes;
    entities[moab::MBEDGE] = edges;
    if (!elements.empty()) {
        element_type = impl->type_from_handle(elements.front());
        assert(element_type == impl->type_from_handle(elements.back()));
        entities[element_type] = elements;
        populate_from_element_type();
    }
}

MeshLevel::MeshLevel(
    moab::Interface * const impl,
    const moab::EntityHandle mesh_set
):
    impl(impl)
{
    moab::ErrorCode ecode;
    for (
        moab::EntityType type = moab::MBVERTEX;
        type != moab::MBMAXTYPE;
        ++type
    ) {
        ecode = impl->get_entities_by_type(mesh_set, type, entities[type]);
        MB_CHK_ERR_RET(ecode);
    }
    std::size_t num_element_types = 0;
    for (
        moab::EntityType type = moab::MBTRI;
        type <= moab::MBPOLYHEDRON;
        ++type
    ) {
        if (!entities[type].empty()) {
            ++num_element_types;
            element_type = type;
        }
    }
    if (num_element_types > 1) {
        throw std::invalid_argument(
            "all elements must be of the same type"
        );
    } else if (num_element_types == 1) {
        if (!(element_type == moab::MBTRI || element_type == moab::MBTET)) {
            throw std::invalid_argument(
                "elements must be triangles or tetrahedra"
            );
        }
        populate_from_element_type();
    }
}

std::size_t MeshLevel::ndof() const {
    return do_ndof();
}

//TODO: look into allowing `type` to be passed here (or just for internal use).
std::size_t MeshLevel::index(const moab::EntityHandle handle) const {
    const moab::EntityType type = impl->type_from_handle(handle);
    const moab::Range &range = entities[type];
    assert(range.psize() == 1);
    assert(!range.empty());
    if (!(range.front() <= handle && handle <= range.back())) {
        throw std::out_of_range("attempt to find index of entity not in mesh");
    }
    return handle - range.front();
}

double MeshLevel::measure(const moab::EntityHandle handle) {
    const::moab::EntityType type = impl->type_from_handle(handle);
    precompute_measures(type);
    moab::ErrorCode ecode = precompute_measures(type);
    MB_CHK_ERR_CONT(ecode);
    return measures[type].at(index(handle));
}

double MeshLevel::containing_elements_measure(const moab::EntityHandle node) {
    const moab::EntityType type = impl->type_from_handle(node);
    if (type != moab::MBVERTEX) {
        throw std::domain_error(
            "can only find measure of elements containing a node"
        );
    }
    precompute_element_measures();
    return preconditioner_divisors.at(index(node));
}

moab::ErrorCode MeshLevel::precompute_element_measures() {
    return precompute_measures(element_type);
}

//Protected member functions.

void MeshLevel::check_system_size(const std::size_t N) const {
    if (N != ndof()) {
        throw std::invalid_argument("system size incorrect");
    }
}

//Private member functions.

void MeshLevel::populate_from_element_type() {
    switch (element_type) {
        case moab::MBTRI:
            topological_dimension = 2;
            num_nodes_per_element = 3;
            break;
        case moab::MBTET:
            topological_dimension = 3;
            num_nodes_per_element = 4;
            break;
        default:
            throw std::invalid_argument(
                "elements must be triangles or tetrahedra"
            );
    }
}

std::size_t MeshLevel::do_ndof() const {
    return entities[moab::MBVERTEX].size();
}

moab::ErrorCode MeshLevel::precompute_measures(const moab::EntityType type) {
    if (entities[type].empty() || !measures[type].empty()) {
        return moab::MB_SUCCESS;
    }
    measures[type].resize(entities[type].size());

    const bool computing_element_measures = type == element_type;
    if (computing_element_measures) {
        if (!preconditioner_divisors.empty()) {
            throw std::logic_error(
                "preconditioner divisors unexpectedly nonempty"
            );
        }
        preconditioner_divisors.resize(ndof(), 0);
    }

    std::pair<std::size_t, EntityMeasureFunction> datum = emf_data[type];
    const std::size_t expected_connectivity_size = datum.first;
    const EntityMeasureFunction f = datum.second;
    if (f == NULL) {
        throw std::invalid_argument(
            "can only find measures of edges, triangles, and tetrahedra"
        );
    }

    std::vector<double> _coordinates(3 * expected_connectivity_size);
    //Just for consistency with `connectivity` below.
    double * const coordinates = _coordinates.data();
    std::vector<double>::iterator p = measures[type].begin();
    for (moab::EntityHandle handle : entities[type]) {
        moab::ErrorCode ecode;

        moab::EntityHandle const *connectivity;
        int num_nodes;
        ecode = impl->get_connectivity(handle, connectivity, num_nodes);
        MB_CHK_ERR(ecode);
        assert(num_nodes >= 0);
        assert(
            static_cast<std::size_t>(num_nodes) == expected_connectivity_size
        );

        ecode = impl->get_coords(connectivity, num_nodes, coordinates);
        MB_CHK_ERR(ecode);

        const double volume = *p++ = f(coordinates);
        //Increment for each node in the element the total measure of the
        //elements containing that node.
        if (computing_element_measures) {
            for (
                moab::EntityHandle const *q = connectivity;
                q != connectivity + num_nodes;
                ++q
            ) {
                preconditioner_divisors.at(index(*q)) += volume;
            }
        }
    }
    return moab::MB_SUCCESS;
}

}
