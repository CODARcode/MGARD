#include "UniformMeshHierarchy.hpp"

#include <cstddef>
#include <cassert>

#include <iterator>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

#include "UniformMeshRefiner.hpp"
#include "utilities.hpp"

namespace mgard {

//Public member functions.
UniformMeshHierarchy::UniformMeshHierarchy(
    const std::vector<MeshLevel> &meshes
):
    MeshHierarchy(meshes)
{
    populate_from_topological_dimension();
}

UniformMeshHierarchy::UniformMeshHierarchy(
    const MeshLevel &mesh, const std::size_t L
):
    MeshHierarchy(mesh, UniformMeshRefiner(), L)
{
    populate_from_topological_dimension();
}

//Protected member functions.
void UniformMeshHierarchy::populate_from_topological_dimension() {
    switch (meshes.back().topological_dimension) {
        case (2):
            log_num_children_per_refinement = 2;
            break;
        case (3):
            log_num_children_per_refinement = 3;
            break;
        default:
            throw std::runtime_error("unsupported topological dimension");
    }
}

//Possibly something to ask `UniformMeshRefiner`.
std::size_t UniformMeshHierarchy::num_children(
    const std::size_t l, const std::size_t m
) const {
    check_mesh_index_bounds(l);
    check_mesh_index_bounds(m);
    check_mesh_indices_nondecreasing(l, m);
    return 1 << (log_num_children_per_refinement * (m - l));
}

moab::EntityHandle UniformMeshHierarchy::get_parent(
    const moab::EntityHandle element, const std::size_t l, const std::size_t m
) const {
    check_mesh_index_bounds(l);
    check_mesh_index_bounds(m);
    check_mesh_indices_nondecreasing(m, l);
    const MeshLevel &MESH = meshes.at(l);
    if (MESH.impl.type_from_handle(element) != MESH.element_type) {
        throw std::domain_error("can only find parent of element");
    }
    return do_get_parent(element, l, m);
}

moab::EntityHandle UniformMeshHierarchy::get_midpoint(
    const moab::EntityHandle edge, const std::size_t l, const std::size_t m
) const {
    check_mesh_index_bounds(l);
    check_mesh_index_bounds(m);
    assert(m == l + 1);
    const MeshLevel &mesh = meshes.at(l);
    const MeshLevel &MESH = meshes.at(m);
    const std::size_t n = ndof(l);
    return MESH.entities[moab::MBVERTEX][n + mesh.index(edge)];
}

//Private member functions.
moab::EntityHandle UniformMeshHierarchy::do_replica(
    const moab::EntityHandle node, const std::size_t l, const std::size_t m
) const {
    const MeshLevel &mesh = meshes.at(l);
    const MeshLevel &MESH = meshes.at(m);
    //The first however many nodes in the new meshes are the old nodes, in the
    //same order they were found in the old meshes.
    return MESH.entities[moab::MBVERTEX][mesh.index(node)];
}

moab::Range UniformMeshHierarchy::do_get_children(
    const moab::EntityHandle t, const std::size_t l, const std::size_t m
) const {
    const MeshLevel &mesh = meshes.at(l);
    const std::size_t index = mesh.index(t);
    const std::size_t n = num_children(l, m);
    const MeshLevel &MESH = meshes.at(m);
    //One refinement has the following effect on the element list:
    //    mesh.triangles       MESH.triangles
    //      t0                   t00
    //                           t01
    //                           t02
    //                           t03
    //      t1                   t10
    //                           t11
    //                           t12
    //                           t13
    //where `t00` is the `0`th child of `t0` and so on.
    const moab::EntityType type = mesh.impl.type_from_handle(t);
    const moab::Range &ENTITIES = MESH.entities[type];
    assert(ENTITIES.psize() == 1);
    return moab::Range(ENTITIES[n * index], ENTITIES[n * (index + 1) - 1]);
}

bool UniformMeshHierarchy::do_is_new_node(
    moab::EntityHandle node, const std::size_t l
) const {
    const MeshLevel &MESH = meshes.at(l);
    //The value isn't needed when `l` is zero, but calling `index` will catch
    //the case that `node` isn't actually in the claimed mesh.
    const std::size_t INDEX = MESH.index(node);
    return !l || INDEX >= ndof(l - 1);
}

double UniformMeshHierarchy::do_measure(
    const moab::EntityHandle handle, const std::size_t l
) const {
    const MeshLevel &MESH = meshes.at(l);
    return (
        (!l || MESH.impl.type_from_handle(handle) != MESH.element_type) ?
            //Copied from `MeshHierarchy::do_measure`. Can't directly call that
            //method since it's private. Not sure what's best. Calling
            //`MeshHierarchy::measure` gets us back here (and eventually to a
            //stack overflow).
            MESH.measure(handle) :
            MeshHierarchy::measure(get_parent(handle, l, 0), 0) /
                num_children(0, l)
    );
}

        //!Interpolate the 'old' values onto the 'new' nodes, scale, and add to
        //!'new' values.
        //!
        //!\param [in, out] u Nodal values of the input function.
        //!\param [in] l Index of the MeshLevel.
        //!\param [in] alpha Factor by which to scale the interpolant.

//!Interpolate the 'old' values onto the 'new' nodes, scale, and add to
//!'new' values.
//!
//!\param [in, out] u Nodal values of the input function.
//!\param [in] l Index of the MeshLevel.
//!\param [in] alpha Factor by which to scale the interpolant.
moab::ErrorCode UniformMeshHierarchy::do_interpolate_old_to_new_and_axpy(
    double * const u, std::size_t l, const double alpha
) const {
    if (!l) {
        return moab::MB_SUCCESS;
    }
    const MeshLevel &mesh = meshes.at(l - 1);
    //`p` now points to the value at the first 'new' node. These 'new' nodes
    //have the same ordering as the 'old' edges.
    //TODO: Look into whether you're doing this multiple times and consider a
    //function.
    double *p = u + ndof(l - 1);
    //TODO: *If* you're doing this multiple times, probably worth defining an
    //iterator to give you the pairs `{child, {parent1, parent2}}`.
    for (const moab::EntityHandle edge : mesh.entities[moab::MBEDGE]) {
        double interpolant = 0;
        for (const moab::EntityHandle node : mesh.connectivity(edge)) {
            interpolant += u[mesh.index(node)];
        }
        interpolant *= 0.5;
        *p++ += alpha * interpolant;
    }
    return moab::MB_SUCCESS;
}

//!Scale a function on the 'old' nodes and add to the 'old' values.
//!
//!\param [in, out] u Nodal values of the input function.
//!\param [in] l Index of the MeshLevel.
//!\param [in] alpha Factor by which to scale the function.
//!\param [in] correction Function to be scaled and added.
moab::ErrorCode UniformMeshHierarchy::do_old_values_axpy(
    double * const u,
    std::size_t l,
    const double alpha,
    double const * const correction
) const {
    //`l` is checked to be nonzero in the caller.
    const std::size_t n = ndof(l - 1);
    double *p = u;
    double const *q = correction;
    for (std::size_t i = 0; i < n; ++i) {
        *p++ += alpha * *q++;
    }
    return moab::MB_SUCCESS;
}

moab::ErrorCode
UniformMeshHierarchy::do_apply_mass_matrix_to_multilevel_component(
    double const * const u, const std::size_t l, double * const b
) const {
    moab::ErrorCode ecode;
    //`l` is checked to be nonzero in the caller.
    const std::size_t n = ndof(l - 1);
    for (double *p = b; p != b + n; ++p) {
        *p = 0;
    }
    const MeshLevel &mesh = meshes.at(l - 1);
    const MeshLevel &MESH = meshes.at(l);
    //Copied from `MeshLevel::mass_matrix_matvec`.
    const std::size_t measure_factor_denominator = (
        (MESH.topological_dimension + 1) * (MESH.topological_dimension + 2)
    );

    //Would put this inside the loop but we know `mesh.num_nodes_per_element`
    //doesn't depend on `element`.
    for (const moab::EntityHandle t : mesh.entities[mesh.element_type]) {
        //We want to integrate hat functions on the fine mesh (call them
        //'Φ(·; X)s') against hat functions on the coarse mesh (call them
        //'φ(·; x)s'). To do this, we write each φ(·; x) as a sum of Φ(·; X)s,
        //each weighted by 1 (if X = x, so that φ(X; x) = 1) or 0.5 (if X ≠ x
        //but x and X are adjacent, so that φ(X; x) = 0.5). We will integrate
        //the input data against the Φs and these values will tell us how to
        //translate the results into integrals against the φs.
        //The map looks like this:
        //    {
        //        X: [{x1, 0.5}, {x2, 0.5}],
        //        Y: [{y, 1}],
        //        ...
        //    }
        //Nodes in the new mesh are mapped to the nodes in the old mesh they
        //'contribute to,' so to speak. (The contribution is to the integrals
        //against the (wide) φ(·, x)s centered at the old nodes x, and it's done
        //by the integrals against the (narrow) Φ(·, X)s centered at the new
        //nodes X.)
        std::unordered_map<
            moab::EntityHandle,
            std::vector<std::pair<moab::EntityHandle, double>>
        > interpolation_key;

        //Mapping a node x on the coarse grid to contributions to the integral
        //of the multilevel component against φ(·, x).
        std::unordered_map<moab::EntityHandle, double> local_b;
        //We'll be finding these nodes again (piecemeal, edge by edge) below.
        //Could therefore eliminate this call if it's slow.
        for (const moab::EntityHandle x : mesh.connectivity(t)) {
            interpolation_key.insert({replica(x, l - 1, l), {{x, 1}}});
            local_b.insert({x, 0});
        }

        std::vector<moab::EntityHandle> edges;
        ecode = mesh.impl.get_adjacencies(&t, 1, 1, false, edges);
        MB_CHK_ERR(ecode);
        for (moab::EntityHandle edge : edges) {
            const moab::EntityHandle MIDPOINT = get_midpoint(edge, l - 1, l);
            const helpers::PseudoArray<
                const moab::EntityHandle
            > connectivity = mesh.connectivity(edge);
            interpolation_key.insert({
                MIDPOINT, {
                    {connectivity[0], 0.5}, {connectivity[1], 0.5}
                }
            });
        }

        const moab::Range TS = get_children(t, l - 1, l);
        for (moab::EntityHandle T : TS) {
            const double measure_factor = (
                measure(T, l) / measure_factor_denominator
            );
            const helpers::PseudoArray<
                const moab::EntityHandle
            > CONNECTIVITY = MESH.connectivity(T);
            for (const moab::EntityHandle X : CONNECTIVITY) {
                //We're decomposing the input, a multilevel component, into a
                //sum of hat functions on the fine level (Φ(·, X)s). The
                //multilevel component is zero on the old nodes, so the
                //corresponding hat functions 'aren't in' our decomposition of
                //the input.
                if (!is_new_node(X, l)) {
                    continue;
                }
                for (const moab::EntityHandle Y : CONNECTIVITY) {
                    //Integration of Φ(·; X) against Φ(·; Y) over `T` (scaled by
                    //the value the input takes at X (that is, the coefficient
                    //of Φ(·; X)).
                    //This is really not great. Have to think.
                    double integral = u[MESH.index(X)] * measure_factor *
                        (X == Y ? 2 : 1);
                    //For each φ(·, y) that Φ(·, Y) is 'a part of', credit the
                    //integral against Φ(·, Y) in part or in full to φ(·, y).
                    for (auto [y, weight] : interpolation_key.at(Y)) {
                        local_b.at(y) += weight * integral;
                    };
                }
            }
        }
        for (auto [x, increment] : local_b) {
            b[mesh.index(x)] += increment;
        }
    }
    return moab::MB_SUCCESS;
}

moab::EntityHandle UniformMeshHierarchy::do_get_parent(
    const moab::EntityHandle element, const std::size_t l, const std::size_t m
) const {
    const MeshLevel &MESH = meshes.at(l);
    const MeshLevel &mesh = meshes.at(m);
    return mesh.entities[mesh.element_type][
        MESH.index(element) / num_children(m, l)
    ];
}

}
