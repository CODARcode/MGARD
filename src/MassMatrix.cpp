#include "MassMatrix.hpp"

#include <algorithm>
#include <stdexcept>

#include "utilities.hpp"

namespace mgard {

//Public member functions.

MassMatrix::MassMatrix(const MeshLevel &mesh):
    LinearOperator(mesh.ndof()),
    mesh(mesh)
{
    mesh.precompute_element_measures();
}

ContiguousSubsetMassMatrix::ContiguousSubsetMassMatrix(
    const MeshLevel &mesh, const moab::Range nodes
):
    LinearOperator(nodes.size()),
    mesh(mesh),
    nodes(nodes),
    min_node(nodes.front()),
    max_node(nodes.back())
{
    if (nodes.psize() != 1) {
        throw std::invalid_argument("Node subset must be contiguous.");
    }
    moab::Range intersection;
    if (moab::intersect(mesh.entities[moab::MBVERTEX], nodes) != nodes) {
        throw std::invalid_argument(
            "Node subset must be contained within the nodes of the mesh."
        );
    }
}

MassMatrixPreconditioner::MassMatrixPreconditioner(const MeshLevel &mesh):
    LinearOperator(mesh.ndof()),
    mesh(mesh)
{
    mesh.precompute_element_measures();
}

//Private member functions.

void MassMatrix::do_operator_parentheses(
    double const * const x, double * const b
) const {
    std::fill(b, b + range_dimension, 0);
    const moab::Range &elements = mesh.entities[mesh.element_type];
    const std::size_t measure_factor_divisor = (
        (mesh.topological_dimension + 1) * (mesh.topological_dimension + 2)
    );
    for (moab::EntityHandle element : elements) {
        const double measure_factor = (
            mesh.measure(element) / measure_factor_divisor
        );
        //Pairs `(i, v)` where `i` is the global index of a node and `v` is
        //`x[i]`, the value of the function there.
        std::vector<std::pair<std::size_t, double>> nodal_pairs;
        nodal_pairs.reserve(mesh.num_nodes_per_element);
        double nodal_values_sum = 0;
        for (const moab::EntityHandle node : mesh.connectivity(element)) {
            std::pair<std::size_t, double> pair;
            nodal_values_sum += pair.second = x[pair.first = mesh.index(node)];
            nodal_pairs.push_back(pair);
        }
        for (std::pair<std::size_t, double> pair : nodal_pairs) {
            b[pair.first] += measure_factor * (nodal_values_sum + pair.second);
        }
    }
}

std::size_t ContiguousSubsetMassMatrix::index(
    const moab::EntityHandle node
) const {
    return node - min_node;
}

bool ContiguousSubsetMassMatrix::contains(const moab::EntityHandle node) const {
    return min_node <= node && node <= max_node;
}

//This is just a slight modification of analogous `MassMatrix` function. Could
//somehow merge them.
void ContiguousSubsetMassMatrix::do_operator_parentheses(
    double const * const x, double * const b
) const {
    std::fill(b, b + range_dimension, 0);
    const moab::Range &elements = mesh.entities[mesh.element_type];
    const std::size_t measure_factor_divisor = (
        (mesh.topological_dimension + 1) * (mesh.topological_dimension + 2)
    );
    for (moab::EntityHandle element : elements) {
        const double measure_factor = (
            mesh.measure(element) / measure_factor_divisor
        );
        //Pairs `(i, v)` where `i` is the global index of a node and `v` is
        //`x[i]`, the value of the function there. Here 'global' refers to the
        //index in `nodes`/`x`/`b`, not the mesh.
        std::vector<std::pair<std::size_t, double>> nodal_pairs;
        nodal_pairs.reserve(mesh.num_nodes_per_element);
        double nodal_values_sum = 0;
        for (const moab::EntityHandle node : mesh.connectivity(element)) {
            if (!contains(node)) {
                continue;
            }
            std::pair<std::size_t, double> pair;
            nodal_values_sum += pair.second = x[pair.first = index(node)];
            nodal_pairs.push_back(pair);
        }
        for (std::pair<std::size_t, double> pair : nodal_pairs) {
            b[pair.first] += measure_factor * (nodal_values_sum + pair.second);
        }
    }
}

void MassMatrixPreconditioner::do_operator_parentheses(
    double const * const x, double * const b
) const {
    double *p = b;
    double const *q = x;
    for (moab::EntityHandle node : mesh.entities[moab::MBVERTEX]) {
        *p++ = *q++ / mesh.containing_elements_measure(node);
    }
}

}
