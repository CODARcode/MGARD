#include <cassert>

#include <stdexcept>

#include "MassMatrix.hpp"

namespace mgard {

//Public member functions.

MassMatrix::MassMatrix(MeshLevel const * const mesh):
    helpers::LinearOperator(mesh->ndof()),
    mesh(mesh)
{
    mesh->precompute_element_measures();
}

MassMatrixPreconditioner::MassMatrixPreconditioner(
    MeshLevel const * const mesh
):
    helpers::LinearOperator(mesh->ndof()),
    mesh(mesh)
{
    mesh->precompute_element_measures();
}

//Private member functions.

void MassMatrix::do_operator_parentheses(
    double const * const x, double * const b
) const {
    for (double *p = b; p != b + range_dimension; ++p) {
        *p = 0;
    }
    const moab::Range &elements = mesh->entities[mesh->element_type];
    const std::size_t measure_factor_divisor = (
        (mesh->topological_dimension + 1) * (mesh->topological_dimension + 2)
    );
    for (moab::EntityHandle element : elements) {
        const double measure_factor = (
            mesh->measure(element) / measure_factor_divisor
        );
        moab::EntityHandle const *connectivity;
        int n;
        moab::ErrorCode ecode = mesh->impl->get_connectivity(
            element, connectivity, n
        );
        MB_CHK_ERR_RET(ecode);
        //Pairs `(i, v)` where `i` is the global index of a node and `v` is
        //`x[i]`, the value of the function there.
        std::vector<std::pair<std::size_t, double>> nodal_pairs(n);
        double nodal_values_sum = 0;
        assert(n >= 0);
        for (std::size_t i = 0; i < static_cast<std::size_t>(n); ++i) {
            std::pair<std::size_t, double> &pair = nodal_pairs.at(i);
            nodal_values_sum += (
                pair.second = x[pair.first = mesh->index(connectivity[i])]
            );
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
    for (moab::EntityHandle node : mesh->entities[moab::MBVERTEX]) {
        *p++ = *q++ / mesh->containing_elements_measure(node);
    }
}

}
