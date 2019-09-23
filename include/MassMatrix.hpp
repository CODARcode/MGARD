#ifndef MASSMATRIX_HPP
#define MASSMATRIX_HPP
//!\file
//!\brief Mass matrix and preconditioner for continuous piecewise linears.

#include <cstddef>

#include "LinearOperator.hpp"
#include "MeshLevel.hpp"

namespace mgard {

//!Mass matrix for continuous piecewise linear functions defined on a
//!`MeshLevel`.
class MassMatrix: public helpers::LinearOperator {
    public:
        MassMatrix(MeshLevel const * const mesh);

    private:
        MeshLevel const * mesh;

        virtual void do_operator_parentheses(
            double const * const x, double * const b
        ) const override;
};

//!Preconditioner for `MassMatrix`.
class MassMatrixPreconditioner: public helpers::LinearOperator {
    public:
        MassMatrixPreconditioner(MeshLevel const * const mesh);

    private:
        MeshLevel const * mesh;

        virtual void do_operator_parentheses(
            double const * const x, double * const b
        ) const override;
};

}

#endif
