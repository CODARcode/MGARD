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
class MassMatrix: public LinearOperator {
    public:
        //!Constructor.
        //!
        //!\param mesh Pointer to underlying mesh.
        MassMatrix(MeshLevel const * const mesh);

    private:
        //!Pointer to underlying mesh.
        MeshLevel const * mesh;

        virtual void do_operator_parentheses(
            double const * const x, double * const b
        ) const override;
};

//!Preconditioner for `MassMatrix`.
class MassMatrixPreconditioner: public LinearOperator {
    public:
        //!Constructor.
        //!
        //!\param mesh Pointer to underlying mesh.
        MassMatrixPreconditioner(MeshLevel const * const mesh);

    private:
        //!Pointer to underlying mesh.
        MeshLevel const * mesh;

        virtual void do_operator_parentheses(
            double const * const x, double * const b
        ) const override;
};

}

#endif
