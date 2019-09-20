#ifndef MASSMATRIX_HPP
#define MASSMATRIX_HPP

#include <cstddef>

#include "LinearOperator.hpp"
#include "MeshLevel.hpp"

namespace mgard {

class MassMatrix: public helpers::LinearOperator {
    public:
        MassMatrix(MeshLevel const * const mesh);

    private:
        MeshLevel const * mesh;

        virtual void do_operator_parentheses(
            double const * const x, double * const b
        ) const override;
};

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
