#ifndef MASSMATRIX_HPP
#define MASSMATRIX_HPP
//!\file
//!\brief Mass matrix and preconditioner for continuous piecewise linears.

#include <cstddef>

#include "moab/EntityHandle.hpp"

#include "LinearOperator.hpp"
#include "MeshLevel.hpp"

namespace mgard {

//!Mass matrix for continuous piecewise linear functions defined on a
//!`MeshLevel`.
class MassMatrix: public LinearOperator {
    public:
        //!Constructor.
        //!
        //!\param mesh Underlying mesh.
        MassMatrix(const MeshLevel &mesh);

    private:
        //!Underlying mesh.
        const MeshLevel &mesh;

        virtual void do_operator_parentheses(
            double const * const x, double * const b
        ) const override;
};

//!Mass matrix for continuous piecewise linear functions defined on a
//!`MeshLevel` where only a subset (contiguous by node ordering) of the hat
//!functions are allowed to be nonzero.
//!
//!This is used to compute the \f$ L^{2} \f$ norm of functions in the image of
//!\f$ I -\Pi_{\ell} \f$ (with \f$ I \f$ the identity and \f$ \Pi_{\ell} \f$ the
//!continuous piecewise linear interpolation onto the next coarser mesh in the
//!hierarchy). Currently, 'new' nodes are contiguous.
class ContiguousSubsetMassMatrix: public LinearOperator {
    public:
        //!Constructor.
        //!
        //!\param mesh Underlying mesh.
        //!\param nodes Contiguous range of nodes whose associated hat functions
        //!are allowed to be nonzero. This parameter determines the ordering of
        //!the input and output vectors.
        ContiguousSubsetMassMatrix(
            const MeshLevel &mesh, const moab::Range nodes
        );

    private:
        //!Underlying mesh.
        const MeshLevel &mesh;

        //!'Allowed' node range.
        const moab::Range nodes;

        //!First (by handle) 'allowed' node.
        const moab::EntityHandle min_node;

        //!Last (by handle) 'allowed' node.
        const moab::EntityHandle max_node;

        //!Find the index (as determined by the subset) of a node. Does not
        //!perform any error checking.
        //!
        //!\param node Handle of the node.
        //!
        //!\return Index of the node.
        std::size_t index(const moab::EntityHandle node) const;

        //!Determine whether a node is included in the subset.
        //!
        //!\param node Handle of the node.
        bool contains(const moab::EntityHandle node) const;

        virtual void do_operator_parentheses(
            double const * const x, double * const b
        ) const override;
};

//!Preconditioner for `MassMatrix`.
class MassMatrixPreconditioner: public LinearOperator {
    public:
        //!Constructor.
        //!
        //!\param mesh Underlying mesh.
        MassMatrixPreconditioner(const MeshLevel &mesh);

    private:
        //!Underlying mesh.
        const MeshLevel &mesh;

        virtual void do_operator_parentheses(
            double const * const x, double * const b
        ) const override;
};

}

#endif
