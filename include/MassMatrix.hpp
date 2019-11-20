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
//!`MeshLevel` where a subset of the hat functions are allowed to be nonzero.
class SubsetMassMatrix: public LinearOperator {
    public:
        //!Constructor.
        //!
        //!\param mesh Underlying mesh.
        //!\param nodes Subset nodes whose associated hat functions are allowed
        //!to be nonzero. This parameter determines the ordering of the input
        //!and output vectors.
        SubsetMassMatrix(const MeshLevel &mesh, const moab::Range &nodes);

    protected:
        //!Underlying mesh.
        const MeshLevel &mesh;

        //!'Allowed' node range.
        const moab::Range &nodes;

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

    private:
        virtual void do_operator_parentheses(
            double const * const x, double * const b
        ) const override;

        virtual std::size_t do_index(const moab::EntityHandle node) const = 0;

        virtual bool do_contains(const moab::EntityHandle node) const = 0;
};

//!Mass matrix for continuous piecewise linear functions defined on a
//!`MeshLevel`.
class MassMatrix: public SubsetMassMatrix {
    public:
        //!Constructor.
        //!
        //!\param mesh Underlying mesh.
        MassMatrix(const MeshLevel &mesh);

    private:
        virtual std::size_t do_index(
            const moab::EntityHandle node
        ) const override;

        virtual bool do_contains(const moab::EntityHandle node) const override;
};

//!Mass matrix for continuous piecewise linear functions defined on a
//!`MeshLevel` where a subset (contiguous by node ordering) of the hat
//!functions are allowed to be nonzero.
//!
//!This is used to compute the \f$ L^{2} \f$ norm of functions in the image of
//!\f$ I -\Pi_{\ell} \f$ (with \f$ I \f$ the identity and \f$ \Pi_{\ell} \f$ the
//!continuous piecewise linear interpolation onto the next coarser mesh in the
//!hierarchy). Currently, 'new' nodes are contiguous.
class ContiguousSubsetMassMatrix: public SubsetMassMatrix {
    public:
        //!Constructor.
        //!
        //!\param mesh Underlying mesh.
        //!\param nodes Contiguous range of nodes whose associated hat functions
        //!are allowed to be nonzero. This parameter determines the ordering of
        //!the input and output vectors.
        ContiguousSubsetMassMatrix(
            const MeshLevel &mesh, const moab::Range &nodes
        );

    private:
        //!First (by handle) 'allowed' node.
        const moab::EntityHandle min_node;

        //!Last (by handle) 'allowed' node.
        const moab::EntityHandle max_node;

        virtual std::size_t do_index(
            const moab::EntityHandle node
        ) const override;

        virtual bool do_contains(const moab::EntityHandle node) const override;
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
