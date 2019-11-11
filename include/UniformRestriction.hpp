#ifndef UNIFORMRESTRICTION_HPP
#define UNIFORMRESTRICTION_HPP
//!\file
//!\brief Restriction for piecewise linears on mesh hierarchies produced by
//!uniform refinement.

#include "LinearOperator.hpp"
#include "MeshLevel.hpp"

namespace mgard {

//!Restriction operator between two `MeshLevel`s, the finer produced by
//!uniformly refining the coarser.
class UniformRestriction: public LinearOperator {
    public:
        //!Constructor.
        //!
        //!\param MESH Fine mesh produced by uniformly refining the coarse mesh.
        //!\param mesh Coarse mesh.
        UniformRestriction(const MeshLevel &MESH, const MeshLevel &mesh);

    private:
        //!Fine mesh.
        const MeshLevel &MESH;

        //!Coarse mesh.
        const MeshLevel &mesh;

        virtual void do_operator_parentheses(
            double const * const F, double * const f
        ) const override;
};

}

#endif
