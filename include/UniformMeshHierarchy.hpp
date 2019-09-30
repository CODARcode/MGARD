#ifndef UNIFORMMESHHIERARCHY_HPP
#define UNIFORMMESHHIERARCHY_HPP

#include <vector>

#include "MeshLevel.hpp"
#include "MeshHierarchy.hpp"

namespace mgard {

//!Hierarchy of meshes produced by uniformly refining an initial mesh.
class UniformMeshHierarchy: public MeshHierarchy {
    public:
        //!Constructor.
        //!
        //!\param meshes Meshes of the hierarchy, from coarsest to finest.
        UniformMeshHierarchy(const std::vector<MeshLevel> &meshes);

        //!Constructor.
        //!
        //!\override
        //!
        //!\param mesh Coarsest mesh in the hierarchy.
        //!\param L Number of times to refine the initial mesh.
        UniformMeshHierarchy(const MeshLevel &mesh, const std::size_t L);

    protected:
        //!Logarithm base 2 of the number of child elements into which each
        //!parent element is split in a single refinement.
        std::size_t log_num_children_per_refinement;

        //!Number of child elements into which each parent element is split when
        //!refining from one mesh level to another.
        //!
        //!\param l Index of mesh containing parent element.
        //!\param m Index of mesh containing child elements.
        std::size_t num_children(
            const std::size_t l, const std::size_t m
        ) const;

        //!Initialize `num_children_per_refinement` from topological dimension
        //!of coarsest mesh.
        void populate_from_topological_dimension();

        //!Find the parent of an element in a fine mesh.
        //!
        //!\param element Handle of an element in the fine mesh.
        //!\param l Index of the fine mesh.
        //!\param m Index of the coarse mesh.
        moab::EntityHandle get_parent(
            const moab::EntityHandle element,
            const std::size_t l,
            const std::size_t m
        ) const;

        //!Find the midpoint of an edge in the mesh immediately finer.
        //!
        //!\param edge Handle of the edge in the coarse mesh.
        //!\param l Index of the coarse mesh.
        //!\param m Index of the fine mesh produced by refining the coarse mesh.
        //!
        //!\return Handle of the node in the fine mesh located at the midpoint
        //!of the edge.
        moab::EntityHandle get_midpoint(
            const moab::EntityHandle edge,
            const std::size_t l,
            const std::size_t m
        ) const;

    private:
        virtual moab::EntityHandle do_replica(
            const moab::EntityHandle node,
            const std::size_t l,
            const std::size_t m
        ) const override;

        virtual moab::Range do_get_children(
            const moab::EntityHandle t,
            const std::size_t l,
            const std::size_t m
        ) const override;

        virtual bool do_is_new_node(
            moab::EntityHandle node, const std::size_t l
        ) const override;

        virtual double do_measure(
            const moab::EntityHandle handle, const std::size_t l
        ) const override;

        //!Interpolate the 'old' values onto the 'new' nodes, scale, and add to
        //!'new' values.
        //!
        //!\param [in, out] u Nodal values of the input function.
        //!\param [in] l Index of the MeshLevel.
        //!\param [in] alpha Factor by which to scale the interpolant.
        virtual moab::ErrorCode do_interpolate_old_to_new_and_axpy(
            double * const u, std::size_t l, const double alpha
        ) const override;

        //!Scale a function on the 'old' nodes and add to the 'old' values.
        //!
        //!\param [in, out] u Nodal values of the input function.
        //!\param [in] l Index of the MeshLevel.
        //!\param [in] alpha Factor by which to scale the function.
        //!\param [in] correction Function to be scaled and added.
        virtual moab::ErrorCode do_old_values_axpy(
            double * const u,
            std::size_t l,
            const double alpha,
            double const * const correction
        ) const override;

        virtual moab::ErrorCode do_apply_mass_matrix_to_multilevel_component(
            double const * const u, const std::size_t l, double * const b
        ) const override;

        virtual moab::EntityHandle do_get_parent(
            const moab::EntityHandle element,
            const std::size_t l,
            const std::size_t m
        ) const;
};

}

#endif
