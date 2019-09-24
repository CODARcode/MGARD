#ifndef MESHHIERARCHY_HPP
#define MESHHIERARCHY_HPP
//!file
//!brief Increasing hierarchy of meshes, with the ability to decompose and
//!recompose functions given on the finest mesh of that hierarchy.

#include <cstddef>

#include <vector>

#include "MeshLevel.hpp"
#include "MeshRefiner.hpp"

namespace mgard {

//I would like to allow `u` in `{de,re}compose` to be something other than
//`double *.` `double *` seems to force certain assumptions.
//    1. The nodal values are contiguous.
//    2. The nodal values have the same order as the nodes.
//    3. The nodal values corresponding to coarser levels are grouped together,
//       and probably at the start of `u`.
//We may well want to violate these assumptions. For example, in the structured
//1D case the nodal values had the same order as the nodes but the values
//corresponding to coarser levels were separated by stride 2^n. Something to
//return to if necessary.

//!Hierarchy of meshes produced by refining an initial mesh.
class MeshHierarchy {
    public:
        //!Constructor.
        //!
        //!\param meshes Meshes of the hierarchy, from coarsest to finest.
        MeshHierarchy(const std::vector<MeshLevel> &meshes);

        //!Report the number of degrees of freedom in the finest MeshLevel.
        std::size_t ndof() const;

        //!Transform from nodal coefficients to multilevel coefficients.
        //!
        //!\param [in, out] u Nodal values of the input function.
        //!\param [in] buffer Scratch space.
        moab::ErrorCode decompose(double * const u, void *buffer = NULL);

        //!Transform from multilevel coefficients to nodal coefficients.
        //!
        //!\param [in, out] u Multilevel coefficients of the input function.
        //!\param [in] buffer Scratch space.
        moab::ErrorCode recompose(double * const u, void *buffer = NULL);

        //!Report the amount of scratch space (in bytes) needed for all
        //!hierarchy operations.
        std::size_t scratch_space_needed() const;

    protected:
        //!Constructor.
        //!
        //!\override
        //!
        //!\param mesh Coarsest mesh in the hierarchy.
        //!\param refiner Function object to refine the meshes.
        //!\param L Number of times to refine the initial mesh.
        MeshHierarchy(
            const MeshLevel &mesh, MeshRefiner &refiner, const std::size_t L
        );

        //!MeshLevels in the hierarchy, ordered from coarsest to finest.
        std::vector<MeshLevel> meshes;

        //!Index of finest MeshLevel.
        std::size_t L;

        //!Scratch space for use in hierarchy operations if no external buffer
        //if provided.
        std::vector<char> scratch_space;

        //!Report the number of degrees of freedom in a MeshLevel.
        //!
        //!\param l Index of the MeshLevel.
        std::size_t ndof(const std::size_t l) const;

        //!Report the amount of scratch space (in bytes) needed for
        //!decomposition.
        std::size_t scratch_space_needed_for_decomposition() const;

        //!Report the amount of scratch space (in bytes) needed for
        //!recomposition.
        std::size_t scratch_space_needed_for_recomposition() const;

        //!Transform from nodal coefficients to multilevel coefficients,
        //!starting at a given level.
        //!
        //!\param [in, out] u Nodal values of the input function.
        //!\param [in] l Index of the MeshLevel to start at.
        //!\param [in] buffer Scratch space.
        moab::ErrorCode decompose(
            double * const u, const std::size_t l, void * const buffer
        ) const;

        //!Transform from multilevel coefficients to nodal coefficients,
        //!starting at a given level.
        //!
        //!\param [in, out] u Multilevel coefficients of the input function.
        //!\param [in] l Index of the MeshLevel to start at. Must be nonzero!
        //!\param [in] buffer Scratch space.
        moab::ErrorCode recompose(
            double * const u, const std::size_t l, void * const buffer
        ) const;

        //!Project a multilevel component onto the next coarser level.
        //!
        //!\param [in] u Nodal values of the input function.
        //!\param [in] l Index of the MeshLevel
        //!\param [out] correction Nodal values of the projection.
        //!\param [in] buffer Scratch space.
        moab::ErrorCode calculate_correction_from_multilevel_component(
            double const * const u,
            const std::size_t l,
            double * const correction,
            void * const buffer
        ) const;

        //!Apply the mass matrix on the coarser level to a multilevel component.
        //!
        //!This achieves the same result as applying the mass matrix on the
        //finer level and restricting back down.
        //!
        //!\param [in] u Nodal values of the input function.
        //!\param [in] l Index of the MeshLevel.
        //!\param [out] b Matrix–vector product.
        moab::ErrorCode apply_mass_matrix_to_multilevel_component(
            double const * const u, const std::size_t l, double * const b
        ) const;

        //!Interpolate the 'old' values onto the 'new' nodes and subtract.
        //!
        //!\param [in, out] Nodal values of the input function.
        //!\param [in] l Index of the MeshLevel.
        moab::ErrorCode subtract_interpolant_from_coarser_level_from_new_values(
            double * const u, const std::size_t l
        ) const;

        //!Interpolate the 'old' values onto the 'new' nodes and add.
        //!
        //!\param [in, out] Nodal values of the input function.
        //!\param [in] l Index of the MeshLevel.
        moab::ErrorCode add_interpolant_from_coarser_level_to_new_values(
            double * const u, const std::size_t l
        ) const;

        //!Add the correction to the values on the 'old' nodes.
        //!
        //!\param [in, out] u Nodal values of the input function.
        //!\param [in] Index of the MeshLevel.
        //!\param [in] correction Nodal values of the correction.
        moab::ErrorCode add_correction_to_old_values(
            double * const u,
            const std::size_t l,
            double const * const correction
        ) const;

        //!Subtract the correction from the values on the 'old' nodes.
        //!
        //!\param [in, out] u Nodal values of the input function.
        //!\param [in] Index of the MeshLevel.
        //!\param [in] correction Nodal values of the correction.
        moab::ErrorCode subtract_correction_from_old_values(
            double * const u,
            const std::size_t l,
            double const * const correction
        ) const;

        //!Find the handle of a node in a finer mesh.
        //!
        //!\param node Handle of the node in the coarse mesh.
        //!\param l Index of the coarse mesh.
        //!\param m Index of the fine mesh.
        //!
        //!\return Handle of the node in the finest mesh.
        moab::EntityHandle replica(
            const moab::EntityHandle node,
            const std::size_t l,
            const std::size_t m
        ) const;

        //!Find the elements obtained by refining a given element.
        //!
        //!\param t Handle of the 'parent' element in the coarse mesh.
        //!\param l Index of the coarse mesh.
        //!\param m Index of the fine mesh produced by refining the coarse mesh.
        //!
        //!\return Handles of the 'child' elements in the finest mesh.
        moab::Range get_children(
            const moab::EntityHandle t,
            const std::size_t l,
            const std::size_t m
        ) const;

        //!Determine whether a node is 'new' to a mesh in the hierarchy.
        //!
        //!\param node Handle of the node.
        //\!param l Index of the mesh.
        bool is_new_node(moab::EntityHandle node, const std::size_t l) const;

        //!Find the measure of an entity of a mesh in the hierarchy.
        //!
        //!\param handle Handle of the entity.
        //!\param l Index of the mesh.
        double measure(moab::EntityHandle handle, const std::size_t l) const;

        void check_mesh_index_bounds(const std::size_t l) const;

        void check_mesh_indices_nondecreasing(
            const std::size_t l, const std::size_t m
        ) const;

        void check_mesh_index_nonzero(const std::size_t l) const;

    private:
        virtual std::size_t do_ndof(const std::size_t l) const;

        virtual std::size_t do_scratch_space_needed() const;

        virtual
        std::size_t do_scratch_space_needed_for_decomposition() const = 0;

        virtual
        std::size_t do_scratch_space_needed_for_recomposition() const = 0;

        virtual moab::EntityHandle do_replica(
            const moab::EntityHandle node,
            const std::size_t l,
            const std::size_t m
        ) const = 0;

        virtual moab::Range do_get_children(
            const moab::EntityHandle t,
            const std::size_t l,
            const std::size_t m
        ) const = 0;

        virtual bool do_is_new_node(
            moab::EntityHandle node, const std::size_t l
        ) const = 0;

        virtual double do_measure(
            const moab::EntityHandle handle, const std::size_t l
        ) const;

        //!Interpolate the 'old' values onto the 'new' nodes, scale, and add to
        //!'new' values.
        //!
        //!\param [in, out] u Nodal values of the input function.
        //!\param [in] l Index of the MeshLevel.
        //!\param [in] alpha Factor by which to scale the interpolant.
        virtual moab::ErrorCode do_interpolate_old_to_new_and_axpy(
            double * const u, std::size_t l, const double alpha
        ) const = 0;

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
        ) const = 0;

        virtual
        moab::ErrorCode do_calculate_correction_from_multilevel_component(
            double const * const u,
            const std::size_t l,
            double * const correction,
            void * const buffer
        ) const;

        virtual moab::ErrorCode do_apply_mass_matrix_to_multilevel_component(
            double const * const u, const std::size_t l, double * const b
        ) const = 0;
};

}

#endif
