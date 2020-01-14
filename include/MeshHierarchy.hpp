#ifndef MESHHIERARCHY_HPP
#define MESHHIERARCHY_HPP
//!\file
//!\brief Increasing hierarchy of meshes, with the ability to decompose and
//! recompose functions given on the finest mesh of that hierarchy.

#include <cstddef>

#include <functional>
#include <vector>

#include "moab/Range.hpp"
#include "moab/Types.hpp"

#include "MeshLevel.hpp"
#include "data.hpp"

namespace mgard {

//! Hierarchy of meshes produced by refining an initial mesh.
//!
//! We make a few assumptions about the ordering of the nodes and the associated
//! values.
//!    - The values of any associated dataset have the same ordering as the
//!      nodes. This allows us to use `u[mesh.index(node)]` to get the value
//!      associated to `node`.
//!    - Meshes in a hierarchy agree on the index of nodes they have in common.
//!      That is, if `node` is a node in `mesh`, `NODE` is a node in `MESH`, and
//!      `node` and `NODE` are the same physical node, then `mesh.index(node) ==
//!      MESH.index(NODE)`.
//!    - Let `new_nodes(l)` be the set of 'new' nodes in mesh `l` (those
//!      introduced in the refinement of mesh `l - 1` or, for `l` zero, all the
//!      nodes in the original, coarsest mesh). The ordering is then
//!        -# `new_nodes(0)`
//!        -# `new_nodes(1)`
//!        -# `new_nodes(2)`
//!        -# and so on.
//!      Consequences:
//!        - The values associated to `new_nodes(l)` are contiguous.
//!        - Let `old_nodes(l)` be the set of 'old' nodes in mesh `l` (those
//!          also present in mesh `l - 1`). Then the values associated to
//!          `old_nodes(l)` are contiguous.
//!        - The ordering for the nodes in mesh `l` is
//!            -# `old_nodes(l)`
//!            -# `new_nodes(l)`.
class MeshHierarchy {
public:
  //! Constructor.
  //!
  //!\param meshes Meshes of the hierarchy, from coarsest to finest.
  MeshHierarchy(const std::vector<MeshLevel> &meshes);

  //! Report the number of degrees of freedom in the finest MeshLevel.
  std::size_t ndof() const;

  //! Report the number of degrees of freedom in a MeshLevel.
  //!
  //!\param l Index of the MeshLevel.
  std::size_t ndof(const std::size_t l) const;

  //! Access the 'old' nodes of a level.
  //!
  //!\param [in] l Index of the MeshLevel.
  moab::Range old_nodes(const std::size_t l) const;

  //! Access the 'new' nodes of a level.
  //!
  //!\param [in] l Index of the MeshLevel.
  moab::Range new_nodes(const std::size_t l) const;

  //! Access the subset of a dataset associated to the 'old' nodes of a
  //! level.
  //!
  //!\param [in] u Values associated to nodes.
  //!\param [in] l Index of the MeshLevel.
  double *on_old_nodes(const HierarchyCoefficients<double> u,
                       const std::size_t l) const;

  //! Access the subset of a dataset associated to the 'new' nodes of a
  //! level.
  //!
  //!\param [in] u Values associated to nodes.
  //!\param [in] l Index of the MeshLevel.
  double *on_new_nodes(const HierarchyCoefficients<double> u,
                       const std::size_t l) const;

  //! Transform from nodal coefficients to multilevel coefficients.
  //!
  //!\param [in, out] u Nodal values of the input function.
  //!\param [in] buffer Scratch space.
  MultilevelCoefficients<double> decompose(const NodalCoefficients<double> u,
                                           void *buffer = NULL);

  //! Transform from multilevel coefficients to nodal coefficients.
  //!
  //!\param [in, out] u Multilevel coefficients of the input function.
  //!\param [in] buffer Scratch space.
  NodalCoefficients<double> recompose(const MultilevelCoefficients<double> u,
                                      void *buffer = NULL);

  //! Report the amount of scratch space (in bytes) needed for all
  //! hierarchy operations.
  std::size_t scratch_space_needed() const;

  //! MeshLevels in the hierarchy, ordered from coarsest to finest.
  std::vector<MeshLevel> meshes;

  //! Index of finest MeshLevel.
  std::size_t L;

protected:
  //! Constructor.
  //!
  //!\overload
  //!
  //!\param mesh Coarsest mesh in the hierarchy.
  //!\param refiner Function object to refine the meshes.
  //!\param L Number of times to refine the initial mesh.
  MeshHierarchy(const MeshLevel &mesh,
                std::function<MeshLevel(const MeshLevel &)> refiner,
                const std::size_t L);

  //! Scratch space for use in hierarchy operations if no external buffer
  //! if provided.
  std::vector<char> scratch_space;

  //! Report the amount of scratch space (in bytes) needed for
  //! decomposition.
  std::size_t scratch_space_needed_for_decomposition() const;

  //! Report the amount of scratch space (in bytes) needed for
  //! recomposition.
  std::size_t scratch_space_needed_for_recomposition() const;

  //! Transform from nodal coefficients to multilevel coefficients,
  //! starting at a given level.
  //!
  //!\param [in, out] u Nodal values of the input function.
  //!\param [in] l Index of the MeshLevel to start at.
  //!\param [in] buffer Scratch space.
  moab::ErrorCode decompose(const NodalCoefficients<double> u,
                            const std::size_t l, void *const buffer) const;

  //! Transform from multilevel coefficients to nodal coefficients,
  //! starting at a given level.
  //!
  //!\param [in, out] u Multilevel coefficients of the input function.
  //!\param [in] l Index of the MeshLevel to start at. Must be nonzero!
  //!\param [in] buffer Scratch space.
  moab::ErrorCode recompose(const MultilevelCoefficients<double> u,
                            const std::size_t l, void *const buffer) const;

  //! Project a multilevel component onto the next coarser level.
  //!
  //!\param [in] u Nodal values of the input function.
  //!\param [in] l Index of the MeshLevel
  //!\param [out] correction Nodal values of the projection.
  //!\param [in] buffer Scratch space.
  moab::ErrorCode calculate_correction_from_multilevel_component(
      const HierarchyCoefficients<double> u, const std::size_t l,
      double *const correction, void *const buffer) const;

  //! Apply the mass matrix on the coarser level to a multilevel component.
  //!
  //! This achieves the same result as applying the mass matrix on the
  // finer level and restricting back down.
  //!
  //!\param [in] u Nodal values of the input function.
  //!\param [in] l Index of the MeshLevel.
  //!\param [out] b Matrix–vector product.
  moab::ErrorCode apply_mass_matrix_to_multilevel_component(
      const HierarchyCoefficients<double> u, const std::size_t l,
      double *const b) const;

  //! Interpolate the 'old' values onto the 'new' nodes and subtract.
  //!
  //!\param [in, out] u Nodal values of the input function.
  //!\param [in] l Index of the MeshLevel.
  moab::ErrorCode subtract_interpolant_from_coarser_level_from_new_values(
      const HierarchyCoefficients<double> u, const std::size_t l) const;

  //! Interpolate the 'old' values onto the 'new' nodes and add.
  //!
  //!\param [in, out] u Nodal values of the input function.
  //!\param [in] l Index of the MeshLevel.
  moab::ErrorCode add_interpolant_from_coarser_level_to_new_values(
      const HierarchyCoefficients<double> u, const std::size_t l) const;

  //! Add the correction to the values on the 'old' nodes.
  //!
  //!\param [in, out] u Nodal values of the input function.
  //!\param [in] l Index of the MeshLevel.
  //!\param [in] correction Nodal values of the correction.
  moab::ErrorCode
  add_correction_to_old_values(const HierarchyCoefficients<double> u,
                               const std::size_t l,
                               double const *const correction) const;

  //! Subtract the correction from the values on the 'old' nodes.
  //!
  //!\param [in, out] u Nodal values of the input function.
  //!\param [in] l Index of the MeshLevel.
  //!\param [in] correction Nodal values of the correction.
  moab::ErrorCode
  subtract_correction_from_old_values(const HierarchyCoefficients<double> u,
                                      const std::size_t l,
                                      double const *const correction) const;

  //! Find the handle of a node in a finer mesh.
  //!
  //!\param node Handle of the node in the coarse mesh.
  //!\param l Index of the coarse mesh.
  //!\param m Index of the fine mesh.
  //!
  //!\return Handle of the node in the finest mesh.
  moab::EntityHandle replica(const moab::EntityHandle node, const std::size_t l,
                             const std::size_t m) const;

  //! Find the elements obtained by refining a given element.
  //!
  //!\param t Handle of the 'parent' element in the coarse mesh.
  //!\param l Index of the coarse mesh.
  //!\param m Index of the fine mesh produced by refining the coarse mesh.
  //!
  //!\return Handles of the 'child' elements in the finest mesh.
  moab::Range get_children(const moab::EntityHandle t, const std::size_t l,
                           const std::size_t m) const;

  //! Determine whether a node is 'new' to a mesh in the hierarchy.
  //!
  //!\param node Handle of the node.
  //!\param l Index of the mesh.
  bool is_new_node(const moab::EntityHandle node, const std::size_t l) const;

  //! Find the measure of an entity of a mesh in the hierarchy.
  //!
  //!\param handle Handle of the entity.
  //!\param l Index of the mesh.
  double measure(const moab::EntityHandle handle, const std::size_t l) const;

  //! Check that a mesh index is in bounds.
  //!
  //!\param l Mesh index.
  void check_mesh_index_bounds(const std::size_t l) const;

  //! Check that a pair of mesh indices are nondecreasing.
  //!
  //!\param l Smaller (nonlarger) mesh index.
  //!\param m Larger (nonsmaller) mesh index.
  void check_mesh_indices_nondecreasing(const std::size_t l,
                                        const std::size_t m) const;

  //! Check that a mesh index is nonzero.
  //!
  //!\param l Mesh index.
  void check_mesh_index_nonzero(const std::size_t l) const;

private:
  virtual std::size_t do_ndof(const std::size_t l) const;

  virtual moab::Range do_old_nodes(const std::size_t l) const;

  virtual moab::Range do_new_nodes(const std::size_t l) const;

  virtual double *do_on_old_nodes(const HierarchyCoefficients<double> u,
                                  const std::size_t) const;

  virtual double *do_on_new_nodes(const HierarchyCoefficients<double> u,
                                  const std::size_t l) const;

  virtual std::size_t do_scratch_space_needed() const;

  virtual std::size_t do_scratch_space_needed_for_decomposition() const;

  virtual std::size_t do_scratch_space_needed_for_recomposition() const;

  virtual moab::EntityHandle do_replica(const moab::EntityHandle node,
                                        const std::size_t l,
                                        const std::size_t m) const = 0;

  virtual moab::Range do_get_children(const moab::EntityHandle t,
                                      const std::size_t l,
                                      const std::size_t m) const = 0;

  virtual bool do_is_new_node(const moab::EntityHandle node,
                              const std::size_t l) const = 0;

  virtual double do_measure(const moab::EntityHandle handle,
                            const std::size_t l) const;

  //! Interpolate the 'old' values onto the 'new' nodes, scale, and add to
  //!'new' values.
  //!
  //!\param [in, out] u Nodal values of the input function.
  //!\param [in] l Index of the MeshLevel.
  //!\param [in] alpha Factor by which to scale the interpolant.
  virtual moab::ErrorCode
  do_interpolate_old_to_new_and_axpy(const HierarchyCoefficients<double> u,
                                     const std::size_t l,
                                     const double alpha) const = 0;

  //! Scale a function on the 'old' nodes and add to the 'old' values.
  //!
  //!\param [in, out] u Nodal values of the input function.
  //!\param [in] l Index of the MeshLevel.
  //!\param [in] alpha Factor by which to scale the function.
  //!\param [in] correction Function to be scaled and added.
  virtual moab::ErrorCode
  do_old_values_axpy(const HierarchyCoefficients<double> u, std::size_t l,
                     const double alpha,
                     double const *const correction) const = 0;

  virtual moab::ErrorCode do_calculate_correction_from_multilevel_component(
      const HierarchyCoefficients<double> u, const std::size_t l,
      double *const correction, void *const buffer) const;

  virtual moab::ErrorCode do_apply_mass_matrix_to_multilevel_component(
      const HierarchyCoefficients<double> u, const std::size_t l,
      double *const b) const = 0;
};

} // namespace mgard

#endif
