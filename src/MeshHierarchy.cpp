#include "MeshHierarchy.hpp"

#include <cassert>

#include <algorithm>
#include <stdexcept>
#include <type_traits>

#include "MassMatrix.hpp"
#include "pcg.hpp"

namespace mgard {

// Public member functions.

MeshHierarchy::MeshHierarchy(const std::vector<MeshLevel> &meshes)
    : meshes(meshes), L(meshes.size() - 1) {
  // Note that `L` will wrap around if `meshes` is empty.
  // TODO: how to deal with empty `meshes`? Currently something like
  //    MeshHierarchy hierarchy({});
  //    hierarchy.ndof();
  // will cause an error.
}

std::size_t MeshHierarchy::ndof() const { return ndof(L); }

std::size_t MeshHierarchy::ndof(const std::size_t l) const {
  check_mesh_index_bounds(l);
  return do_ndof(l);
}

std::size_t MeshHierarchy::ndof_old(const std::size_t l) const {
  return l ? ndof(l - 1) : 0;
}

std::size_t MeshHierarchy::ndof_new(const std::size_t l) const {
  // Not calling `check_mesh_index_bounds` here because `ndof` will call it.
  return ndof(l) - ndof_old(l);
}

RangeSlice<moab::Range::const_iterator>
MeshHierarchy::old_nodes(const std::size_t l) const {
  check_mesh_index_bounds(l);
  return do_old_nodes(l);
}

RangeSlice<moab::Range::const_iterator>
MeshHierarchy::new_nodes(const std::size_t l) const {
  check_mesh_index_bounds(l);
  return do_new_nodes(l);
}

RangeSlice<double *>
MeshHierarchy::on_old_nodes(const HierarchyCoefficients<double> u,
                            const std::size_t l) const {
  check_mesh_index_bounds(l);
  return do_on_old_nodes(u, l);
}

RangeSlice<double *>
MeshHierarchy::on_new_nodes(const HierarchyCoefficients<double> u,
                            const std::size_t l) const {
  check_mesh_index_bounds(l);
  return do_on_new_nodes(u, l);
}

std::size_t MeshHierarchy::scratch_space_needed() const {
  return do_scratch_space_needed();
}

MultilevelCoefficients<double>
MeshHierarchy::decompose(const NodalCoefficients<double> u, void *buffer) {
  if (buffer == NULL) {
    const std::size_t N = scratch_space_needed_for_decomposition();
    if (scratch_space.size() < N) {
      scratch_space.resize(N);
    }
    buffer = scratch_space.data();
  }
  const moab::ErrorCode ecode = decompose(u, L, buffer);
  if (ecode != moab::MB_SUCCESS) {
    throw std::runtime_error("MOAB error in decomposition");
  }
  return MultilevelCoefficients<double>(u.data);
}

NodalCoefficients<double>
MeshHierarchy::recompose(const MultilevelCoefficients<double> u, void *buffer) {
  if (buffer == NULL) {
    const std::size_t N = scratch_space_needed_for_recomposition();
    if (scratch_space.size() < N) {
      scratch_space.resize(N);
    }
    buffer = scratch_space.data();
  }
  // Bit of awkwardness because the recomposition involves level `l - 1` and
  // nothing is set up to consider -1 a valid mesh index (no least because
  //`l` is unsigned). So, though nothing is done on level 0, we can't handle
  // that elegantly (run `recompose(u, 0, buffer)` to no effect).
  if (L) {
    const moab::ErrorCode ecode = recompose(u, 1, buffer);
    if (ecode != moab::MB_SUCCESS) {
      throw std::runtime_error("MOAB error in recomposition");
    }
  }
  return NodalCoefficients<double>(u.data);
}

bool operator==(const MeshHierarchy &a, const MeshHierarchy &b) {
  return a.meshes == b.meshes;
}

bool operator!=(const MeshHierarchy &a, const MeshHierarchy &b) {
  return !operator==(a, b);
}

// Protected member functions.

// I am not positive that the following is correct. Take it as an explanation of
// why I made the argument type `std::function`.
//`refiner` is intended to be a `MeshRefiner`. If we were to specify its type as
//`MeshRefiner`, it would need to be passed as a pointer or reference because
//`MeshRefiner` is an abstract type. But when `UniformMeshHierarchy` calls this
// constructor we want to pass a temporary `UniformMeshRefiner`, so it has to be
// a const reference. But what if we want `refiner` to keep some state between
// refinements or something?
MeshHierarchy::MeshHierarchy(
    const MeshLevel &mesh, std::function<MeshLevel(const MeshLevel &)> refiner,
    const std::size_t _L)
    : MeshHierarchy({mesh}) {
  //`this->L` is now set to zero. We'll increment it every time we append to
  //`meshes`.
  // Reserving rather than resizing so we don't need to provide a default
  // constructor for `MeshLevel`.
  meshes.reserve(_L + 1);
  for (std::size_t i = 0; i < _L; ++i) {
    meshes.push_back(refiner(meshes.back()));
    ++L;
  }
}

std::size_t MeshHierarchy::scratch_space_needed_for_decomposition() const {
  return do_scratch_space_needed_for_decomposition();
}

std::size_t MeshHierarchy::scratch_space_needed_for_recomposition() const {
  return do_scratch_space_needed_for_recomposition();
}

// Let `nodes(l)` be the set of nodes in level `l`. `nodes(l)` is the disjoint
// union of `old_nodes(l)`, the set of nodes in level `l - 1`, and
//`new_nodes(l)`, the set of nodes 'new' to level `l`.

moab::ErrorCode MeshHierarchy::decompose(const NodalCoefficients<double> u,
                                         const std::size_t l,
                                         void *const buffer) const {
  check_mesh_index_bounds(l);
  if (!l) {
    return moab::MB_SUCCESS;
  }

  moab::ErrorCode ecode;
  const std::size_t n = ndof(l - 1);
  char *_buffer = static_cast<char *>(buffer);
  // Projection of the multilevel component on level `l` onto level `l - 1`
  //(size `n`).
  double *const correction = reinterpret_cast<double *>(_buffer);
  _buffer += 1 * n * sizeof(*correction);

  // At this stage, the values on `N` encode `Q_{l}u`.

  // Subtract the interpolant onto level `l` from the values on `new_nodes(l)`.
  ecode = subtract_interpolant_from_coarser_level_from_new_values(u, l);
  MB_CHK_ERR(ecode);
  // This leaves `(I - Π_{l - 1})Q_{l}u` encoded on `new_nodes(l)` and
  //`Π_{l - 1}Q_{l}u` encoded on `old_nodes(l)`.

  // Compute the correction `Q_{l - 1}u - Π_{l - 1}Q_{l}u` from the multilevel
  // component `(I - Π_{l - 1})Q_{l}u`.
  ecode = calculate_correction_from_multilevel_component(
      u, l, correction, static_cast<void *>(_buffer));
  MB_CHK_ERR(ecode);

  // Add the correction to `Π_{l - 1}Q_{l}u`, stored on `old_nodes(l)`, to
  // obtain `Q_{l - 1}u` on `old_nodes(l)`.
  ecode = add_correction_to_old_values(u, l, correction);
  MB_CHK_ERR(ecode);

  // Continue recursively.
  return decompose(u, l - 1, buffer);
}

moab::ErrorCode MeshHierarchy::recompose(const MultilevelCoefficients<double> u,
                                         const std::size_t l,
                                         void *const buffer) const {
  check_mesh_index_bounds(l);
  check_mesh_index_nonzero(l);

  moab::ErrorCode ecode;
  const std::size_t n = ndof(l - 1);
  char *_buffer = static_cast<char *>(buffer);
  double *const correction = reinterpret_cast<double *>(_buffer);
  _buffer += 1 * n * sizeof(*correction);

  // We start with `Q_{l - 1}u` on `old_nodes(l)` and the multilevel component
  //`(I - Π_{l - 1})Q_{l}u` on `new_nodes(l)`. The first step is to transform
  //`Q_{l - 1}u` to `Π_{l - 1}Q_{l}u`. This is done by projecting
  //`(I - Π_{l - 1})Q_{l}u` to level `l - 1` (computing the 'correction') and
  // subtracting from `Q_{l - 1}u`.
  ecode = calculate_correction_from_multilevel_component(
      u, l, correction, static_cast<void *>(_buffer));
  MB_CHK_ERR(ecode);
  // We now subtract the correction from `Q_{l - 1}u` on `old_nodes(l)`.
  ecode = subtract_correction_from_old_values(u, l, correction);
  MB_CHK_ERR(ecode);

  // This leaves us with `(I - Π_{l - 1})Q_{l}u` on `new_nodes(l)` and
  //`Π_{l - 1}Q_{l}u` on `old_nodes(l)`. Once we update the values on
  //`new_nodes(l)` by adding the interpolant we'll have `Q_{l}u` on `N`.
  ecode = add_interpolant_from_coarser_level_to_new_values(u, l);
  MB_CHK_ERR(ecode);

  return (l != L) ? recompose(u, l + 1, buffer) : moab::MB_SUCCESS;
}

moab::ErrorCode MeshHierarchy::calculate_correction_from_multilevel_component(
    const HierarchyCoefficients<double> u, const std::size_t l,
    double *const correction, void *const buffer) const {
  check_mesh_index_bounds(l);
  check_mesh_index_nonzero(l);
  return do_calculate_correction_from_multilevel_component(u, l, correction,
                                                           buffer);
}

moab::ErrorCode MeshHierarchy::apply_mass_matrix_to_multilevel_component(
    const HierarchyCoefficients<double> u, const std::size_t l,
    double *const b) const {
  check_mesh_index_bounds(l);
  check_mesh_index_nonzero(l);
  return do_apply_mass_matrix_to_multilevel_component(u, l, b);
}

moab::ErrorCode
MeshHierarchy::subtract_interpolant_from_coarser_level_from_new_values(
    const HierarchyCoefficients<double> u, const std::size_t l) const {
  check_mesh_index_bounds(l);
  check_mesh_index_nonzero(l);
  return do_interpolate_old_to_new_and_axpy(u, l, -1);
}

moab::ErrorCode MeshHierarchy::add_interpolant_from_coarser_level_to_new_values(
    const HierarchyCoefficients<double> u, const std::size_t l) const {
  check_mesh_index_bounds(l);
  check_mesh_index_nonzero(l);
  return do_interpolate_old_to_new_and_axpy(u, l, 1);
}

moab::ErrorCode MeshHierarchy::add_correction_to_old_values(
    const HierarchyCoefficients<double> u, const std::size_t l,
    double const *const correction) const {
  check_mesh_index_bounds(l);
  check_mesh_index_nonzero(l);
  return do_old_values_axpy(u, l, 1, correction);
}

moab::ErrorCode MeshHierarchy::subtract_correction_from_old_values(
    const HierarchyCoefficients<double> u, const std::size_t l,
    double const *const correction) const {
  check_mesh_index_bounds(l);
  check_mesh_index_nonzero(l);
  return do_old_values_axpy(u, l, -1, correction);
}

moab::EntityHandle MeshHierarchy::replica(const moab::EntityHandle node,
                                          const std::size_t l,
                                          const std::size_t m) const {
  check_mesh_index_bounds(l);
  check_mesh_index_bounds(m);
  check_mesh_indices_nondecreasing(l, m);
  return do_replica(node, l, m);
}

moab::Range MeshHierarchy::get_children(const moab::EntityHandle t,
                                        const std::size_t l,
                                        const std::size_t m) const {
  check_mesh_index_bounds(l);
  check_mesh_index_bounds(m);
  check_mesh_indices_nondecreasing(l, m);
  const MeshLevel &mesh = meshes.at(l);
  if (mesh.impl.type_from_handle(t) != mesh.element_type) {
    throw std::domain_error("can only find children of elements");
  }
  return do_get_children(t, l, m);
}

double MeshHierarchy::measure(const moab::EntityHandle handle,
                              const std::size_t l) const {
  check_mesh_index_bounds(l);
  return do_measure(handle, l);
}

bool MeshHierarchy::is_new_node(const moab::EntityHandle node,
                                const std::size_t l) const {
  check_mesh_index_bounds(l);
  const MeshLevel &mesh = meshes.at(l);
  if (mesh.impl.type_from_handle(node) != moab::MBVERTEX) {
    throw std::domain_error("entity not node as claimed");
  }
  return do_is_new_node(node, l);
}

void MeshHierarchy::check_mesh_index_bounds(const std::size_t l) const {
  //`l` is guaranteed to be nonnegative because `std::size_t` is unsigned.
  if (!(std::is_unsigned<std::size_t>::value && l <= L)) {
    throw std::out_of_range("mesh index out of range encountered");
  }
}

void MeshHierarchy::check_mesh_indices_nondecreasing(
    const std::size_t l, const std::size_t m) const {
  if (!(l <= m)) {
    throw std::invalid_argument("mesh indices should be nondecreasing");
  }
}

void MeshHierarchy::check_mesh_index_nonzero(const std::size_t l) const {
  if (!l) {
    throw std::out_of_range("mesh index should be nonzero");
  }
}

// Private member functions.

std::size_t MeshHierarchy::do_ndof(const std::size_t l) const {
  return meshes.at(l).ndof();
}

RangeSlice<moab::Range::const_iterator>
MeshHierarchy::do_old_nodes(const std::size_t l) const {
  const moab::Range &nodes = meshes.at(l).entities[moab::MBVERTEX];
  const moab::Range::const_iterator start = nodes.begin();
  return {start, start + ndof_old(l)};
}

RangeSlice<moab::Range::const_iterator>
MeshHierarchy::do_new_nodes(const std::size_t l) const {
  const moab::Range &nodes = meshes.at(l).entities[moab::MBVERTEX];
  return {nodes.begin() + ndof_old(l), nodes.end()};
}

RangeSlice<double *>
MeshHierarchy::do_on_old_nodes(const HierarchyCoefficients<double> u,
                               const std::size_t l) const {
  double *const start = u.data;
  return {start, start + ndof_old(l)};
}

RangeSlice<double *>
MeshHierarchy::do_on_new_nodes(const HierarchyCoefficients<double> u,
                               const std::size_t l) const {
  double *const old_start = u.data;
  return {old_start + ndof_old(l), old_start + ndof(l)};
}

std::size_t MeshHierarchy::do_scratch_space_needed() const {
  std::vector<std::size_t> sizes = {scratch_space_needed_for_decomposition(),
                                    scratch_space_needed_for_recomposition()};
  return *std::max_element(sizes.begin(), sizes.end());
}

std::size_t MeshHierarchy::do_scratch_space_needed_for_decomposition() const {
  return L ? 6 * ndof(L - 1) * sizeof(double) : 0;
}

std::size_t MeshHierarchy::do_scratch_space_needed_for_recomposition() const {
  return L ? 6 * ndof(L - 1) * sizeof(double) : 0;
}

double MeshHierarchy::do_measure(const moab::EntityHandle handle,
                                 const std::size_t l) const {
  return meshes.at(l).measure(handle);
}

moab::ErrorCode
MeshHierarchy::do_calculate_correction_from_multilevel_component(
    const HierarchyCoefficients<double> u, const std::size_t l,
    double *const correction, void *const buffer) const {
  const MeshLevel &mesh = meshes.at(l - 1);
  // As of this writing, this is already computed in the functions calling this
  // function. Could reuse it, or maybe split up `buffer`.
  const std::size_t n = ndof(l - 1);

  char *_buffer = static_cast<char *>(buffer);
  // Product of the mass matrix on level `l - 1` and the multilevel component
  // on level `l` (size `n`) (with restriction from level `l` to level `l - 1`
  // built in).
  double *b = reinterpret_cast<double *>(_buffer);
  _buffer += 1 * n * sizeof(*b);
  // Scratch space for the PCG implementation (size `4 * n`).
  double *pcg_buffer = reinterpret_cast<double *>(_buffer);
  _buffer += 4 * n * sizeof(*pcg_buffer);

  // Apply the mass matrix to get the righthand side of our system.
  moab::ErrorCode ecode = apply_mass_matrix_to_multilevel_component(u, l, b);
  MB_CHK_ERR(ecode);
  // Invert the system to obtain `Q_{l - 1}u - Π_{l - 1}Q_{l}u` on
  //`old_nodes(l)`.
  const MassMatrix M(mesh);
  const MassMatrixPreconditioner P(mesh);
  std::fill(correction, correction + n, 0);
  [[maybe_unused]] const pcg::Diagnostics diagnostics =
      pcg::pcg(M, b, P, correction, pcg_buffer);
  assert(diagnostics.converged);
  return moab::MB_SUCCESS;
}

} // namespace mgard
