#include "estimators.hpp"

#include <cassert>

#include <limits>
#include <stdexcept>

#include "blas.hpp"

#include "MassMatrix.hpp"

namespace mgard {

// We might be able to obtain better bounds by using some information about
// the hierarchy. If I recall correctly, our proofs depend on (at least) the
// refinement strategy used. For now, we just check its dimension.
RatioBounds s_square_estimator_bounds(const MeshHierarchy &hierarchy) {
  const std::size_t d = hierarchy.meshes.front().topological_dimension;
  if (d == 2) {
    return {.realism = 0.1, .reliability = 1};
  } else if (d == 3) {
    return {.realism = 1.f / 24, .reliability = 1};
  } else {
    throw std::domain_error("unsupported topological dimension");
  }
}

static double s_square_estimator(const MultilevelCoefficients<double> u,
                                 const MeshHierarchy &hierarchy,
                                 const float s) {
  std::vector<double> squares_for_estimate(hierarchy.L + 1);
  // TODO: allow passing in of memory.
  std::vector<double> scratch(hierarchy.ndof_new(hierarchy.L));
  double *const rhs = scratch.data();
  for (std::size_t l = 0; l <= hierarchy.L; ++l) {
    const MeshLevel &mesh = hierarchy.meshes.at(l);
    const std::size_t n = hierarchy.ndof_new(l);
    // Originally, `hierarchy.new_nodes` returned a `moab::Range`, which could
    // be passed straight to the constructor below. In order to avoid problems
    // with iterators pointing to temporaries that have gone out of scope,
    // `hierarchy.new_nodes` now returns a pair of iterators to the appropriate
    // `moab::Range` in `hierarchy`. Rather than rewriting `SubsetMassMatrix` to
    // deal with iterators, we will just manually construct the `moab::Range`.
    const RangeSlice<moab::Range::const_iterator> iterators =
        hierarchy.new_nodes(l);
    const moab::Range::const_iterator new_nodes_begin = iterators.begin();
    moab::Range::const_iterator new_nodes_end = iterators.end();
    // Check that it's safe to decrement `new_nodes_end`.
    assert(new_nodes_begin != new_nodes_end);
    const moab::Range new_nodes(*new_nodes_begin, *--new_nodes_end);
    ContiguousSubsetMassMatrix M(mesh, new_nodes);
    // Nodal values of multilevel component on level `l`.
    double const *const mc = hierarchy.on_new_nodes(u, l).begin();
    M(mc, rhs);
    squares_for_estimate.at(l) = blas::dotu(n, mc, rhs);
  }

  // Could have accumulated this as we went.
  double square_estimate = 0;
  for (std::size_t l = 0; l <= hierarchy.L; ++l) {
    // Code repeated here from `norms.cpp`.
    square_estimate += std::pow(2, 2 * s * l) * squares_for_estimate.at(l);
  }
  return square_estimate;
}

double estimator(const MultilevelCoefficients<double> u,
                 const MeshHierarchy &hierarchy, const float s) {
  if (s == std::numeric_limits<double>::infinity()) {
    throw std::domain_error(
        "pointwise estimator not implemented for unstructured grids");
  } else {
    return std::sqrt(s_square_estimator(u, hierarchy, s));
  }
}

} // namespace mgard
