#include <cmath>

#include <algorithm>

#include "TensorNorms.hpp"
#include "shuffle.hpp"

namespace mgard {

template <std::size_t N, typename Real>
template <typename Functional>
TensorQuantityOfInterest<N, Real>::TensorQuantityOfInterest(
    const TensorMeshHierarchy<N, Real> &hierarchy, const Functional &functional)
    : hierarchy(hierarchy) {
  const std::size_t ndof = hierarchy.ndof();
  // Riesz representative of the functional.
  Real *const representative = new Real[ndof];
  // Product of the mass matrix and the Riesz representative.
  Real *const f = new Real[ndof];
  {
    // Product of the mass matrix and the Riesz representative (unshuffled
    // order). Reusing `representative`'s memory.
    Real *const f_unshuffled = representative;
    {
      // Test function to apply the functional to. Reusing `f`'s memory.
      Real *const phi = f;
      std::fill(phi, phi + ndof, 0);
      for (std::size_t i = 0; i < ndof; ++i) {
        phi[i] = 1;
        f_unshuffled[i] = functional(hierarchy, phi);
        phi[i] = 0;
      }
    }
    shuffle(hierarchy, f_unshuffled, f);
  }
  {
    std::copy(f, f + ndof, representative);
    TensorMassMatrixInverse<N, Real> m_inv(hierarchy, hierarchy.L);
    m_inv(representative);
  }
  component_square_norms =
      orthogonal_component_square_norms(hierarchy, representative, f);
  delete[] f;
  delete[] representative;
}

template <std::size_t N, typename Real>
Real TensorQuantityOfInterest<N, Real>::norm(const Real s) const {
  Real square_norm = 0;
  for (std::size_t l = 0; l <= hierarchy.L; ++l) {
    square_norm += std::exp2(2 * -s * l) * component_square_norms.at(l);
  }
  return std::sqrt(square_norm);
}

} // namespace mgard
