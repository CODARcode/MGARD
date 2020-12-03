#include <vector>

#include "TensorMeshHierarchyIteration.hpp"

namespace mgard {

template <std::size_t N, typename Real>
void shuffle(const TensorMeshHierarchy<N, Real> &hierarchy,
             Real const *const src, Real *const dst) {
  std::vector<Real *> writers(hierarchy.L + 1);
  writers.at(0) = dst;
  for (std::size_t l = 1; l <= hierarchy.L; ++l) {
    writers.at(l) = dst + hierarchy.ndof(l - 1);
  }

  Real const *p = src;
  for (const TensorNode<N> node :
       UnshuffledTensorNodeRange<N, Real>(hierarchy, hierarchy.L)) {
    *writers.at(hierarchy.date_of_birth(node.multiindex))++ = *p++;
  }
}

template <std::size_t N, typename Real>
void unshuffle(const TensorMeshHierarchy<N, Real> &hierarchy,
               Real const *const src, Real *const dst) {
  std::vector<Real const *> readers(hierarchy.L + 1);
  readers.at(0) = src;
  for (std::size_t l = 1; l <= hierarchy.L; ++l) {
    readers.at(l) = src + hierarchy.ndof(l - 1);
  }

  Real *q = dst;
  for (const TensorNode<N> node :
       UnshuffledTensorNodeRange<N, Real>(hierarchy, hierarchy.L)) {
    *q++ = *readers.at(hierarchy.date_of_birth(node.multiindex))++;
  }
}

} // namespace mgard
