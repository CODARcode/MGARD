#ifndef TENSORMESHLEVEL_HPP
#define TENSORMESHLEVEL_HPP
//!\file
//!\brief Class for representing a level in a tensor mesh hierarchy.

#include <cstddef>

#include <array>

#include "utilities.hpp"

namespace mgard {

// TODO: Add members relating to cell spacing and create class hierarchy,
// presumably, with equispaced and nonequispaced children.

template <std::size_t N, typename Real>
//! Tensor mesh (either freestanding or part of a hierarchy).
class TensorMeshLevel {
public:
  TensorMeshLevel(const std::array<std::size_t, N> shape);

  //! Report the number of degrees of freedom of the mesh.
  std::size_t ndof() const;

  const std::array<std::size_t, N> shape;
};

//! Equality comparison.
template <std::size_t N, typename Real>
bool operator==(const TensorMeshLevel<N, Real> &a,
                const TensorMeshLevel<N, Real> &b);

//! Inequality comparison.
template <std::size_t N, typename Real>
bool operator!=(const TensorMeshLevel<N, Real> &a,
                const TensorMeshLevel<N, Real> &b);

} // namespace mgard

#include "TensorMeshLevel.tpp"
#endif
