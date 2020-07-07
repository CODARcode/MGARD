#ifndef TENSORRESTRICTION_HPP
#define TENSORRESTRICTION_HPP
//!\file
//!\brief Restriction for tensor product grids.

#include "TensorLinearOperator.hpp"

namespace mgard {

//! Restriction for continuous piecewise linear functions defined on a
//! `TensorMeshLevel` 'spear.'
template <std::size_t N, typename Real>
class ConstituentRestriction : public ConstituentLinearOperator<N, Real> {
public:
  //! Constructor.
  //!
  //! This constructor is provided so that arrays of this class may be formed. A
  //! default-constructed instance must be assigned to before being used.
  ConstituentRestriction() = default;

  //! Constructor.
  //!
  //!\param hierarchy Mesh hierarchy on which the functions are defined.
  //!\param l Index of the mesh on which the restriction is to be applied. This
  //! is the index of the fine mesh (corresponding to the domain).
  //!\param dimension Index of the dimension in which the restriction is to be
  //! applied.
  ConstituentRestriction(const TensorMeshHierarchy<N, Real> &hierarchy,
                         const std::size_t l, const std::size_t dimension);

private:
  using CLO = ConstituentLinearOperator<N, Real>;

  std::vector<std::size_t> coarse_indices;

  virtual void
  do_operator_parentheses(const std::array<std::size_t, N> multiindex,
                          Real *const v) const override;
};

} // namespace mgard

#include "TensorRestriction.tpp"
#endif
