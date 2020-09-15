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

  TensorIndexRange coarse_indices;

  virtual void
  do_operator_parentheses(const std::array<std::size_t, N> multiindex,
                          Real *const v) const override;
};

//! Restriction for tensor products of continuous piecewise linear functions
//! defined on a Cartesian product mesh hierarchy.
template <std::size_t N, typename Real>
class TensorRestriction : public TensorLinearOperator<N, Real> {
public:
  //! Constructor.
  //!
  //!\param hierarchy Mesh hierarchy on which the functions are defined.
  //!\param l Index of the mesh on which the restriction is to be applied. This
  //! is the index of the fine mesh (corresponding to the domain).
  TensorRestriction(const TensorMeshHierarchy<N, Real> &hierarchy,
                    const std::size_t l);

private:
  using TLO = TensorLinearOperator<N, Real>;

  //! Constituent restrictions for each dimension.
  const std::array<ConstituentRestriction<N, Real>, N> restrictions;
};

} // namespace mgard

#include "TensorRestriction.tpp"
#endif
