#ifndef TENSORPROLONGATION_HPP
#define TENSORPROLONGATION_HPP
//!\file
//!\brief Prolongation–addition for tensor product grids.

#include <vector>

#include "TensorLinearOperator.hpp"

namespace mgard {

//! Prolongation–addition (interpolate the values on the 'old' nodes to the
//! 'new' nodes and add (not overwrite)) for continuous piecewise linear
//! functions defined on a `TensorMeshLevel` 'spear.'
template <std::size_t N, typename Real>
class ConstituentProlongationAddition
    : public ConstituentLinearOperator<N, Real> {
public:
  //! Constructor.
  //!
  //! This constructor is provided so that arrays of this class may be formed. A
  //! default-constructed instance must be assigned to before being used.
  ConstituentProlongationAddition() = default;

  //! Constructor.
  //!
  //!\param hierarchy Mesh hierarchy on which the functions are defined.
  //!\param l Index of the mesh on which the prolongation–addition is to be
  //! applied.
  //!\param dimension Index of the dimension in which the prolongation–addition
  //! is to be applied.
  ConstituentProlongationAddition(const TensorMeshHierarchy<N, Real> &hierarchy,
                                  const std::size_t l,
                                  const std::size_t dimension);

private:
  using CLO = ConstituentLinearOperator<N, Real>;

  std::vector<std::size_t> coarse_indices;

  virtual void
  do_operator_parentheses(const std::array<std::size_t, N> multiindex,
                          Real *const v) const override;
};

} // namespace mgard

#include "TensorProlongation.tpp"
#endif
