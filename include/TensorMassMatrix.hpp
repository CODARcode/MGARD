#ifndef TENSORMASSMATRIX_HPP
#define TENSORMASSMATRIX_HPP
//!\file
//!\brief Mass matrix for tensor product grids.

#include "TensorLinearOperator.hpp"

namespace mgard {

//! Mass matrix for continuous piecewise linear functions defined on a
//! `TensorMeshLevel` 'spear.'
template <std::size_t N, typename Real>
class ConstituentMassMatrix : public ConstituentLinearOperator<N, Real> {
public:
  //! Constructor.
  //!
  //!\param hierarchy Mesh hierarchy on which the function is defined.
  //!\param l Index of the mesh on which the mass matrix is to be applied.
  //!\param dimension Index of the dimension in which the mass matrix is to
  //! be applied.
  ConstituentMassMatrix(const TensorMeshHierarchy<N, Real> &hierarchy,
                        const std::size_t l, const std::size_t dimension);

private:
  using CLO = ConstituentLinearOperator<N, Real>;

  virtual void
  do_operator_parentheses(const std::array<std::size_t, N> multiindex,
                          Real *const v) const override;
};

} // namespace mgard

#include "TensorMassMatrix.tpp"
#endif