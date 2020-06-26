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
  //! This constructor is provided so that arrays of this class may be formed. A
  //! default-constructed instance must be assigned to before being used.
  ConstituentMassMatrix() = default;

  //! Constructor.
  //!
  //!\param hierarchy Mesh hierarchy on which the functions are defined.
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

//! Mass matrix for tensor products of continuous piecewise linear functions
//! defined on a Cartesian product mesh hierarchy.
template <std::size_t N, typename Real>
class TensorMassMatrix : public TensorLinearOperator<N, Real> {
public:
  //! Constructor.
  //!
  //!\param hierarchy Mesh hierarchy on which the functions are defined.
  //!\param l Index of the mesh on which the mass matrix is to be applied.
  TensorMassMatrix(const TensorMeshHierarchy<N, Real> &hierarchy,
                   const std::size_t l);

private:
  using TLO = TensorLinearOperator<N, Real>;

  //! Constituent mass matrices for each dimension.
  const std::array<ConstituentMassMatrix<N, Real>, N> mass_matrices;
};

//! Inverse of mass matrix for continuous piecewise linear functions defined on
//! a `TensorMeshLevel` 'spear.'
template <std::size_t N, typename Real>
class ConstituentMassMatrixInverse : public ConstituentLinearOperator<N, Real> {
public:
  //! Constructor.
  //!
  //! This constructor is provided so that arrays of this class may be formed. A
  //! default-constructed instance must be assigned to before being used.
  ConstituentMassMatrixInverse() = default;

  //! Constructor.
  //!
  //!\param hierarchy Mesh hierarchy on which the functions are defined.
  //!\param l Index of the mesh on which the mass matrix is to be inverted.
  //!\param dimension Index of the dimension in which the mass matrix is to
  //! be inverted.
  //!\param buffer Buffer of size the dimension of the function space.
  ConstituentMassMatrixInverse(const TensorMeshHierarchy<N, Real> &hierarchy,
                               const std::size_t l, const std::size_t dimension,
                               Real *const buffer);

private:
  using CLO = ConstituentLinearOperator<N, Real>;

  //! Buffer to store divisors for the Thomas algorithm.
  Real *divisors;

  virtual void
  do_operator_parentheses(const std::array<std::size_t, N> multiindex,
                          Real *const v) const override;
};

} // namespace mgard

#include "TensorMassMatrix.tpp"
#endif
