#ifndef TENSORLINEAROPERATOR_HPP
#define TENSORLINEAROPERATOR_HPP
//!\file
//!\brief Base class for representing linear operators \f$R^{N_{1}} \otimes
//! \cdots \otimes R^{N_{d}} \to R^{N_{1}} \otimes \cdots \otimes R^{N_{d}}\f$.

#include <cstddef>

#include <array>
#include <vector>

#include "TensorMeshHierarchy.hpp"

namespace mgard {

//! Linear operator \f$R^{N} \to R^{N}\f$ with respect to some fixed bases.
//!
//! These operators are designed to be tensored together to form a
//! `TensorLinearOperator`.
template <std::size_t N, typename Real> class ConstituentLinearOperator {
public:
  //! Constructor.
  //!
  //! This constructor is provided so that arrays of derived classes may be
  //! formed. A default-constructed operator must be assigned to before being
  //! used.
  ConstituentLinearOperator() = default;

  //! Constructor.
  //!
  //!\param hierarchy Mesh hierarchy on which the domain and range are defined.
  //!\param l Index of the mesh on which the operator is to be applied.
  //!\param dimension Index of the dimension in which the operator is to
  //! be applied.
  ConstituentLinearOperator(const TensorMeshHierarchy<N, Real> &hierarchy,
                            const std::size_t l, const std::size_t dimension);

  //! Return the dimension of the domain and range.
  //!
  //! Note that this is *not* the spatial dimension in which the operator is
  //! applied.
  std::size_t dimension() const;

  //! Apply the operator to an element in place.
  //!
  //!\param [in] multiindex Starting multiindex of the one-dimensional 'spear'
  //! along which the operator is to be applied.
  //!\param [in, out] Element in the domain, to be transformed into an element
  //! in the range.
  void operator()(const std::array<std::size_t, N> multiindex,
                  Real *const v) const;

protected:
  //! Mesh hierarchy on which the domain and range are defined.
  TensorMeshHierarchy<N, Real> const *hierarchy;

  //! Index of the dimension in which the operator is to be applied.
  std::size_t dimension_;

  //! Indices of the 'spear' in the chosen dimension.
  TensorIndexRange indices;

private:
  virtual void
  do_operator_parentheses(const std::array<std::size_t, N> multiindex,
                          Real *const v) const = 0;
};

//! Linear operator with respect to some fixed bases formed by tensoring
//! operators on each factor of the tensor product vector space.
template <std::size_t N, typename Real> class TensorLinearOperator {
public:
  //! Constructor.
  //!
  //!\param hierarchy Mesh hierarchy on which the domain and range are defined.
  //!\param l Index of the mesh on which the operator is to be applied.
  //!\param operators Constituent linear operators for each dimension.
  TensorLinearOperator(
      const TensorMeshHierarchy<N, Real> &hierarchy, const std::size_t l,
      const std::array<ConstituentLinearOperator<N, Real> const *, N>
          operators);

  //! Apply the operator to an element in place.
  //!
  //!\param [in, out] v Element in the domain, to be transformed into an element
  //! in the range.
  void operator()(Real *const v) const;

protected:
  //! Constructor.
  //!
  //! This constructor is provided for derived classes to call when the entries
  //! of `operators` will point to members of the derived class (so to
  //! `ConstituentLinearOperator`s which do not exist at the time that this
  //! constructor is called). Since `operators` is not provided, this
  //! constructor does not check that the operators have the right sizes.
  //!
  //!\param hierarchy Mesh hierarchy on which the domain and range are defined.
  //!\param l Index of the mesh on which the operator is to be applied.
  TensorLinearOperator(const TensorMeshHierarchy<N, Real> &hierarchy,
                       const std::size_t l);

  //! Mesh hierarchy on which the domain and range are defined.
  const TensorMeshHierarchy<N, Real> &hierarchy;

  //! Constituent linear operators.
  //!
  //! This member can't be `const` because derived class constructors will write
  //! to it after the base class constructor has returned.
  std::array<ConstituentLinearOperator<N, Real> const *, N> operators;

  //! Indices of the nodes of the mesh on which the operator is to be applied,
  //! grouped by dimension.
  const std::array<TensorIndexRange, N> multiindex_components;
};

} // namespace mgard

#include "TensorLinearOperator.tpp"
#endif
