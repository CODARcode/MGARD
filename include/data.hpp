#ifndef DATA_HPP
#define DATA_HPP
//!\file
//!\brief Classes for datasets associated to mesh hierarchies.

#include "moab/EntityHandle.hpp"

namespace mgard {

//! Base class for datasets associated to mesh hierarchies.
template <typename T> class HierarchyCoefficients {
public:
  //! Constructor.
  //!
  //!\param data Dataset.
  //!
  //! I believe that this constructor is implicitly called when, for
  //! example, a mesh hierarchy calls `hierarchy.decompose(double * u)`.
  //! In that case, the multilevel coefficients can then be read from the
  //! returned `MultilevelCoefficients` or from `u`.
  HierarchyCoefficients(T *const data);

  //! Conversion.
  template <typename U> operator HierarchyCoefficients<U>() const;

  //! Dataset.
  T *const data;
};

// Repeating the declarations rather than using a macro so that Doxygen can see
// the documentation. Maybe there is some workaround.

//! Nodal coefficients for a function defined on a mesh hierarchy.
template <typename T>
class NodalCoefficients : public HierarchyCoefficients<T> {
public:
  //! Constructor.
  //!
  //!\param data Dataset.
  NodalCoefficients(T *const data);

  //! Conversion.
  template <typename U> operator NodalCoefficients<U>() const;
};

//! Multilevel coefficients for a function defined on a mesh hierarchy.
template <typename T>
class MultilevelCoefficients : public HierarchyCoefficients<T> {
public:
  //! Constructor.
  //!
  //!\param data Dataset.
  MultilevelCoefficients(T *const data);

  //! Conversion.
  template <typename U> operator MultilevelCoefficients<U>() const;
};

//! Indicator coefficients for a function defined on a mesh hierarchy.
template <typename T>
class IndicatorCoefficients : public HierarchyCoefficients<T> {
public:
  //! Constructor.
  //!
  //!\param data Dataset.
  IndicatorCoefficients(T *const data);

  //! Conversion.
  template <typename U> operator IndicatorCoefficients<U>() const;
};

} // namespace mgard

#include "data.tpp"
#endif
