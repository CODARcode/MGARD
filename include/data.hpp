#ifndef DATA_HPP
#define DATA_HPP
//!\file
//!\brief Classes for datasets associated to mesh hierarchies.

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
  explicit HierarchyCoefficients(T *const data);

  //! Conversion.
  template <typename U> operator HierarchyCoefficients<U>() const;

  //! Dataset.
  T *const data;
};

// No equality comparisons here. Don't want to accidentally compare `{Nodal,
// Multilevel}Coefficients`.

// Repeating the declarations rather than using a macro so that Doxygen can see
// the documentation. Maybe there is some workaround.

//! Nodal coefficients for a function defined on a mesh hierarchy.
template <typename T>
class NodalCoefficients : public HierarchyCoefficients<T> {
public:
  //! Constructor.
  //!
  //!\param data Dataset.
  explicit NodalCoefficients(T *const data);

  //! Conversion.
  template <typename U> operator NodalCoefficients<U>() const;
};

//! Equality comparison.
template <typename T>
bool operator==(const NodalCoefficients<T> &a, const NodalCoefficients<T> &b);

//! Inequality comparison.
template <typename T>
bool operator!=(const NodalCoefficients<T> &a, const NodalCoefficients<T> &b);

//! Multilevel coefficients for a function defined on a mesh hierarchy.
template <typename T>
class MultilevelCoefficients : public HierarchyCoefficients<T> {
public:
  //! Constructor.
  //!
  //!\param data Dataset.
  explicit MultilevelCoefficients(T *const data);

  //! Conversion.
  template <typename U> operator MultilevelCoefficients<U>() const;
};

//! Equality comparison.
template <typename T>
bool operator==(const MultilevelCoefficients<T> &a,
                const MultilevelCoefficients<T> &b);

//! Inequality comparison.
template <typename T>
bool operator!=(const MultilevelCoefficients<T> &a,
                const MultilevelCoefficients<T> &b);

} // namespace mgard

#include "data.tpp"
#endif
