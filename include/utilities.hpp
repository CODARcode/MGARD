#ifndef UTILITIES_HPP
#define UTILITIES_HPP
//!\file
//!\brief Utilities for use in the MGARD implementation.

#include <cstddef>

namespace mgard {

//! Mimic an array for range-based for loops.
template <typename T> struct PseudoArray {
  //! Constructor.
  //!
  //!\param data Pointer to the first element in the array.
  //!\param size Length of the array.
  PseudoArray(T *const data, const std::size_t size);

  //! Constructor.
  //!
  //!\overload
  PseudoArray(T *const data, const int size);

  //! Return an iterator to the beginning of the array.
  T *begin() const;

  //! Return an iterator to the end of the array.
  T *end() const;

  //! Read an entry of the array.
  //!
  //!\param i Index.
  T operator[](const std::size_t i) const;

  //! Pointer to the first element of the array.
  T *const data;

  //! Length of the array.
  const std::size_t size;
};

} // namespace mgard

#include "utilities.tpp"
#endif
