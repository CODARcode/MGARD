#ifndef MULTIDIMENSIONALARRAY_HPP
#define MULTIDIMENSIONALARRAY_HPP
//!\file
//!\brief Multidimensional arrays with basic indexing support.

#include <cstddef>

#include <array>

namespace mgard {

//! Multidimensional array with basic indexing support.
template <typename T, std::size_t N> class MultidimensionalArray {
public:
  static_assert(N, "dimension must be nonzero");

  //! Constructor.
  //!
  //!\param data Underlying memory buffer.
  //!\param dimensions Dimensions of the array.
  //!\param stride Distance between neighboring elements.
  MultidimensionalArray(T *const data,
                        const std::array<std::size_t, N> dimensions,
                        const std::size_t stride = 1);

  //! Compute the size of the array.
  std::size_t size() const;

  //! Access an element of the array by multiindex.
  //!
  //!\param multiindex Multiindex of the element
  //!
  //!\return Element at the multiindex.
  T &at(const std::array<std::size_t, N> multiindex) const;

  //! Underlying memory buffer.
  T *const data;

  //! Dimensions of the array.
  const std::array<std::size_t, N> dimensions;

  //! Stride between neighboring elements.
  const std::size_t stride;

protected:
  //! Check that a multiindex is within bounds.
  void
  check_multiindex_bounds(const std::array<std::size_t, N> multiindex) const;
};

} // namespace mgard

#include "MultidimensionalArray.tpp"

#endif
