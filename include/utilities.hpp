#ifndef UTILITIES_HPP
#define UTILITIES_HPP
//!\file
//!\brief Utilities for use in the MGARD implementation.

#include <cstddef>

#include <iterator>
#include <utility>

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

//! Element of a range along with its index in that range. Replacement for
//! `std::pair<std::size_t, const T &>` because I was having trouble
//! constructing pairs (something having to do with the reference).
template <typename T> struct IndexedElement {
  //! Index of the element.
  const std::size_t index;

  //! Value of the element.
  const T &value;
};

//! Mimic Python's `enumerate` builtin. Iterating over this object yields
//! (objects containing) *references* to the original elements.
template <typename It> struct Enumeration {
  //! Constructor.
  //!
  //!\param begin Iterator to the beginning of the range to be iterated over.
  //!\param end Iterator to the end of the range to be iterated over.
  Enumeration(const It begin, const It end);

  //!\overload
  //!
  //!\param container Container to be iterated over.
  template <typename T> Enumeration(const T &container);

  // Prevent temporaries. See <https://stackoverflow.com/questions/36868442/
  // avoid-exponential-grow-of-const-references-and-rvalue-references-in-
  // constructor> for some candidate approaches. Really, the issue is the
  // invalidation of `container.{begin,end}()`. If those iterators remain valid,
  // there is (as far as I know) nothing wrong with passing in a temporary.
  template <typename T> Enumeration(const T &&container) = delete;

  // Forward declaration.
  class iterator;

  //! Return an iterator to the beginning of the enumeration.
  iterator begin() const;

  //! Return an iterator to the end of the enumeration.
  iterator end() const;

  //! Iterator to the beginning of the range to be iterated over.
  const It begin_;

  //! Iterator to the end of the range to be iterated over.
  const It end_;
};

//! Equality comparison.
template <typename It>
bool operator==(const Enumeration<It> &a, const Enumeration<It> &b);

//! Inequality comparison.
template <typename It>
bool operator!=(const Enumeration<It> &a, const Enumeration<It> &b);

//! Iterator over an enumeration.
template <typename It> class Enumeration<It>::iterator {
public:
  //! Type iterated over by inner iterator.
  using T = typename std::iterator_traits<It>::value_type;
  //! Category of the iterator.
  using iterator_category = std::input_iterator_tag;
  //! Type iterated over.
  using value_type = IndexedElement<T>;
  //! Type for distance between iterators.
  using difference_type = std::ptrdiff_t;
  //! Pointer to `value_type`.
  using pointer = value_type *;
  //! Type returned by the dereference operator.
  using reference = value_type;

  //! Constructor.
  //!
  //!\param iterable Associated enumeration.
  //!\param index Index in the associated sequence.
  //!\param inner Position in the associated sequence.
  iterator(const Enumeration<It> &iterable, const std::size_t index,
           const It inner);

  //! Equality comparison.
  bool operator==(const iterator &other) const;

  //! Inequality comparison.
  bool operator!=(const iterator &other) const;

  //! Preincrement.
  iterator &operator++();

  //! Postincrement.
  iterator operator++(int);

  //! Dereference.
  reference operator*() const;

private:
  //! Associated enumeration.
  const Enumeration<It> &iterable;

  //! Index in the associated sequence.
  std::size_t index;

  //! Position in the associated sequence.
  It inner;
};

//! Mimic Python's `zip` builtin. Iterating over this object yields *copies* of
//! the original elements.
template <typename It, typename Jt> struct ZippedRange {
  //! Constructor.
  //!
  //!\param begin_first Iterator to the beginning of the first range to be
  //! iterated over.
  //!\param end_first Iterator to the end of the first range to be
  //! iterated over.
  //!\param begin_second Iterator to the beginning of the second range to be
  //! iterated over.
  //!\param end_second Iterator to the end of the second range to be
  //! iterated over.
  ZippedRange(const It begin_first, const It end_first, const Jt begin_second,
              const Jt end_second);

  template <typename T, typename U>
  //!\overload
  //!
  //!\param container_first First container to be iterated over.
  //!\param container_second Second container to be iterated over.
  //!
  //! See the note about temporaries in `Enumeration`. If a temporary is passed
  //! in here, *it is the responsibility of the caller to ensure that the
  //! corresponding iterators will remain valid!*
  ZippedRange(const T &container_first, const U &container_second);

  // Forward declaration.
  class iterator;

  //! Return an iterator to the beginning of the zipped range.
  iterator begin() const;

  //! Return an iterator to the end of the zipped range.
  iterator end() const;

  //! Iterator to the beginning of the first range to be iterated over.
  const It begin_first;

  //! Iterator to the end of the first range to be iterated over.
  const It end_first;

  //! Iterator to the beginning of the second range to be iterated over.
  const Jt begin_second;

  //! Iterator to the end of the second range to be iterated over.
  const Jt end_second;
};

//! Equality comparison.
template <typename It, typename Jt>
bool operator==(const ZippedRange<It, Jt> &a, const ZippedRange<It, Jt> &b);

//! Inequality comparison.
template <typename It, typename Jt>
bool operator!=(const ZippedRange<It, Jt> &a, const ZippedRange<It, Jt> &b);

//! Iterator over a zipped range.
template <typename It, typename Jt> class ZippedRange<It, Jt>::iterator {
public:
  //! Type iterated over by first inner iterator.
  using T = typename std::iterator_traits<It>::value_type;
  //! Type iterated over by second inner iterator.
  using U = typename std::iterator_traits<Jt>::value_type;
  //! Category of the iterator.
  using iterator_category = std::input_iterator_tag;
  //! Type iterated over.
  using value_type = std::pair<T, U>;
  //! Type for distance between iterators.
  using difference_type = std::ptrdiff_t;
  //! Pointer to `value_type`.
  using pointer = value_type *;
  //! Type returned by the dereference operator.
  using reference = value_type;

  //! Constructor.
  //!
  //!\param iterable Associated zipped range.
  //!\param inner_first Position in the first associated sequence.
  //!\param inner_second Position in the second associated sequence.
  iterator(const ZippedRange<It, Jt> &iterable, const It inner_first,
           const Jt inner_second);

  //! Copy assignment.
  iterator &operator=(const iterator &other);

  //! Equality comparison.
  bool operator==(const iterator &other) const;

  //! Inequality comparison.
  bool operator!=(const iterator &other) const;

  //! Preincrement.
  iterator &operator++();

  //! Postincrement.
  iterator operator++(int);

  //! Dereference.
  reference operator*() const;

private:
  //! Associated zipped range.
  const ZippedRange<It, Jt> &iterable;

  //! Position in the first associated sequence
  It inner_first;

  //! Position in the second associated sequence
  Jt inner_second;
};

//! Mimic a slice of a range for range-based for loops. Use aggregate
//! initialization to construct.
template <typename It> struct RangeSlice {
public:
  //! Return an iterator to the beginning of the slice.
  It begin() const;

  //! Return an iterator to the end of the slice.
  It end() const;

  //! Beginning of the slice.
  const It begin_;

  //! End of the slice.
  const It end_;
};

//!\deprecated Use `CartesianProduct` instead.
//! Collection of multiindices \f$\vec{\alpha}\f$ satisfying a bound of the form
//! \f$\vec{\beta} \leq \vec{\alpha} < \vec{\gamma}\f$ (elementwise).
template <std::size_t N> struct MultiindexRectangle {
  //! Constructor.
  //!
  //!\param corner Lower bound (elementwise) for the multiindex set, the
  //! 'lower left' vertex of the rectangle.
  //!\param shape Dimensions of the multiindex set.
  MultiindexRectangle(const std::array<std::size_t, N> &corner,
                      const std::array<std::size_t, N> &shape);

  //! Constructor
  //!
  //!\overload
  //!
  //! `corner` defaults to the zero multiindex.
  MultiindexRectangle(const std::array<std::size_t, N> &shape);

  // Forward declaration.
  class iterator;

  //! Return an iterator to the beginning of the indices with the given stride.
  iterator begin(const std::size_t stride) const;

  //! Return an iterator to the end of the indices with the given stride.
  iterator end(const std::size_t stride) const;

  //! Access the multiindices contained in the rectangle.
  RangeSlice<iterator> indices(const std::size_t stride) const;

  //! 'Lower left' vertex of the rectangle.
  const std::array<std::size_t, N> corner;

  //! Dimensions of the rectangle.
  const std::array<std::size_t, N> shape;
};

//! Equality comparison.
template <std::size_t N>
bool operator==(const MultiindexRectangle<N> &a,
                const MultiindexRectangle<N> &b);

//! Inequality comparison.
template <std::size_t N>
bool operator!=(const MultiindexRectangle<N> &a,
                const MultiindexRectangle<N> &b);

//! Iterator over a rectangle of multiindices.
template <std::size_t N> class MultiindexRectangle<N>::iterator {
public:
  //! Category of the iterator.
  using iterator_category = std::input_iterator_tag;
  //! Type iterated over.
  using value_type = std::array<std::size_t, N>;
  //! Type for distance between iterators.
  using difference_type = std::ptrdiff_t;
  //! Pointer to `value_type`.
  using pointer = value_type *;
  //! Type returned by the dereference operator.
  using reference = value_type;

  //! Constructor.
  //!
  //!\param rectangle Associated multiindex set.
  //!\param stride Stride to use in iterating over the multiindex set.
  //!\param indices Starting position in the multiindex set.
  iterator(const MultiindexRectangle &rectangle, const std::size_t stride,
           const std::array<std::size_t, N> &indices);

  //! Equality comparison.
  bool operator==(const iterator &other) const;

  //! Inequality comparison.
  bool operator!=(const iterator &other) const;

  //! Preincrement.
  iterator &operator++();

  //! Postincrement.
  iterator operator++(int);

  //! Dereference;
  reference operator*() const;

  //! Bounding rectangle.
  const MultiindexRectangle &rectangle;

  //! Stride with which to traverse the rectangle.
  const std::size_t stride;

private:
  //! Current multiindex.
  std::array<std::size_t, N> indices;
};

//! Mimic Python's `itertools.product`. Allow iteration over the Cartesian
//! product of a collection of ranges.
//!
//! `typename T::iterator` (roughly – see `T_iterator` in `iterator`) must allow
//! multiple passes over the associated `T` object. It is too much, though, to
//! require it to be a forward iterator.
//!
// We could template on the iterator rather than the container. `factors` could
// then be something like an array of iterator pairs (one iterator to the
// beginning and one to the end of each factor). Then, though, you need to make
// sure that those iterators remain valid. Maybe we could store thex
// `TensorIndexRange`s in the mesh hierarchy or something.
template <typename T, std::size_t N> struct CartesianProduct {
public:
  //! Constructor.
  //!
  //!\param factors Factors of the Cartesian product. None of the factors may be
  //! empty.
  CartesianProduct(const std::array<T, N> factors);

  // Forward declaration.
  class iterator;

  //! Return an iterator to the beginning of the product range.
  iterator begin() const;

  //! Return an iterator to the beginning of the product range.
  iterator end() const;

  //! Factors of the Cartesian product.
  const std::array<T, N> factors;
};

//! Equality comparison.
template <typename T, std::size_t N>
bool operator==(const CartesianProduct<T, N> &a,
                const CartesianProduct<T, N> &b);

//! Inequality comparison.
template <typename T, std::size_t N>
bool operator!=(const CartesianProduct<T, N> &a,
                const CartesianProduct<T, N> &b);

//! Iterator over a Cartesian product.
template <typename T, std::size_t N> class CartesianProduct<T, N>::iterator {
public:
  //! Iterator over `T`.
  // When `T` is `TensorIndexRange`, we just want `TensorIndexRange::iterator`.
  // But when `T` is `std::vector<int>` (as of this writing, only in testing),
  // we need `std::vector<int>::const_iterator`.
  using T_iterator = decltype(
      std::declval<typename std::array<T, N>::const_reference>().begin());

  //! Category of the iterator.
  using iterator_category = std::input_iterator_tag;
  //! Type iterated over.
  using value_type =
      std::array<typename std::iterator_traits<T_iterator>::value_type, N>;
  //! Type for distance between iterators.
  using difference_type = std::ptrdiff_t;
  //! Pointer to `value_type`.
  using pointer = value_type *;
  //! Type returned by the dereference operator.
  using reference = value_type;

  //! Constructor.
  //!
  //!\param iterable Associated Cartesian product.
  //!\param inner Position in the Cartesian product.
  iterator(const CartesianProduct &iterable,
           const std::array<T_iterator, N> inner);

  //! Equality comparison.
  bool operator==(const iterator &other) const;

  //! Inequality comparison.
  bool operator!=(const iterator &other) const;

  //! Preincrement.
  iterator &operator++();

  //! Postincrement.
  iterator operator++(int);

  //! Dereference.
  reference operator*() const;

  //! Associated Cartesian product.
  //!
  //! A pointer rather than a reference so we can assign.
  CartesianProduct const *iterable;

  //! Return the iterator to the left in a given dimension.
  //!
  //! If this iterator is at the beginning in that dimension, this iterator will
  //! be returned.
  //!
  //! This member calls `T_iterator::operator--`.
  //!
  //!\param i Index of the dimension.
  iterator predecessor(const std::size_t i) const;

  //! Return the node to the right in a given dimension.
  //!
  //! If this iterator is immediately before the end in that dimension, this
  //! iterator will be returned. Do not call this member if this iterator is at
  //! the end of `iterable`.
  //!
  //! This member calls `T_iterator::operator--`.
  //!
  //!\param i Index of the dimension.
  iterator successor(const std::size_t i) const;

private:
  //! Position in the Cartesian product.
  std::array<T_iterator, N> inner;
};

//! Check that a dimension index is in bounds.
//!
//!\param dimension Dimension index.
template <std::size_t N>
void check_dimension_index_bounds(const std::size_t dimension);

} // namespace mgard

#include "utilities.tpp"
#endif
