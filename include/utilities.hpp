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

//! Mimic Python's `enumerate` builtin. `T` is expected to be an STL container
//! or something like it.
template <typename T> struct Enumeration {
  //! Constructor.
  //!
  //!\param container Container to be iterated over.
  Enumeration(const T &container);

  // Forward declaration.
  class iterator;

  //! Return an iterator to the beginning of the enumeration.
  iterator begin() const;

  //! Return an iterator to the end of the enumeration.
  iterator end() const;

  //! Underlying container.
  const T &container;
};

//! Equality comparison.
template <typename T>
bool operator==(const Enumeration<T> &a, const Enumeration<T> &b);

//! Inequality comparison.
template <typename T>
bool operator!=(const Enumeration<T> &a, const Enumeration<T> &b);

//! Iterator over an enumeration.
template <typename T>
class Enumeration<T>::iterator
    : public std::iterator<
          std::input_iterator_tag,
          std::pair<typename T::size_type, typename T::value_type>> {
public:
  //! Constructor.
  //!
  //!\param iterable Associated enumeration.
  //!\param index Index in the associated sequence.
  //!\param inner Position in the associated sequence.
  iterator(const Enumeration<T> &iterable, const typename T::size_type index,
           const typename T::const_iterator inner);

  //! Equality comparison.
  bool operator==(const iterator &other) const;

  //! Inequality comparison.
  bool operator!=(const iterator &other) const;

  //! Preincrement.
  iterator &operator++();

  //! Postincrement.
  iterator operator++(int);

  //Would like to return `value_type` from the dereference operator, but it
  //seems like inheriting from a class template prevents `value_type` from being
  //recognized as a type. See http://www.cs.technion.ac.il/users/yechiel/
  //c++-faq/nondependent-name-lookup-members.html> and http://www.open-std.org/
  //jtc1/sc22/wg21/docs/papers/2016/p0174r2.html#2.1>.

  //! Dereference.
  std::pair<typename T::size_type, typename T::value_type> operator*() const;

private:
  //! Associated enumeration.
  const Enumeration<T> &iterable;

  //! Index in the associated sequence.
  typename T::size_type index;

  //! Position in the associated sequence.
  typename T::const_iterator inner;
};

//! Mimic Python's `zip` builtin. `T` and `U` are expected to be STL containers
//! or something like it.
template <typename T, typename U> struct ZippedRange {
  //! Constructor.
  //!
  //!\param container_first First container to be iterated over.
  //!\param container_second Second container to be iterated over.
  ZippedRange(const T &container_first, const U &container_second);

  // Forward declaration.
  class iterator;

  //! Return an iterator to the beginning of the zipped range.
  iterator begin() const;

  //! Return an iterator to the end of the zipped range.
  iterator end() const;

  //!First underlying container.
  const T &container_first;

  //!Second underlying container.
  const U &container_second;
};

//! Equality comparison.
template <typename T, typename U>
bool operator==(const ZippedRange<T, U> &a, const ZippedRange<T, U> &b);

//! Inequality comparison.
template <typename T, typename U>
bool operator!=(const ZippedRange<T, U> &a, const ZippedRange<T, U> &b);

//! Iterator over a zipped range.
template <typename T, typename U>
class ZippedRange<T, U>::iterator
    : public std::iterator<
          std::input_iterator_tag,
          std::pair<typename T::value_type, typename U::value_type>> {
public:
  //! Constructor.
  //!
  //!\param iterable Associated zipper range.
  //!\param inner_first Position in the first associated sequence.
  //!\param inner_second Position in the second associated sequence.
  iterator(
    const ZippedRange<T, U> &iterable,
    const typename T::const_iterator inner_first,
    const typename U::const_iterator inner_second
  );

  //! Equality comparison.
  bool operator==(const iterator &other) const;

  //! Inequality comparison.
  bool operator!=(const iterator &other) const;

  //! Preincrement.
  iterator &operator++();

  //! Postincrement.
  iterator operator++(int);

  //! Dereference.
  std::pair<typename T::value_type, typename U::value_type> operator*() const;

private:
  //! Associated zipped range.
  const ZippedRange<T, U> &iterable;

  //! Position in the first associated sequence
  typename T::const_iterator inner_first;

  //! Position in the second associated sequence
  typename U::const_iterator inner_second;
};

} // namespace mgard

#include "utilities.tpp"
#endif
