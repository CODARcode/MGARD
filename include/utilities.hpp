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

//! Mimic Python's `enumerate` builtin.
template <typename It> struct Enumeration {
  //! Constructor.
  //!
  //!\param begin Iterator to the beginning of the range to be iterated over.
  //!\param end Iterator to the end of the range to be iterated over.
  Enumeration(const It begin, const It end);

  //!\override
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
template <typename It>
class Enumeration<It>::iterator
    : public std::iterator<std::input_iterator_tag,
                           std::pair<std::size_t, typename std::iterator_traits<
                                                      It>::value_type>> {
public:
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

  // Would like to return `value_type` from the dereference operator, but it
  // seems like inheriting from a class template prevents `value_type` from
  // being recognized as a type. See http://www.cs.technion.ac.il/users/yechiel/
  // c++-faq/nondependent-name-lookup-members.html> and http://www.open-std.org/
  // jtc1/sc22/wg21/docs/papers/2016/p0174r2.html#2.1>.

  //! Dereference.
  std::pair<std::size_t, typename std::iterator_traits<It>::value_type>
  operator*() const;

private:
  //! Associated enumeration.
  const Enumeration<It> &iterable;

  //! Index in the associated sequence.
  std::size_t index;

  //! Position in the associated sequence.
  It inner;
};

//! Mimic Python's `zip` builtin.
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
  //!\override
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
template <typename It, typename Jt>
class ZippedRange<It, Jt>::iterator
    : public std::iterator<
          std::input_iterator_tag,
          std::pair<typename std::iterator_traits<It>::value_type,
                    typename std::iterator_traits<Jt>::value_type>> {
public:
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
  std::pair<typename std::iterator_traits<It>::value_type,
            typename std::iterator_traits<Jt>::value_type>
  operator*() const;

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

} // namespace mgard

#include "utilities.tpp"
#endif
