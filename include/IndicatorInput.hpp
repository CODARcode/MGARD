#ifndef INDICATORINPUT_HPP
#define INDICATORINPUT_HPP
//!\file
//!\brief Classes to support iteration over a dataset and the associated nodes.

#include <cstddef>

#include <array>
#include <iterator>
#include <optional>
#include <vector>

#include "moab/EntityHandle.hpp"
#include "moab/Range.hpp"

#include "MeshHierarchy.hpp"
#include "MeshLevel.hpp"
#include "utilities.hpp"

namespace mgard {

//! Auxiliary mesh data for an element of a dataset. As of this writing, the
//! purpose of an `IndicatorInput` is to calculate an indicator coefficient
//! factor to decide how to quantize a multilevel coefficient.
struct IndicatorInput {
  //! Index of the mesh in the hierarchy.
  const std::size_t l;

  //! Mesh containing the node.
  const MeshLevel &mesh;

  //! Node corresponding to the coefficient.
  const moab::EntityHandle node;
};

//! Range of auxiliary mesh data to be iterated over.
class IndicatorInputRange {
public:
  //! Constructor.
  //!
  //!\param hierarchy Mesh hierarchy.
  explicit IndicatorInputRange(const MeshHierarchy &hierarchy);

  // Forward declaration.
  class iterator;

  //! Return an interator to the beginning of the range.
  iterator begin() const;

  //! Return an iterator to the end of the range.
  iterator end() const;

  //! Mesh hierarchy.
  const MeshHierarchy &hierarchy;

  //! Meshes in the hierarchy, along with their indices.
  const Enumeration<std::vector<MeshLevel>::const_iterator> indexed_meshes;

  //! End of the indexed mesh range.
  const Enumeration<std::vector<MeshLevel>::const_iterator>::iterator
      indexed_meshes_end;
};

//! Equality comparison.
bool operator==(const IndicatorInputRange &a, const IndicatorInputRange &b);

//! Inequality comparison.
bool operator!=(const IndicatorInputRange &a, const IndicatorInputRange &b);

//! Iterator over a range of auxiliary mesh data.
class IndicatorInputRange::iterator {
public:
  //! Category of the iterator.
  using iterator_category = std::input_iterator_tag;
  //! Type iterated over.
  using value_type = IndicatorInput;
  //! Type for distance between iterators.
  using difference_type = std::ptrdiff_t;
  //! Pointer to `value_type`.
  using pointer = value_type *;
  //! Type returned by the dereference operator.
  using reference = value_type;

  //! Constructor.
  //!
  //!\param iterable Associated indicator input range.
  //!\param inner_mesh Position in the indexed mesh range.
  //!
  //! The node range iterator will be initialized to the beginning of the range
  //! associated to the given mesh.
  iterator(const IndicatorInputRange &iterable,
           const Enumeration<std::vector<MeshLevel>::const_iterator>::iterator
               inner_mesh);

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
  //! Associated auxiliary mesh data range.
  const IndicatorInputRange iterable;

  //! Current position in mesh hierarchy.
  Enumeration<std::vector<MeshLevel>::const_iterator>::iterator inner_mesh;

  //! Current position in the node range and the end of that range.
  std::optional<std::array<moab::Range::const_iterator, 2>> inner_node;

  //! Reset the node iterator pair.
  //!
  //! If the end of the indexed mesh range has been reached, the value of the
  //! pair is destroyed. Otherwise, it's set to the beginning and end of the
  //! latest node range.
  void reset_node_iterator_pair();
};

} // namespace mgard

#endif
