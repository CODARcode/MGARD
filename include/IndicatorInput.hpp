#ifndef INDICATORINPUT_HPP
#define INDICATORINPUT_HPP
//!\file
//!\brief Classes to support iteration over a dataset and the associated nodes.

#include <array>
#include <iterator>
#include <optional>
#include <vector>

#include "moab/EntityHandle.hpp"

#include "MeshHierarchy.hpp"
#include "MeshLevel.hpp"
#include "SituatedCoefficientRange.hpp"
#include "data.hpp"
#include "utilities.hpp"

namespace mgard {

//! Element of a dataset and the location of its node in the hierarchy. As of
//! this writing, the purpose of an `IndicatorInput` is to calculate an
//! indicator coefficient to decide how to quantize a multilevel coefficient.
template <typename Real> struct IndicatorInput {
  //! Index of the mesh in the hierarchy.
  const std::size_t l;

  //! Mesh containing the node.
  const MeshLevel &mesh;

  //! Node corresponding to the coefficient.
  const moab::EntityHandle node;

  //! Coefficient at the node.
  const Real coefficient;
};

//! Range of indexed coefficients to be iterated over.
template <typename Real> class IndicatorInputRange {
public:
  //! Constructor.
  //!
  //!\param hierarchy Hierarchy on which the dataset is defined.
  //!\param u Dataset to be iterated over.
  IndicatorInputRange(const MeshHierarchy &hierarchy,
                      const MultilevelCoefficients<Real> u);

  //! Forward declaration.
  class iterator;

  //! Return an interator to the beginning of the range.
  iterator begin() const;

  //! Return an iterator to the end of the range.
  iterator end() const;

  //! Mesh hierarchy.
  const MeshHierarchy &hierarchy;

  //! Dataset defined on the mesh hierarchy.
  const MultilevelCoefficients<Real> u;

  //! Meshes in the hierarchy, along with their indices.
  const Enumeration<std::vector<MeshLevel>::const_iterator> indexed_meshes;

  //! End of the indexed mesh range.
  const Enumeration<std::vector<MeshLevel>::const_iterator>::iterator
      indexed_meshes_end;
};

//! Equality comparison.
template <typename Real>
bool operator==(const IndicatorInputRange<Real> &a,
                const IndicatorInputRange<Real> &b);

//! Inequality comparison.
template <typename Real>
bool operator!=(const IndicatorInputRange<Real> &a,
                const IndicatorInputRange<Real> &b);

template <typename Real>
class IndicatorInputRange<Real>::iterator
    : public std::iterator<std::input_iterator_tag, IndicatorInput<Real>> {
public:
  //! Constructor.
  //!
  //!\param iterable Associated indicator input range.
  //!\param inner_mesh Position in the indexed mesh range.
  //!
  //! The node–coefficient iterator will be initialized to the beginning of the
  //! range associated to the given mesh.
  iterator(const IndicatorInputRange<Real> &iterable,
           const Enumeration<std::vector<MeshLevel>::const_iterator>::iterator
               inner_mesh);

  //! Equality comparison.
  bool operator==(const iterator &other) const;

  //! Inequality comparison.
  bool operator!=(const iterator &other) const;

  // Preincrement.
  iterator &operator++();

  //! Postincrement.
  iterator operator++(int);

  //! Dereference.
  IndicatorInput<Real> operator*() const;

private:
  //! Associated coefficient range.
  const IndicatorInputRange<Real> iterable;

  //! Current position in mesh hierarchy.
  Enumeration<std::vector<MeshLevel>::const_iterator>::iterator inner_mesh;

  using NCIterator = typename SituatedCoefficientRange<Real>::iterator;

  //! Current position in the node–coefficient range and the end of that range.
  std::optional<std::array<NCIterator, 2>> inner_node;

  //! Reset the node–coefficient iterator pair.
  //!
  //! If the end of the indexed mesh range has been reached, the value of the
  //! pair is destroyed. Otherwise, it's set to the beginning and end of the
  //! latest node–coefficient range.
  void reset_nc_iterator_pair();
};

} // namespace mgard

#include "IndicatorInput.tpp"
#endif
