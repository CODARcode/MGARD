#ifndef SITUATEDCOEFFICIENTRANGE_HPP
#define SITUATEDCOEFFICIENTRANGE_HPP
//!\file
//!\brief Situated (that is, accompanied by the corresponding nodes)
//! multilevel coefficients corresponding to the 'new' nodes of a level in a
//!`MeshHierarchy`.

#include <utility>

#include "moab/EntityHandle.hpp"
#include "moab/Range.hpp"

#include "MeshHierarchy.hpp"
#include "utilities.hpp"

namespace mgard {

//! 'New' nodes at a level in a mesh hierarchy and the associated multilevel
//! coefficients.
template <typename Real>
class SituatedCoefficientRange
    : public ZippedRange<moab::Range::const_iterator, Real const *> {
public:
  //! Constructor.
  //!
  //!\param hierarchy Associated mesh hierarchy.
  //!\param u Multilevel coefficients defined on the hierarchy.
  //!\param l Index of the MeshLevel.
  SituatedCoefficientRange(const MeshHierarchy &hierarchy,
                           const MultilevelCoefficients<Real> &u,
                           const std::size_t l);
};

} // namespace mgard

#include "SituatedCoefficientRange.tpp"
#endif
