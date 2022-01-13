#ifndef TESTING_UTILITIES_HPP
#define TESTING_UTILITIES_HPP

#include <cstddef>

#include <array>
#include <ostream>
#include <random>
#include <string>

static const double APPROX_MARGIN_DEFAULT = 0;

#include "TensorMeshHierarchy.hpp"

// TODO: Change these? See <https://github.com/catchorg/Catch2/pull/1499>.

//`T` and `U` should be iterators dereferencing to `double` or similar.
template <typename T, typename U, typename SizeType>
void require_vector_equality(T p, U q, const SizeType N,
                             const double margin = APPROX_MARGIN_DEFAULT);

//`T` and `U` should be `SequenceContainer`s to `double` or similar.
template <typename T, typename U>
void require_vector_equality(const T &t, const U &u,
                             const double margin = APPROX_MARGIN_DEFAULT);

//! Results of a series of trials.
struct TrialTracker {
  //! Constructor.
  TrialTracker();

  //! Record the result of a trial.
  TrialTracker &operator+=(const bool result);

  //! Check whether any trials so far have failed.
  explicit operator bool() const;

  //! Number of successful trials so far.
  std::size_t nsuccesses;

  //! Number of unsuccessful trials so far.
  std::size_t nfailures;

  //! Number of trials (successful and unsuccessful) so far.
  std::size_t ntrials;
};

std::ostream &operator<<(std::ostream &os, const TrialTracker &tracker);

template <std::size_t N, typename Real>
std::array<Real, N>
coordinates(const mgard::TensorMeshHierarchy<N, Real> &hierarchy,
            const mgard::TensorNode<N> &node);

//! Make a copy of a mesh hierarchy with extra 'flat' dimensions.
//!
//!\param hierarchy Input hierarchy to be copied.
//!\param shape Shape of the returned hierarchy.
template <std::size_t N, std::size_t M, typename Real>
mgard::TensorMeshHierarchy<M, Real>
make_flat_hierarchy(const mgard::TensorMeshHierarchy<N, Real> &hierarchy,
                    const std::array<std::size_t, M> shape);

#include "testing_utilities.tpp"
#endif
