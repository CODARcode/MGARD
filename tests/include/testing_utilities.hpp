#ifndef TESTING_UTILITIES_HPP
#define TESTING_UTILITIES_HPP

#include <cstddef>

#include <array>
#include <experimental/filesystem>
#include <ostream>
#include <random>
#include <string>

#include "moab/Interface.hpp"
static const double APPROX_MARGIN_DEFAULT = 0;

#include "TensorMeshHierarchy.hpp"

std::experimental::filesystem::path mesh_path(const std::string &filename);

std::experimental::filesystem::path output_path(const std::string &filename);

void require_moab_success(const moab::ErrorCode ecode);

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
            const mgard::TensorNode<N, Real> &node);

#include "testing_utilities.tpp"
#endif
