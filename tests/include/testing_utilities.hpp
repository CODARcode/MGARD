#ifndef TESTING_UTILITIES_HPP
#define TESTING_UTILITIES_HPP

#include <cstddef>

#include <experimental/filesystem>
#include <string>

#include "moab/Interface.hpp"

static const double APPROX_MARGIN_DEFAULT = 0;

std::experimental::filesystem::path mesh_path(const std::string &filename);

std::experimental::filesystem::path output_path(const std::string &filename);

void require_moab_success(const moab::ErrorCode ecode);

//`T` and `U` should be iterators dereferencing to `double` or similar.
template <typename T, typename U, typename SizeType>
void require_vector_equality(T p, U q, const SizeType N,
                             const double margin = APPROX_MARGIN_DEFAULT);

//`T` and `U` should be `SequenceContainer`s to `double` or similar.
template <typename T, typename U>
void require_vector_equality(const T &t, const U &u,
                             const double margin = APPROX_MARGIN_DEFAULT);

#include "testing_utilities.tpp"
#endif
