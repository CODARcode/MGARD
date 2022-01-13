#ifndef UNSTRUCTURED_TESTING_UTILITIES_HPP
#define UNSTRUCTURED_TESTING_UTILITIES_HPP

#include <string>

#include "moab/Interface.hpp"

void require_moab_success(const moab::ErrorCode ecode);

std::string mesh_path(const std::string &filename);

std::string output_path(const std::string &filename);

#endif
