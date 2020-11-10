#include "subcommand_arguments.hpp"

#include <sstream>

namespace cli {

DataShape &DataShape::operator=(const std::string &value) {
  std::istringstream stream(value);
  shape.clear();
  std::string token;
  while (std::getline(stream, token, 'x')) {
    std::size_t dimension;
    std::istringstream(token) >> dimension;
    shape.push_back(dimension);
  }
  return *this;
}

CompressionArguments::CompressionArguments(
    TCLAP::ValueArg<std::string> &datatype, TCLAP::ValueArg<DataShape> &shape,
    TCLAP::ValueArg<std::string> &input, TCLAP::ValueArg<double> &smoothness,
    TCLAP::ValueArg<double> &tolerance, TCLAP::ValueArg<std::string> &output)
    : datatype(datatype.getValue()), shape(shape.getValue().shape),
      dimension(this->shape.size()), coordinate_filenames(dimension),
      input(input.getValue()), s(smoothness.getValue()),
      tolerance(tolerance.getValue()), output(output.getValue()) {
  for (std::size_t i = 0; i < dimension; ++i) {
    std::stringstream filename;
    filename << "coordinates_" << i << ".dat";
    coordinate_filenames.at(i) = filename.str();
  }
}

DecompressionArguments::DecompressionArguments(
    TCLAP::ValueArg<std::string> &input, TCLAP::ValueArg<std::string> &output)
    : input(input.getValue()), output(output.getValue()) {}

} // namespace cli
