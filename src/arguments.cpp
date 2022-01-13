#include "arguments.hpp"

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
    const TCLAP::ValueArg<std::string> &datatype,
    const TCLAP::ValueArg<DataShape> &shape,
    const TCLAP::ValueArg<SmoothnessParameter<double>> &smoothness,
    const TCLAP::ValueArg<double> &tolerance,
    const TCLAP::ValueArg<std::string> &input,
    const TCLAP::ValueArg<std::string> &output)
    : datatype(datatype.getValue()), shape(shape.getValue().shape),
      dimension(this->shape.size()), s(smoothness.getValue()),
      tolerance(tolerance.getValue()), input(input.getValue()),
      output(output.getValue()) {}

DecompressionArguments::DecompressionArguments(
    const TCLAP::ValueArg<std::string> &input,
    const TCLAP::ValueArg<std::string> &output)
    : input(input.getValue()), output(output.getValue()) {}

} // namespace cli
