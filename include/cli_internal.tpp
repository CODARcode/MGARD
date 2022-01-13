#include <cstddef>

#include <algorithm>
#include <array>
#include <fstream>
#include <iostream>

#include "compress.hpp"

namespace cli {

template <std::size_t N>
int compress_N(const cli::CompressionArguments &arguments) {
  if (arguments.datatype == "float") {
    return compress_N_Real<N, float>(arguments);
  } else if (arguments.datatype == "double") {
    return compress_N_Real<N, double>(arguments);
  } else {
    std::cerr << "unrecognized datatype '" << arguments.datatype << "'"
              << std::endl;
    return 1;
  }
}

template <std::size_t N, typename Real>
int compress_N_Real(const cli::CompressionArguments &arguments) {
  std::array<std::size_t, N> shape;
  std::copy(arguments.shape.begin(), arguments.shape.end(), shape.begin());
  const mgard::TensorMeshHierarchy<N, Real> hierarchy(shape);
  const std::size_t ndof = hierarchy.ndof();

  std::ifstream infile(arguments.input, std::ios_base::binary);
  if (not infile) {
    std::cerr << "failed to open '" << arguments.input << "'" << std::endl;
    return 1;
  }
  infile.seekg(0, std::ios_base::end);
  const std::fstream::pos_type insize = infile.tellg();
  if (not infile) {
    std::cerr << "failed to seek to end of '" << arguments.input << "'"
              << std::endl;
    return 1;
  }
  // Data type size.
  const std::size_t dts = sizeof(Real);
  const std::size_t expected_size = ndof * dts;
  if (insize < 0 || static_cast<std::size_t>(insize) != expected_size) {
    std::cerr << "expected " << expected_size << " bytes (";
    if (N) {
      const std::array<std::size_t, N> &SHAPE = hierarchy.shapes.back();
      std::cerr << SHAPE.at(0);
      for (std::size_t i = 1; i < N; ++i) {
        std::cerr << " × " << SHAPE.at(i);
      }
    } else {
      std::cerr << "0";
    }
    std::cerr << " elements and " << dts << " bytes per element)"
              << " but size of '" << arguments.input << "' is " << insize
              << " bytes" << std::endl;
    return 1;
  }
  Real *const v = new Real[ndof];
  infile.seekg(0, std::ios_base::beg);
  if (not infile) {
    std::cerr << "failed to seek to beginning of '" << arguments.input << "'"
              << std::endl;
    return 1;
  }
  infile.read(reinterpret_cast<char *>(v), insize);
  if (not infile) {
    std::cerr << "failed to read from '" << arguments.input << "'" << std::endl;
    return 1;
  }

  const mgard::CompressedDataset<N, Real> compressed =
      mgard::compress<N, Real>(hierarchy, v, arguments.s, arguments.tolerance);
  delete[] v;

  std::ofstream outfile(arguments.output, std::ios_base::binary);
  if (not outfile) {
    std::cerr << "failed to open '" << arguments.output << "'" << std::endl;
    return 1;
  }
  const std::size_t outsize = compressed.size();
  outfile.write(static_cast<char const *>(compressed.data()), outsize);
  if (not outfile) {
    std::cerr << "failed to write to '" << arguments.output << "'" << std::endl;
    return 1;
  }
  std::cout << "input size (bytes):  " << insize << std::endl;
  std::cout << "output size (bytes): " << outsize << std::endl;
  std::cout << "compression ratio:   " << static_cast<float>(insize) / outsize
            << std::endl;
  return 0;
}

} // namespace cli
