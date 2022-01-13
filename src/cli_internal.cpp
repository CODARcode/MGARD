#include "cli_internal.hpp"

#include <fstream>

#include "compress.hpp"
#include "utilities.hpp"

namespace cli {

int compress(const cli::CompressionArguments &arguments) {
  return compress(arguments, arguments.dimension);
}

int decompress(const cli::DecompressionArguments &arguments) {
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
  mgard::MemoryBuffer<unsigned char> inbuffer(insize);
  infile.seekg(0, std::ios_base::beg);
  if (not infile) {
    std::cerr << "failed to seek to beginning of '" << arguments.input << "'"
              << std::endl;
    return 1;
  }
  infile.read(reinterpret_cast<char *>(inbuffer.data.get()), inbuffer.size);
  if (not infile) {
    std::cerr << "failed to read from '" << arguments.input << "'" << std::endl;
    return 1;
  }

  const mgard::MemoryBuffer<const unsigned char> outbuffer =
      mgard::decompress(inbuffer.data.get(), inbuffer.size);

  std::ofstream outfile(arguments.output, std::ios_base::binary);
  if (not outfile) {
    std::cerr << "failed to open '" << arguments.output << "'" << std::endl;
    return 1;
  }
  outfile.write(reinterpret_cast<char const *>(outbuffer.data.get()),
                outbuffer.size);
  if (not outfile) {
    std::cerr << "failed to write to '" << arguments.output << "'" << std::endl;
    return 1;
  }
  return 0;
}

} // namespace cli
