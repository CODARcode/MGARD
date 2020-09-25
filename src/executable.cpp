#include <algorithm>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <archive.h>
#include <archive_entry.h>

#include <tclap/CmdLine.h>
#include <yaml-cpp/yaml.h>

#include "mgard_api.h"

struct DataShape {
  std::vector<std::size_t> shape;

  DataShape &operator=(const std::string &value) {
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
};

namespace TCLAP {

template <> struct ArgTraits<DataShape> { typedef StringLike ValueCategory; };

} // namespace TCLAP

struct CompressionArguments {
  CompressionArguments(TCLAP::ValueArg<std::string> &datatype,
                       TCLAP::ValueArg<DataShape> &shape,
                       TCLAP::ValueArg<std::string> &input,
                       TCLAP::ValueArg<double> &smoothness,
                       TCLAP::ValueArg<double> &tolerance,
                       TCLAP::ValueArg<std::string> &output)
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

  std::string datatype;

  std::vector<std::size_t> shape;

  std::size_t dimension;

  std::vector<std::string> coordinate_filenames;

  std::string input;

  double s;

  double tolerance;

  std::string output;
};

YAML::Emitter &operator<<(YAML::Emitter &emitter,
                          const CompressionArguments &arguments) {
  emitter << YAML::BeginMap;
  emitter << YAML::Key << "datatype" << YAML::Value << arguments.datatype;
  emitter << YAML::Key << "shape" << YAML::Value << arguments.shape;
  emitter << YAML::Key << "node coordinate files" << YAML::Value
          << arguments.coordinate_filenames;
  emitter << YAML::Key << "s" << YAML::Value << arguments.s;
  emitter << YAML::Key << "tolerance" << YAML::Value << arguments.tolerance;
  emitter << YAML::EndMap;
  return emitter;
}

void write_archive_entry(archive *const a, const std::string entryname,
                         void const *const data, const std::size_t size) {
  archive_entry *const entry = archive_entry_new();
  archive_entry_copy_pathname(entry, entryname.c_str());
  archive_entry_set_filetype(entry, AE_IFREG);
  archive_entry_set_perm(entry, S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH);
  archive_entry_set_size(entry, size);

  if (archive_write_header(a, entry) != ARCHIVE_OK) {
    throw std::runtime_error("error writing archive entry header");
  }

  if (archive_write_data(a, data, size) < 0) {
    throw std::runtime_error("error writing archive entry data");
  }

  archive_entry_free(entry);
}

template <std::size_t N, typename Real>
void write_archive(const CompressionArguments &arguments,
                   const mgard::TensorMeshHierarchy<N, Real> &hierarchy,
                   const mgard::CompressedDataset<N, Real> &compressed) {
  archive *const a = archive_write_new();
  if (a == nullptr) {
    throw std::runtime_error("error creating new archive");
  }
  if (archive_write_set_format_gnutar(a) != ARCHIVE_OK) {
    throw std::runtime_error("error setting archive format to GNU tar");
  }
  if (archive_write_open_filename(a, arguments.output.c_str()) != ARCHIVE_OK) {
    throw std::runtime_error("error opening the archive file");
  }

  YAML::Emitter emitter;
  emitter << arguments;
  write_archive_entry(a, "metadata.yaml", emitter.c_str(), emitter.size());

  write_archive_entry(a, "coefficients.dat", compressed.data(),
                      compressed.size());

  for (std::size_t i = 0; i < N; ++i) {
    write_archive_entry(a, arguments.coordinate_filenames.at(i),
                        hierarchy.coordinates.at(i).data(),
                        arguments.shape.at(i) * sizeof(Real));
  }

  if (archive_write_free(a) != ARCHIVE_OK) {
    throw std::runtime_error("error freeing archive");
  }
}

template <std::size_t N, typename Real>
int read_compress_write(const CompressionArguments &arguments) {
  std::array<std::size_t, N> shape;
  std::copy(arguments.shape.begin(), arguments.shape.end(), shape.begin());
  const mgard::TensorMeshHierarchy<N, Real> hierarchy(shape);
  const std::size_t ndof = hierarchy.ndof();
  std::fstream inputfile(arguments.input,
                         std::ios_base::binary | std::ios_base::in);
  {
    inputfile.seekg(0, std::ios_base::end);
    const std::fstream::pos_type read = inputfile.tellg();
    const std::size_t expected = ndof * sizeof(Real);
    if (read != expected) {
      std::cerr << "expected " << expected << " bytes (";
      if (N > 1) {
        for (std::size_t i = 0; i + 1 < N; ++i) {
          std::cout << arguments.shape.at(i) << " × ";
        }
        std::cout << arguments.shape.at(N - 1) << " = ";
      }
      std::cout << ndof << " elements and " << sizeof(Real)
                << " bytes per element) but read " << read << " bytes"
                << std::endl;
      return 1;
    }
    inputfile.seekg(0, std::ios_base::beg);
  }
  Real *const v = static_cast<Real *>(std::malloc(ndof * sizeof(*v)));
  inputfile.read(reinterpret_cast<char *>(v), ndof * sizeof(*v));
  inputfile.close();

  const mgard::CompressedDataset<N, Real> compressed =
      mgard::compress(hierarchy, v, static_cast<Real>(arguments.s),
                      static_cast<Real>(arguments.tolerance));
  std::free(v);

  std::cout << "size of compressed dataset: " << compressed.size() << " bytes"
            << std::endl;
  std::cout << "compression ratio: "
            << static_cast<Real>(ndof * sizeof(Real)) / compressed.size()
            << std::endl;

  try {
    write_archive(arguments, hierarchy, compressed);
  } catch (const std::runtime_error &e) {
    std::cerr << "error in writing archive: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}

int compress(const int argc, char const *const *const argv) {
  try {
    TCLAP::CmdLine cmd("The compressor subcommand of MGARD.");

    std::vector<std::string> datatype_allowed = {"float", "double"};
    TCLAP::ValuesConstraint<std::string> datatype_constraint(datatype_allowed);
    TCLAP::ValueArg<std::string> datatype(
        "", "datatype", "floating precision format of the data", true, "",
        &datatype_constraint);
    cmd.add(datatype);

    TCLAP::ValueArg<DataShape> shape(
        "", "shape", "shape of the data", true, {},
        "the shape of the data, given as an 'x'-delimited list of the "
        "dimensions of the array");
    cmd.add(shape);

    // Reading these in as `double`s for now.
    TCLAP::ValueArg<double> smoothness(
        "", "smoothness", "smoothness parameter", true, 0,
        "determines norm in which compression error is measured");
    cmd.add(smoothness);

    TCLAP::ValueArg<double> tolerance("", "tolerance",
                                      "absolute error tolerance", true, 0,
                                      "absolute error tolerance");
    cmd.add(tolerance);

    TCLAP::ValueArg<std::string> input(
        "", "input", "input file", true, "",
        "file containing the dataset to be compressed");
    cmd.add(input);

    TCLAP::ValueArg<std::string> output(
        "", "output", "output file", true, "",
        "file in which to store the compressed dataset");
    cmd.add(output);

    cmd.parse(argc, argv);

    const CompressionArguments arguments(datatype, shape, input, smoothness,
                                         tolerance, output);

    std::cout << "summary of arguments passed:" << std::endl;
    std::cout << "  datatype: " << arguments.datatype << std::endl;
    std::cout << "  shape: {";
    {
      for (std::size_t i = 0; i + 1 < arguments.dimension; ++i) {
        std::cout << arguments.shape.at(i) << ", ";
      }
      if (arguments.dimension) {
        std::cout << arguments.shape.at(arguments.dimension - 1);
      }
    }
    std::cout << "}" << std::endl;
    std::cout << "  smoothness parameter: " << arguments.s << std::endl;
    std::cout << "  tolerance: " << arguments.tolerance << std::endl;
    std::cout << "  input: " << arguments.input << std::endl;
    std::cout << "  output: " << arguments.output << std::endl;

    if (arguments.datatype == "float") {
      switch (arguments.dimension) {
      case 1:
        return read_compress_write<1, float>(arguments);
        break;
      case 2:
        return read_compress_write<2, float>(arguments);
        break;
      case 3:
        return read_compress_write<3, float>(arguments);
        break;
      default:
        std::cerr << "unsupported dimension " << arguments.dimension
                  << std::endl;
        return 1;
      }
    } else if (arguments.datatype == "double") {
      switch (arguments.dimension) {
      case 1:
        return read_compress_write<1, double>(arguments);
        break;
      case 2:
        return read_compress_write<2, double>(arguments);
        break;
      case 3:
        return read_compress_write<3, double>(arguments);
        break;
      default:
        std::cerr << "unsupported dimension " << arguments.dimension
                  << std::endl;
        return 1;
      }
    } else {
      std::cerr << "unsupported datatype " << arguments.datatype << std::endl;
      return 1;
    }
  } catch (TCLAP::ArgException &e) {
    std::cerr << "error for argument " << e.argId() << ": " << e.error()
              << std::endl;
    return 1;
  }
  return 0;
}

int decompress(const int argc, char const *const *const argv) { return 0; }

int main(const int argc, char const *const *const argv) {
  try {
    TCLAP::CmdLine cmd("MGARD is a compressor for scientific data.");

    std::vector<std::string> subcommand_allowed = {"compress", "decompress"};
    TCLAP::ValuesConstraint<std::string> subcommand_constraint(
        subcommand_allowed);
    TCLAP::UnlabeledValueArg<std::string> subcommand(
        "subcommand", "whether to compress or decompress the input", true, "",
        &subcommand_constraint);
    cmd.add(subcommand);

    TCLAP::UnlabeledMultiArg<std::string> subarguments(
        "subarguments", "All remaining arguments are passed to the subcommand.",
        false, "arguments to pass to the subcommand");
    cmd.add(subarguments);

    cmd.parse(argc, argv);

    const std::string parsed_subcommand = subcommand.getValue();
    if (parsed_subcommand == "compress") {
      return compress(argc - 1, argv + 1);
    } else if (parsed_subcommand == "decompress") {
      return decompress(argc - 1, argv + 1);
    } else {
      return 1;
    }
  } catch (TCLAP::ArgException &e) {
    std::cerr << "error for argument " << e.argId() << ": " << e.error()
              << std::endl;
    return 1;
  }

  return 0;
}
