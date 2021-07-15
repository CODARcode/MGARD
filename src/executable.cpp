#include <algorithm>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <archive.h>
#include <archive_entry.h>

#include <yaml-cpp/yaml.h>

#include <tclap/ValueArg.h>
#include <tclap/ValuesConstraint.h>

#include "MGARDConfig.hpp"
#include "compress.hpp"

#include "arguments.hpp"
#include "cmdline.hpp"
#include "metadata.hpp"
#include "output.hpp"

const std::string METADATA_ENTRYNAME = "metadata.yaml";
const std::string QUANTIZED_COEFFICIENTS_ENTRYNAME = "coefficients.dat";

void write_archive_entry(archive *const a, const std::string &entryname,
                         void const *const data, const std::size_t size) {
  struct archive_entry *const entry = archive_entry_new();
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
void write_archive(const cli::CompressionArguments &arguments,
                   const mgard::TensorMeshHierarchy<N, Real> &hierarchy,
                   const mgard::CompressedDataset<N, Real> &compressed) {
  struct archive *const a = archive_write_new();
  if (a == nullptr) {
    throw std::runtime_error("error creating new archive");
  }
  if (archive_write_set_format_gnutar(a) != ARCHIVE_OK) {
    throw std::runtime_error("error setting archive format to GNU tar");
  }
  if (archive_write_open_filename(a, arguments.output.c_str()) != ARCHIVE_OK) {
    throw std::runtime_error("error opening the archive file");
  }

  const cli::Metadata metadata(arguments);

  YAML::Emitter emitter;
  emitter << metadata;
  write_archive_entry(a, METADATA_ENTRYNAME, emitter.c_str(), emitter.size());

  write_archive_entry(a, QUANTIZED_COEFFICIENTS_ENTRYNAME, compressed.data(),
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

template <typename T>
std::ostream &operator<<(std::ostream &stream, const std::vector<T> &values) {
  const std::size_t N = values.size();
  stream << "{";
  for (std::size_t i = 0; i + 1 < N; ++i) {
    stream << values.at(i) << ", ";
  }
  if (N) {
    stream << values.at(N - 1);
  }
  stream << "}";
  return stream;
}

std::unordered_map<std::string, std::vector<unsigned char>>
read_archive(const cli::DecompressionArguments &arguments) {
  struct archive *const a = archive_read_new();
  if (a == nullptr) {
    throw std::runtime_error("error creating new archive");
  }
  if (archive_read_support_filter_all(a) != ARCHIVE_OK) {
    throw std::runtime_error("error enabling reading filters");
  }
  if (archive_read_support_format_all(a) != ARCHIVE_OK) {
    throw std::runtime_error("error enabling reading formats");
  }

  struct stat input_stat;
  char const *const input = arguments.input.c_str();
  if (stat(input, &input_stat)) {
    throw std::runtime_error("error reading block size for input file");
  }

  if (archive_read_open_filename(a, input, input_stat.st_blksize) !=
      ARCHIVE_OK) {
    throw std::runtime_error("error opening the archive file");
  }

  std::unordered_map<std::string, std::vector<unsigned char>> entries;
  {
    struct archive_entry *entry;
    while (archive_read_next_header(a, &entry) == ARCHIVE_OK) {
      char const *const p = archive_entry_pathname(entry);
      const int64_t entry_size = archive_entry_size(entry);
      std::vector<unsigned char> buffer(entry_size);
      {
        const la_ssize_t data_read =
            archive_read_data(a, buffer.data(), entry_size);
        if (data_read != entry_size) {
          throw std::runtime_error("error reading in full entry");
        }
      }
      entries.insert({p, buffer});
    }

    if (archive_read_close(a) != ARCHIVE_OK) {
      throw std::runtime_error("error closing the archive file");
    }

    if (archive_read_free(a) != ARCHIVE_OK) {
      throw std::runtime_error("error freeing archive reading resources");
    }
  }
  return entries;
}

template <std::size_t N, typename Real>
int read_compress_write(const cli::CompressionArguments &arguments) {
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
    if (read < 0 || static_cast<std::size_t>(read) != expected) {
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

int compress(TCLAP::ValueArg<std::string> &datatype,
             TCLAP::ValueArg<cli::DataShape> &shape,
             TCLAP::ValueArg<cli::SmoothnessParameter<double>> &smoothness,
             TCLAP::ValueArg<double> &tolerance,
             TCLAP::ValueArg<std::string> &input,
             TCLAP::ValueArg<std::string> &output) {
  const cli::CompressionArguments arguments(datatype, shape, input, smoothness,
                                            tolerance, output);
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
      std::cerr << "unsupported dimension " << arguments.dimension << std::endl;
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
      std::cerr << "unsupported dimension " << arguments.dimension << std::endl;
      return 1;
    }
  } else {
    std::cerr << "unsupported datatype " << arguments.datatype << std::endl;
    return 1;
  }
  return 0;
}

cli::Metadata read_metadata_from_entry(
    std::unordered_map<std::string, std::vector<unsigned char>> &entries) {
  try {
    std::vector<unsigned char> &entry = entries.at(METADATA_ENTRYNAME);
    // Make sure we have a terminating null byte. I expect, but haven't
    // checked, that we'll get an error later if `entry` is empty.
    if (!entry.empty()) {
      // Avoiding assuming zero is represented the same way by `char` and
      // `unsigned char`. We want the final byte to be zero when interpreted
      // as a `char`.
      unsigned char unsigned_null_byte;
      *reinterpret_cast<char *>(&unsigned_null_byte) = 0;
      if (entry.back() != unsigned_null_byte) {
        entry.push_back(unsigned_null_byte);
      }
    }
    return cli::Metadata(
        YAML::Load(reinterpret_cast<char const *>(entry.data())));
  } catch (const std::out_of_range &e) {
    throw std::runtime_error("no metadata found in archive (expected '" +
                             METADATA_ENTRYNAME + "')");
  }
}

template <std::size_t N, typename Real>
mgard::TensorMeshHierarchy<N, Real> read_hierarchy_from_entries(
    const cli::MeshMetadata &metadata,
    const std::unordered_map<std::string, std::vector<unsigned char>>
        &entries) {
  if (metadata.location != "internal" ||
      metadata.meshtype != "Cartesian product") {
    throw std::runtime_error(
        "only internal Cartesian product meshes currently supported");
  }

  std::array<std::size_t, N> shape;
  {
    const std::vector<std::size_t> &shape_ = metadata.shape;
    std::copy(shape_.begin(), shape_.end(), shape.begin());
  }

  std::array<std::vector<Real>, N> coordinates;
  {
    const std::vector<std::string> &filenames = metadata.node_coordinate_files;
    for (std::size_t i = 0; i < N; ++i) {
      const std::vector<unsigned char> &entry = entries.at(filenames.at(i));
      unsigned char const *const src = entry.data();
      const std::size_t M = entry.size();

      const std::size_t expected_coords_size = shape.at(i);
      if (M != expected_coords_size * sizeof(Real)) {
        throw std::runtime_error("coordinate array improperly sized");
      }

      std::vector<Real> &coords = coordinates.at(i);
      coords.resize(expected_coords_size);
      unsigned char *const dst =
          reinterpret_cast<unsigned char *>(coords.data());

      std::copy(src, src + M, dst);
    }
  }

  return mgard::TensorMeshHierarchy<N, Real>(shape, coordinates);
}

template <std::size_t N, typename Real>
int decompress_write(
    const cli::DecompressionArguments &arguments, const cli::Metadata &metadata,
    const std::unordered_map<std::string, std::vector<unsigned char>>
        &entries) {
  const mgard::TensorMeshHierarchy<N, Real> hierarchy =
      read_hierarchy_from_entries<N, Real>(metadata.mesh_metadata, entries);
  const std::size_t ndof = hierarchy.ndof();

  const std::vector<unsigned char> &buffer_ =
      entries.at(QUANTIZED_COEFFICIENTS_ENTRYNAME);
  const std::size_t buffer_size = buffer_.size();
  // This buffer is freed in the `CompressedDataset` destructor.
  unsigned char *const buffer = new unsigned char[buffer_size];
  {
    unsigned char const *const src = buffer_.data();
    std::copy(src, src + buffer_size, buffer);
  }

  const cli::CompressionMetadata &compression_metadata =
      metadata.compression_metadata;
  const double s = compression_metadata.s;
  const double tolerance = compression_metadata.tolerance;

  const mgard::CompressedDataset<N, Real> compressed(hierarchy, s, tolerance,
                                                     buffer, buffer_size);
  const mgard::DecompressedDataset<N, Real> decompressed =
      mgard::decompress(compressed);

  std::fstream outputfile(arguments.output,
                          std::ios_base::binary | std::ios_base::out);
  outputfile.write(reinterpret_cast<char const *>(decompressed.data()),
                   ndof * sizeof(Real));
  return 0;
}

int decompress(TCLAP::ValueArg<std::string> &input,
               TCLAP::ValueArg<std::string> &output) {
  const cli::DecompressionArguments arguments(input, output);

  std::unordered_map<std::string, std::vector<unsigned char>> entries =
      read_archive(arguments);

  const cli::Metadata metadata = read_metadata_from_entry(entries);
  const std::string datatype = metadata.dataset_metadata.datatype;
  const std::size_t dimension = metadata.mesh_metadata.shape.size();

  if (datatype == "float") {
    switch (dimension) {
    case 1:
      return decompress_write<1, float>(arguments, metadata, entries);
      break;
    case 2:
      return decompress_write<2, float>(arguments, metadata, entries);
      break;
    case 3:
      return decompress_write<3, float>(arguments, metadata, entries);
      break;
    default:
      std::cerr << "unsupported dimension " << dimension << std::endl;
      return 1;
    }
  } else if (datatype == "double") {
    switch (dimension) {
    case 1:
      return decompress_write<1, double>(arguments, metadata, entries);
      break;
    case 2:
      return decompress_write<2, double>(arguments, metadata, entries);
      break;
    case 3:
      return decompress_write<3, double>(arguments, metadata, entries);
      break;
    default:
      std::cerr << "unsupported dimension " << dimension << std::endl;
      return 1;
    }
  } else {
    std::cerr << "unsupported datatype " << datatype << std::endl;
    return 1;
  }
  return 0;
}

int main(const int argc, char const *const *const argv) {
  SubCmdLine subcompress(MGARD_VERSION_STR, "Compress a dataset using MGARD.");

  TCLAP::ValueArg<std::string> subcompress_output(
      "", "output", "file in which to store the compressed dataset", true, "",
      "filename");
  subcompress.add(subcompress_output);

  TCLAP::ValueArg<std::string> subcompress_input(
      "", "input", "file containing the dataset to be compressed", true, "",
      "filename");
  subcompress.add(subcompress_input);

  TCLAP::ValueArg<double> tolerance("", "tolerance", "absolute error tolerance",
                                    true, 0, "τ");
  subcompress.add(tolerance);

  // Reading these in as `double`s for now.
  TCLAP::ValueArg<cli::SmoothnessParameter<double>> smoothness(
      "", "smoothness", "index of norm in which compression error is measured",
      true, 0, "s");
  subcompress.add(smoothness);

  TCLAP::ValueArg<cli::DataShape> shape(
      "", "shape",
      "shape of the data, given as an 'x'-delimited list of dimensions", true,
      {}, "list");
  subcompress.add(shape);

  std::vector<std::string> datatype_allowed = {"float", "double"};
  TCLAP::ValuesConstraint<std::string> datatype_constraint(datatype_allowed);
  TCLAP::ValueArg<std::string> datatype(
      "", "datatype", "floating-point format of the data", true,
      "floating point type", &datatype_constraint);
  subcompress.add(datatype);

  subcompress.addHelpArgument();

  SubCmdLine subdecompress(MGARD_VERSION_STR,
                           "Decompress a dataset compressed using MGARD.");

  TCLAP::ValueArg<std::string> subdecompress_output(
      "", "output", "file in which to store the decompressed dataset", true, "",
      "filename");
  subdecompress.add(subdecompress_output);

  TCLAP::ValueArg<std::string> subdecompress_input(
      "", "input", "file containing the compressed dataset", true, "",
      "filename");
  subdecompress.add(subdecompress_input);

  subdecompress.addHelpArgument();

  std::map<std::string, SubCmdLine *> subcommands = {
      {"compress", &subcompress}, {"decompress", &subdecompress}};
  SuperCmdLine cmd(MGARD_VERSION_STR,
                   "MGARD is a compressor for scientific data.", subcommands);

  cmd.parse(argc, argv);
  SubCmdLine *const p = cmd.getSubcommand();
  if (p == &subcompress) {
    return compress(datatype, shape, smoothness, tolerance, subcompress_input,
                    subcompress_output);
  } else if (p == &subdecompress) {
    return decompress(subdecompress_input, subdecompress_output);
  } else {
    return 1;
  }

  return 0;
}
