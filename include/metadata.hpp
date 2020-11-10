#ifndef METADATA_HPP
#define METADATA_HPP
//!\file
//!\brief Metadata to store alongside a compressed dataset.

#include <cstddef>

#include <string>
#include <vector>

#include <yaml-cpp/yaml.h>

#include "subcommand_arguments.hpp"

namespace cli {

//! Metadata about the input dataset.
struct DatasetMetadata {
  //! Constructor.
  //!
  //!\param arguments Arguments passed to the compression subcommand.
  DatasetMetadata(const CompressionArguments &arguments);

  //! Constructor.
  DatasetMetadata() = default;

  //! Constructor.
  //!
  //!\param node YAML mapping containing the metadata.
  DatasetMetadata(const YAML::Node &node);

  //! Name to use when writing to a YAML stream.
  const static std::string YAML_name;

  //! Type of the input dataset.
  std::string datatype;
};

//! Write the dataset metadata to a YAML stream.
YAML::Emitter &operator<<(YAML::Emitter &emitter,
                          const DatasetMetadata &metadata);

//! Metadata about the mesh.
struct MeshMetadata {
  //! Constructor.
  MeshMetadata(const CompressionArguments &arguments);

  //! Constructor.
  MeshMetadata() = default;

  //! Constructor.
  //!
  //!\param node YAML mapping containing the metadata.
  MeshMetadata(const YAML::Node &node);

  //! Name to use when writing to a YAML stream.
  const static std::string YAML_name;

  //! Location of the mesh data.
  std::string location;

  //! Type of the mesh.
  std::string meshtype;

  //! Shape of the mesh.
  std::vector<std::size_t> shape;

  //! Names of the files containing the coordinates of the mesh nodes.
  std::vector<std::string> node_coordinate_files;
};

//! Write the mesh metadata to a YAML stream.
YAML::Emitter &operator<<(YAML::Emitter &emitter, const MeshMetadata &metadata);

//! Metadata about the compression.
struct CompressionMetadata {
  //! Constructor.
  CompressionMetadata(const CompressionArguments &arguments);

  //! Constructor.
  CompressionMetadata() = default;

  //! Constructor.
  //!
  //!\param node YAML mapping containing the metadata.
  CompressionMetadata(const YAML::Node &node);

  //! Name to use when writing to a YAML stream.
  const static std::string YAML_name;

  //! Smoothness parameter determining norm used.
  double s;

  //! Absolute error tolerance.
  double tolerance;
};

//! Write the compression metadata to a YAML stream.
YAML::Emitter &operator<<(YAML::Emitter &emitter,
                          const CompressionMetadata &metadata);

struct Metadata {
  //! Constructor.
  //!
  //!\param arguments Arguments passed to the compression subcommand.
  Metadata(const CompressionArguments &arguments);

  //! Constructor.
  //!
  //!\param node YAML mapping containing the metadata.
  Metadata(const YAML::Node &node);

  //! MGARD version used to compress the data.
  // TODO: Set this.
  std::string version;

  //! Metadata about the input dataset.
  DatasetMetadata dataset_metadata;

  //! Metadata about the mesh.
  MeshMetadata mesh_metadata;

  //! Metadata about the compression.
  CompressionMetadata compression_metadata;
};

//! Write the metadata to a YAML stream.
YAML::Emitter &operator<<(YAML::Emitter &emitter, const Metadata &metadata);

} // namespace cli

#endif
