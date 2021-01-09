#include "metadata.hpp"

#include <stdexcept>
#include <unordered_map>

namespace cli {

namespace {

void check_all_specified(
    const std::unordered_map<std::string, bool> &specified) {
  for (const std::pair<std::string, bool> pair : specified) {
    if (!pair.second) {
      throw std::invalid_argument(pair.first + " not specified");
    }
  }
}

} // namespace

const std::string DatasetMetadata::YAML_name = "dataset";

DatasetMetadata::DatasetMetadata(const CompressionArguments &arguments)
    : datatype(arguments.datatype) {}

DatasetMetadata::DatasetMetadata(const YAML::Node &node) {
  if (!node.IsMap()) {
    throw std::invalid_argument("expected YAML mapping");
  }
  std::unordered_map<std::string, bool> specified = {{"datatype", false}};
  for (const YAML::detail::iterator_value value : node) {
    const std::string key = value.first.as<std::string>();
    if (specified[key]) {
      throw std::invalid_argument(key + " multiply specified");
    }
    if (key == "datatype") {
      datatype = value.second.as<std::string>();
    } else {
      throw std::invalid_argument("unexpected metadata key '" + key + "'");
    }
    specified[key] = true;
  }
  check_all_specified(specified);
}

YAML::Emitter &operator<<(YAML::Emitter &emitter,
                          const DatasetMetadata &metadata) {
  emitter << YAML::BeginMap;
  emitter << YAML::Key << "datatype" << YAML::Value << metadata.datatype;
  emitter << YAML::EndMap;
  return emitter;
}

const std::string MeshMetadata::YAML_name = "mesh";

MeshMetadata::MeshMetadata(const CompressionArguments &arguments)
    : location("internal"), meshtype("Cartesian product"),
      shape(arguments.shape),
      node_coordinate_files(arguments.coordinate_filenames) {}

MeshMetadata::MeshMetadata(const YAML::Node &node) {
  if (!node.IsMap()) {
    throw std::invalid_argument("expected YAML mapping");
  }
  std::unordered_map<std::string, bool> specified = {
      {"location", false},
      {"meshtype", false},
      {"shape", false},
      {"node coordinate files", false},
  };
  for (const YAML::detail::iterator_value value : node) {
    const std::string key = value.first.as<std::string>();
    if (specified[key]) {
      throw std::invalid_argument(key + " multiply specified");
    }
    if (key == "location") {
      location = value.second.as<std::string>();
    } else if (key == "meshtype") {
      meshtype = value.second.as<std::string>();
    } else if (key == "shape") {
      shape = value.second.as<std::vector<std::size_t>>();
    } else if (key == "node coordinate files") {
      node_coordinate_files = value.second.as<std::vector<std::string>>();
    } else {
      throw std::invalid_argument("unexpected metadata key '" + key + "'");
    }
    specified[key] = true;
  }
  check_all_specified(specified);
}

YAML::Emitter &operator<<(YAML::Emitter &emitter,
                          const MeshMetadata &metadata) {
  emitter << YAML::BeginMap;
  emitter << YAML::Key << "location" << YAML::Value << metadata.location;
  emitter << YAML::Key << "meshtype" << YAML::Value << metadata.meshtype;
  emitter << YAML::Key << "shape" << YAML::Value << metadata.shape;
  emitter << YAML::Key << "node coordinate files" << YAML::Value
          << metadata.node_coordinate_files;
  emitter << YAML::EndMap;
  return emitter;
}

const std::string CompressionMetadata::YAML_name = "compression";

CompressionMetadata::CompressionMetadata(const CompressionArguments &arguments)
    : s(arguments.s), tolerance(arguments.tolerance) {}

CompressionMetadata::CompressionMetadata(const YAML::Node &node) {
  if (!node.IsMap()) {
    throw std::invalid_argument("expected YAML mapping");
  }
  std::unordered_map<std::string, bool> specified = {{"s", false},
                                                     {"tolerance", false}};
  for (const YAML::detail::iterator_value value : node) {
    const std::string key = value.first.as<std::string>();
    if (specified[key]) {
      throw std::invalid_argument(key + " multiply specified");
    }
    if (key == "s") {
      s = value.second.as<double>();
    } else if (key == "tolerance") {
      tolerance = value.second.as<double>();
    } else {
      throw std::invalid_argument("unexpected metadata key '" + key + "'");
    }
    specified[key] = true;
  }
  check_all_specified(specified);
}

YAML::Emitter &operator<<(YAML::Emitter &emitter,
                          const CompressionMetadata &metadata) {
  emitter << YAML::BeginMap;
  emitter << YAML::Key << "s" << YAML::Value << metadata.s;
  emitter << YAML::Key << "tolerance" << YAML::Value << metadata.tolerance;
  emitter << YAML::EndMap;
  return emitter;
}

Metadata::Metadata(const YAML::Node &node) {
  if (!node.IsMap()) {
    throw std::invalid_argument("expected YAML mapping");
  }
  std::unordered_map<std::string, bool> specified = {
      {DatasetMetadata::YAML_name, false},
      {MeshMetadata::YAML_name, false},
      {CompressionMetadata::YAML_name, false}};
  for (const YAML::detail::iterator_value value : node) {
    const std::string key = value.first.as<std::string>();
    if (specified[key]) {
      throw std::invalid_argument(key + " multiply specified");
    }
    if (key == DatasetMetadata::YAML_name) {
      dataset_metadata = DatasetMetadata(value.second);
    } else if (key == MeshMetadata::YAML_name) {
      mesh_metadata = MeshMetadata(value.second);
    } else if (key == CompressionMetadata::YAML_name) {
      compression_metadata = CompressionMetadata(value.second);
    } else {
      throw std::invalid_argument("unexpected metadata key '" + key + "'");
    }
    specified[key] = true;
  }
  check_all_specified(specified);
}

Metadata::Metadata(const CompressionArguments &arguments)
    : dataset_metadata(arguments), mesh_metadata(arguments),
      compression_metadata(arguments) {}

YAML::Emitter &operator<<(YAML::Emitter &emitter, const Metadata &metadata) {
  emitter << YAML::BeginMap;
  emitter << YAML::Key << metadata.dataset_metadata.YAML_name << YAML::Value
          << metadata.dataset_metadata;
  emitter << YAML::Key << metadata.mesh_metadata.YAML_name << YAML::Value
          << metadata.mesh_metadata;
  emitter << YAML::Key << metadata.compression_metadata.YAML_name << YAML::Value
          << metadata.compression_metadata;
  emitter << YAML::EndMap;
  return emitter;
}

} // namespace cli
