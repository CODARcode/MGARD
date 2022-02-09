/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-X: MultiGrid Adaptive Reduction of Data Portable across GPUs and CPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: December 1, 2021
 */

#include <chrono>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

#include "compress_x.hpp"
#include "mgard-x/Hierarchy.h"
#include "mgard-x/HighLevelAPI.h"
#include "mgard-x/Metadata.hpp"
#include "mgard-x/RuntimeX/RuntimeXPublic.h"

namespace mgard_x {

template <typename DeviceType>
void compress(DIM D, data_type dtype, std::vector<SIZE> shape, double tol,
              double s, enum error_bound_type mode, const void *original_data,
              void *&compressed_data, size_t &compressed_size, Config config,
              bool output_pre_allocated) {
  if (dtype == data_type::Float) {
    if (D == 1) {
      compress<1, float, DeviceType>(shape, tol, s, mode, original_data,
                                     compressed_data, compressed_size, config,
                                     output_pre_allocated);
    } else if (D == 2) {
      compress<2, float, DeviceType>(shape, tol, s, mode, original_data,
                                     compressed_data, compressed_size, config,
                                     output_pre_allocated);
    } else if (D == 3) {
      compress<3, float, DeviceType>(shape, tol, s, mode, original_data,
                                     compressed_data, compressed_size, config,
                                     output_pre_allocated);
    } else if (D == 4) {
      compress<4, float, DeviceType>(shape, tol, s, mode, original_data,
                                     compressed_data, compressed_size, config,
                                     output_pre_allocated);
    } else if (D == 5) {
      compress<5, float, DeviceType>(shape, tol, s, mode, original_data,
                                     compressed_data, compressed_size, config,
                                     output_pre_allocated);
    } else {
      std::cout << log::log_err
                << "do not support higher than five dimentions!\n";
      exit(-1);
    }
  } else if (dtype == data_type::Double) {
    if (D == 1) {
      compress<1, double, DeviceType>(shape, tol, s, mode, original_data,
                                      compressed_data, compressed_size, config,
                                      output_pre_allocated);
    } else if (D == 2) {
      compress<2, double, DeviceType>(shape, tol, s, mode, original_data,
                                      compressed_data, compressed_size, config,
                                      output_pre_allocated);
    } else if (D == 3) {
      compress<3, double, DeviceType>(shape, tol, s, mode, original_data,
                                      compressed_data, compressed_size, config,
                                      output_pre_allocated);
    } else if (D == 4) {
      compress<4, double, DeviceType>(shape, tol, s, mode, original_data,
                                      compressed_data, compressed_size, config,
                                      output_pre_allocated);
    } else if (D == 5) {
      compress<5, double, DeviceType>(shape, tol, s, mode, original_data,
                                      compressed_data, compressed_size, config,
                                      output_pre_allocated);
    } else {
      std::cout << log::log_err
                << "do not support higher than five dimentions!\n";
      exit(-1);
    }
  } else {
    std::cout << log::log_err
              << "do not support types other than double and float!\n";
    exit(-1);
  }
}

template <typename DeviceType>
void compress(DIM D, data_type dtype, std::vector<SIZE> shape, double tol,
              double s, enum error_bound_type mode, const void *original_data,
              void *&compressed_data, size_t &compressed_size,
              bool output_pre_allocated) {

  Config config;
  compress<DeviceType>(D, dtype, shape, tol, s, mode, original_data,
                       compressed_data, compressed_size, config,
                       output_pre_allocated);
}

template <typename DeviceType>
void compress(DIM D, data_type dtype, std::vector<SIZE> shape, double tol,
              double s, enum error_bound_type mode, const void *original_data,
              void *&compressed_data, size_t &compressed_size,
              std::vector<const Byte *> coords, Config config,
              bool output_pre_allocated) {

  if (dtype == data_type::Float) {
    std::vector<float *> float_coords;
    for (auto &coord : coords)
      float_coords.push_back((float *)coord);
    if (D == 1) {
      compress<1, float, DeviceType>(shape, tol, s, mode, original_data,
                                     compressed_data, compressed_size, config,
                                     float_coords, output_pre_allocated);
    } else if (D == 2) {
      compress<2, float, DeviceType>(shape, tol, s, mode, original_data,
                                     compressed_data, compressed_size, config,
                                     float_coords, output_pre_allocated);
    } else if (D == 3) {
      compress<3, float, DeviceType>(shape, tol, s, mode, original_data,
                                     compressed_data, compressed_size, config,
                                     float_coords, output_pre_allocated);
    } else if (D == 4) {
      compress<4, float, DeviceType>(shape, tol, s, mode, original_data,
                                     compressed_data, compressed_size, config,
                                     float_coords, output_pre_allocated);
    } else if (D == 5) {
      compress<5, float, DeviceType>(shape, tol, s, mode, original_data,
                                     compressed_data, compressed_size, config,
                                     float_coords, output_pre_allocated);
    } else {
      std::cout << log::log_err
                << "do not support higher than five dimentions!\n";
      exit(-1);
    }
  } else if (dtype == data_type::Double) {
    std::vector<double *> double_coords;
    for (auto &coord : coords)
      double_coords.push_back((double *)coord);
    if (D == 1) {
      compress<1, double, DeviceType>(shape, tol, s, mode, original_data,
                                      compressed_data, compressed_size, config,
                                      double_coords, output_pre_allocated);
    } else if (D == 2) {
      compress<2, double, DeviceType>(shape, tol, s, mode, original_data,
                                      compressed_data, compressed_size, config,
                                      double_coords, output_pre_allocated);
    } else if (D == 3) {
      compress<3, double, DeviceType>(shape, tol, s, mode, original_data,
                                      compressed_data, compressed_size, config,
                                      double_coords, output_pre_allocated);
    } else if (D == 4) {
      compress<4, double, DeviceType>(shape, tol, s, mode, original_data,
                                      compressed_data, compressed_size, config,
                                      double_coords, output_pre_allocated);
    } else if (D == 5) {
      compress<5, double, DeviceType>(shape, tol, s, mode, original_data,
                                      compressed_data, compressed_size, config,
                                      double_coords, output_pre_allocated);
    } else {
      std::cout << log::log_err
                << "do not support higher than five dimentions!\n";
      exit(-1);
    }
  } else {
    std::cout << log::log_err
              << "do not support types other than double and float!\n";
    exit(-1);
  }
}

template <typename DeviceType>
void compress(DIM D, data_type dtype, std::vector<SIZE> shape, double tol,
              double s, enum error_bound_type mode, const void *original_data,
              void *&compressed_data, size_t &compressed_size,
              std::vector<const Byte *> coords, bool output_pre_allocated) {
  Config config;
  compress<DeviceType>(D, dtype, shape, tol, s, mode, original_data,
                       compressed_data, compressed_size, coords, config,
                       output_pre_allocated);
}

template <typename DeviceType>
void decompress(const void *compressed_data, size_t compressed_size,
                void *&decompressed_data, Config config,
                bool output_pre_allocated) {

  std::vector<SIZE> shape = infer_shape(compressed_data, compressed_size);
  data_type dtype = infer_data_type(compressed_data, compressed_size);
  data_structure_type dstype =
      infer_data_structure(compressed_data, compressed_size);

  if (dtype == data_type::Float) {
    if (dstype == data_structure_type::Cartesian_Grid_Uniform) {
      if (shape.size() == 1) {
        decompress<1, float, DeviceType>(shape, compressed_data,
                                         compressed_size, decompressed_data,
                                         config, output_pre_allocated);
      } else if (shape.size() == 2) {
        decompress<2, float, DeviceType>(shape, compressed_data,
                                         compressed_size, decompressed_data,
                                         config, output_pre_allocated);
      } else if (shape.size() == 3) {
        decompress<3, float, DeviceType>(shape, compressed_data,
                                         compressed_size, decompressed_data,
                                         config, output_pre_allocated);
      } else if (shape.size() == 4) {
        decompress<4, float, DeviceType>(shape, compressed_data,
                                         compressed_size, decompressed_data,
                                         config, output_pre_allocated);
      } else if (shape.size() == 5) {
        decompress<5, float, DeviceType>(shape, compressed_data,
                                         compressed_size, decompressed_data,
                                         config, output_pre_allocated);
      } else {
        std::cout << log::log_err
                  << "do not support higher than five dimentions!\n";
        exit(-1);
      }
    } else if (dstype == data_structure_type::Cartesian_Grid_Non_Uniform) {

      std::vector<float *> coords =
          infer_coords<float>(compressed_data, compressed_size);

      if (shape.size() == 1) {
        decompress<1, float, DeviceType>(shape, compressed_data,
                                         compressed_size, decompressed_data,
                                         coords, config, output_pre_allocated);
      } else if (shape.size() == 2) {
        decompress<2, float, DeviceType>(shape, compressed_data,
                                         compressed_size, decompressed_data,
                                         coords, config, output_pre_allocated);
      } else if (shape.size() == 3) {
        decompress<3, float, DeviceType>(shape, compressed_data,
                                         compressed_size, decompressed_data,
                                         coords, config, output_pre_allocated);
      } else if (shape.size() == 4) {
        decompress<4, float, DeviceType>(shape, compressed_data,
                                         compressed_size, decompressed_data,
                                         coords, config, output_pre_allocated);
      } else if (shape.size() == 5) {
        decompress<5, float, DeviceType>(shape, compressed_data,
                                         compressed_size, decompressed_data,
                                         coords, config, output_pre_allocated);
      } else {
        std::cout << log::log_err
                  << "do not support higher than five dimentions!\n";
        exit(-1);
      }
    }
  } else if (dtype == data_type::Double) {
    if (dstype == data_structure_type::Cartesian_Grid_Uniform) {
      if (shape.size() == 1) {
        decompress<1, double, DeviceType>(shape, compressed_data,
                                          compressed_size, decompressed_data,
                                          config, output_pre_allocated);
      } else if (shape.size() == 2) {
        decompress<2, double, DeviceType>(shape, compressed_data,
                                          compressed_size, decompressed_data,
                                          config, output_pre_allocated);
      } else if (shape.size() == 3) {
        decompress<3, double, DeviceType>(shape, compressed_data,
                                          compressed_size, decompressed_data,
                                          config, output_pre_allocated);
      } else if (shape.size() == 4) {
        decompress<4, double, DeviceType>(shape, compressed_data,
                                          compressed_size, decompressed_data,
                                          config, output_pre_allocated);
      } else if (shape.size() == 5) {
        decompress<5, double, DeviceType>(shape, compressed_data,
                                          compressed_size, decompressed_data,
                                          config, output_pre_allocated);
      } else {
        std::cout << log::log_err
                  << "do not support higher than five dimentions!\n";
        exit(-1);
      }
    } else {
      std::cout << log::log_err
                << "do not support types other than double and float!\n";
      exit(-1);
    }
  } else if (dstype == data_structure_type::Cartesian_Grid_Non_Uniform) {

    std::vector<double *> coords =
        infer_coords<double>(compressed_data, compressed_size);

    if (shape.size() == 1) {
      decompress<1, double, DeviceType>(shape, compressed_data, compressed_size,
                                        decompressed_data, coords, config,
                                        output_pre_allocated);
    } else if (shape.size() == 2) {
      decompress<2, double, DeviceType>(shape, compressed_data, compressed_size,
                                        decompressed_data, coords, config,
                                        output_pre_allocated);
    } else if (shape.size() == 3) {
      decompress<3, double, DeviceType>(shape, compressed_data, compressed_size,
                                        decompressed_data, coords, config,
                                        output_pre_allocated);
    } else if (shape.size() == 4) {
      decompress<4, double, DeviceType>(shape, compressed_data, compressed_size,
                                        decompressed_data, coords, config,
                                        output_pre_allocated);
    } else if (shape.size() == 5) {
      decompress<5, double, DeviceType>(shape, compressed_data, compressed_size,
                                        decompressed_data, coords, config,
                                        output_pre_allocated);
    } else {
      std::cout << log::log_err
                << "do not support higher than five dimentions!\n";
      exit(-1);
    }
  }
}

template <typename DeviceType>
void decompress(const void *compressed_data, size_t compressed_size,
                void *&decompressed_data, bool output_pre_allocated) {
  Config config;
  decompress<DeviceType>(compressed_data, compressed_size, decompressed_data,
                         config, output_pre_allocated);
}

enum device_type auto_detect_device() {
  enum device_type dev_type = device_type::None;
#ifdef MGARD_ENABLE_SERIAL
  dev_type = device_type::Serial;
#endif
#ifdef MGARD_ENABLE_CUDA
  if (deviceAvailable<CUDA>()) {
    dev_type = device_type::CUDA;
  }
#endif
#ifdef MGARD_ENABLE_HIP
  if (deviceAvailable<HIP>()) {
    dev_type = device_type::HIP;
  }
#endif
  if (dev_type == device_type::None) {
    std::cout << log::log_err << "MGARD-X was not built with any backend.\n";
    exit(-1);
  }
  return dev_type;
}

void compress(DIM D, data_type dtype, std::vector<SIZE> shape, double tol,
              double s, enum error_bound_type mode, const void *original_data,
              void *&compressed_data, size_t &compressed_size, Config config,
              bool output_pre_allocated) {

  enum device_type dev_type = config.dev_type;
  if (dev_type == device_type::Auto) {
    dev_type = auto_detect_device();
  }

  if (dev_type == device_type::Serial) {
#ifdef MGARD_ENABLE_SERIAL
    compress<Serial>(D, dtype, shape, tol, s, mode, original_data,
                     compressed_data, compressed_size, config,
                     output_pre_allocated);
#else
    std::cout << log::log_err << "MGARD-X was not built with Serial backend.\n";
    exit(-1);
#endif
  } else if (dev_type == device_type::CUDA) {
#ifdef MGARD_ENABLE_CUDA
    compress<CUDA>(D, dtype, shape, tol, s, mode, original_data,
                   compressed_data, compressed_size, config,
                   output_pre_allocated);
#else
    std::cout << log::log_err << "MGARD-X was not built with CUDA backend.\n";
    exit(-1);
#endif
  } else if (dev_type == device_type::HIP) {
#ifdef MGARD_ENABLE_HIP
    compress<HIP>(D, dtype, shape, tol, s, mode, original_data, compressed_data,
                  compressed_size, config, output_pre_allocated);
#else
    std::cout << log::log_err << "MGARD-X was not built with HIP backend.\n";
    exit(-1);
#endif
  } else {
    std::cout << log::log_err << "Unsupported backend.\n";
  }
}

void compress(DIM D, data_type dtype, std::vector<SIZE> shape, double tol,
              double s, enum error_bound_type mode, const void *original_data,
              void *&compressed_data, size_t &compressed_size,
              bool output_pre_allocated) {

  enum device_type dev_type = auto_detect_device();

  if (dev_type == device_type::Serial) {
#ifdef MGARD_ENABLE_SERIAL
    compress<Serial>(D, dtype, shape, tol, s, mode, original_data,
                     compressed_data, compressed_size, output_pre_allocated);
#else
    std::cout << log::log_err << "MGARD-X was not built with Serial backend.\n";
    exit(-1);
#endif
  } else if (dev_type == device_type::CUDA) {
#ifdef MGARD_ENABLE_CUDA
    compress<CUDA>(D, dtype, shape, tol, s, mode, original_data,
                   compressed_data, compressed_size, output_pre_allocated);
#else
    std::cout << log::log_err << "MGARD-X was not built with CUDA backend.\n";
    exit(-1);
#endif
  } else if (dev_type == device_type::HIP) {
#ifdef MGARD_ENABLE_HIP
    compress<HIP>(D, dtype, shape, tol, s, mode, original_data, compressed_data,
                  compressed_size, output_pre_allocated);
#else
    std::cout << log::log_err << "MGARD-X was not built with HIP backend.\n";
    exit(-1);
#endif
  } else {
    std::cout << log::log_err << "Unsupported backend.\n";
  }
}

void compress(DIM D, data_type dtype, std::vector<SIZE> shape, double tol,
              double s, enum error_bound_type mode, const void *original_data,
              void *&compressed_data, size_t &compressed_size,
              std::vector<const Byte *> coords, Config config,
              bool output_pre_allocated) {

  enum device_type dev_type = config.dev_type;
  if (dev_type == device_type::Auto) {
    dev_type = auto_detect_device();
  }

  if (dev_type == device_type::Serial) {
#ifdef MGARD_ENABLE_SERIAL
    compress<Serial>(D, dtype, shape, tol, s, mode, original_data,
                     compressed_data, compressed_size, coords, config,
                     output_pre_allocated);
#else
    std::cout << log::log_err << "MGARD-X was not built with Serial backend.\n";
    exit(-1);
#endif
  } else if (dev_type == device_type::CUDA) {
#ifdef MGARD_ENABLE_CUDA
    compress<CUDA>(D, dtype, shape, tol, s, mode, original_data,
                   compressed_data, compressed_size, coords, config,
                   output_pre_allocated);
#else
    std::cout << log::log_err << "MGARD-X was not built with CUDA backend.\n";
    exit(-1);
#endif
  } else if (dev_type == device_type::HIP) {
#ifdef MGARD_ENABLE_HIP
    compress<HIP>(D, dtype, shape, tol, s, mode, original_data, compressed_data,
                  compressed_size, coords, config, output_pre_allocated);
#else
    std::cout << log::log_err << "MGARD-X was not built with HIP backend.\n";
    exit(-1);
#endif
  } else {
    std::cout << log::log_err << "Unsupported backend.\n";
  }
}

void compress(DIM D, data_type dtype, std::vector<SIZE> shape, double tol,
              double s, enum error_bound_type mode, const void *original_data,
              void *&compressed_data, size_t &compressed_size,
              std::vector<const Byte *> coords, bool output_pre_allocated) {

  enum device_type dev_type = auto_detect_device();

  if (dev_type == device_type::Serial) {
#ifdef MGARD_ENABLE_SERIAL
    compress<Serial>(D, dtype, shape, tol, s, mode, original_data,
                     compressed_data, compressed_size, coords,
                     output_pre_allocated);
#else
    std::cout << log::log_err << "MGARD-X was not built with Serial backend.\n";
    exit(-1);
#endif
  } else if (dev_type == device_type::CUDA) {
#ifdef MGARD_ENABLE_CUDA
    compress<CUDA>(D, dtype, shape, tol, s, mode, original_data,
                   compressed_data, compressed_size, coords,
                   output_pre_allocated);
#else
    std::cout << log::log_err << "MGARD-X was not built with CUDA backend.\n";
    exit(-1);
#endif
  } else if (dev_type == device_type::HIP) {
#ifdef MGARD_ENABLE_HIP
    compress<HIP>(D, dtype, shape, tol, s, mode, original_data, compressed_data,
                  compressed_size, coords, output_pre_allocated);
#else
    std::cout << log::log_err << "MGARD-X was not built with HIP backend.\n";
    exit(-1);
#endif
  } else {
    std::cout << log::log_err << "Unsupported backend.\n";
  }
}

void decompress(const void *compressed_data, size_t compressed_size,
                void *&decompressed_data, Config config,
                bool output_pre_allocated) {

  enum device_type dev_type = config.dev_type;
  if (dev_type == device_type::Auto) {
    dev_type = auto_detect_device();
  }

  if (dev_type == device_type::Serial) {
#ifdef MGARD_ENABLE_SERIAL
    decompress<Serial>(compressed_data, compressed_size, decompressed_data,
                       config, output_pre_allocated);
#else
    std::cout << log::log_err << "MGARD-X was not built with Serial backend.\n";
    exit(-1);
#endif
  } else if (dev_type == device_type::CUDA) {
#ifdef MGARD_ENABLE_CUDA
    decompress<CUDA>(compressed_data, compressed_size, decompressed_data,
                     config, output_pre_allocated);
#else
    std::cout << log::log_err << "MGARD-X was not built with CUDA backend.\n";
    exit(-1);
#endif
  } else if (dev_type == device_type::HIP) {
#ifdef MGARD_ENABLE_HIP
    decompress<HIP>(compressed_data, compressed_size, decompressed_data, config,
                    output_pre_allocated);
#else
    std::cout << log::log_err << "MGARD-X was not built with HIP backend.\n";
    exit(-1);
#endif
  } else {
    std::cout << log::log_err << "Unsupported backend.\n";
  }
}

void decompress(const void *compressed_data, size_t compressed_size,
                void *&decompressed_data, bool output_pre_allocated) {

  enum device_type dev_type = auto_detect_device();

  if (dev_type == device_type::Serial) {
#ifdef MGARD_ENABLE_SERIAL
    decompress<Serial>(compressed_data, compressed_size, decompressed_data,
                       output_pre_allocated);
#else
    std::cout << log::log_err << "MGARD-X was not built with Serial backend.\n";
    exit(-1);
#endif
  } else if (dev_type == device_type::CUDA) {
#ifdef MGARD_ENABLE_CUDA
    decompress<CUDA>(compressed_data, compressed_size, decompressed_data,
                     output_pre_allocated);
#else
    std::cout << log::log_err << "MGARD-X was not built with CUDA backend.\n";
    exit(-1);
#endif
  } else if (dev_type == device_type::HIP) {
#ifdef MGARD_ENABLE_HIP
    decompress<HIP>(compressed_data, compressed_size, decompressed_data,
                    output_pre_allocated);
#else
    std::cout << log::log_err << "MGARD-X was not built with HIP backend.\n";
    exit(-1);
#endif
  } else {
    std::cout << log::log_err << "Unsupported backend.\n";
  }
}

void BeginAutoTuning(enum device_type dev_type) {

  if (dev_type == device_type::Auto) {
    dev_type = auto_detect_device();
  }

  if (dev_type == device_type::Serial) {
#ifdef MGARD_ENABLE_SERIAL
    mgard_x::BeginAutoTuning<mgard_x::Serial>();
#else
    std::cout << log::log_err << "MGARD-X was not built with Serial backend.\n";
    exit(-1);
#endif
  } else if (dev_type == device_type::CUDA) {
#ifdef MGARD_ENABLE_CUDA
    mgard_x::BeginAutoTuning<mgard_x::CUDA>();
#else
    std::cout << log::log_err << "MGARD-X was not built with CUDA backend.\n";
    exit(-1);
#endif
  } else if (dev_type == device_type::HIP) {
#ifdef MGARD_ENABLE_HIP
    mgard_x::BeginAutoTuning<mgard_x::HIP>();
#else
    std::cout << log::log_err << "MGARD-X was not built with HIP backend.\n";
    exit(-1);
#endif
  } else {
    std::cout << log::log_err << "Unsupported backend.\n";
  }
}

void EndAutoTuning(enum device_type dev_type) {

  if (dev_type == device_type::Auto) {
    dev_type = auto_detect_device();
  }

  if (dev_type == device_type::Serial) {
#ifdef MGARD_ENABLE_SERIAL
    mgard_x::EndAutoTuning<mgard_x::Serial>();
#else
    std::cout << log::log_err << "MGARD-X was not built with Serial backend.\n";
    exit(-1);
#endif
  } else if (dev_type == device_type::CUDA) {
#ifdef MGARD_ENABLE_CUDA
    mgard_x::EndAutoTuning<mgard_x::CUDA>();
#else
    std::cout << log::log_err << "MGARD-X was not built with CUDA backend.\n";
    exit(-1);
#endif
  } else if (dev_type == device_type::HIP) {
#ifdef MGARD_ENABLE_HIP
    mgard_x::EndAutoTuning<mgard_x::HIP>();
#else
    std::cout << log::log_err << "MGARD-X was not built with HIP backend.\n";
    exit(-1);
#endif
  } else {
    std::cout << log::log_err << "Unsupported backend.\n";
  }
}

} // namespace mgard_x
