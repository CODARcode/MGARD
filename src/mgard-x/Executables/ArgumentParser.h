#include <cstdlib>
#include <stdexcept>
#include <string>

bool has_arg(int argc, char *argv[], std::string option_abbr,
             std::string option_full) {
  for (int i = 0; i < argc; i++) {
    if (option_abbr.compare(std::string(argv[i])) == 0 ||
        option_full.compare(std::string(argv[i])) == 0) {
      return true;
    }
  }
  return false;
}

bool require_arg(int argc, char *argv[], std::string option_abbr,
                 std::string option_full) {
  for (int i = 0; i < argc; i++) {
    if (has_arg(argc, argv, option_abbr, option_full)) {
      return true;
    }
  }
  throw std::runtime_error("Missing option: " + option_abbr + " or " +
                           option_full);
  return false;
}

template <typename T>
T get_arg(int argc, char *argv[], std::string option_name,
          std::string option_abbr, std::string option_full) {
  if (require_arg(argc, argv, option_abbr, option_full)) {
    for (int i = 0; i < argc; i++) {
      if (option_abbr.compare(std::string(argv[i])) == 0 ||
          option_full.compare(std::string(argv[i])) == 0) {
        try {
          std::string arg(argv[i + 1]);
          mgard_x::log::info(option_name + ": " + std::string(arg), true);
          if constexpr (std::is_same<T, std::string>::value) {
            return arg;
          } else if constexpr (std::is_integral<T>::value) {
            return std::stoi(arg);
          } else if constexpr (std::is_same<T, double>::value) {
            return std::stod(arg);
          } else {
            return 0;
          }
        } catch (std::invalid_argument const &e) {
          throw std::runtime_error("Illegal argument for option: " +
                                   option_abbr + " or " + option_full);
        }
      }
    }
  }
  return 0;
}

template <typename T>
std::vector<T> get_args(int argc, char *argv[], std::string option_name,
                        std::string option_abbr, std::string option_full) {
  std::vector<T> args;
  if (require_arg(argc, argv, option_abbr, option_full)) {
    std::string arg;
    int arg_idx = 0, i;
    for (i = 0; i < argc; i++) {
      if (option_abbr.compare(std::string(argv[i])) == 0 ||
          option_full.compare(std::string(argv[i])) == 0) {
        arg = std::string(argv[i + 1]);
        arg_idx = i + 1;
      }
    }
    try {
      int d = std::stoi(arg);
      std::string output_string = "";
      for (int i = 0; i < d; i++) {
        std::string arg(argv[arg_idx + 1 + i]);
        output_string = output_string + arg + " ";
        if constexpr (std::is_same<T, std::string>::value) {
          args.push_back(std::string(argv[arg_idx + 1 + i]));
        } else if constexpr (std::is_integral<T>::value) {
          args.push_back(std::stoi(arg));
        } else if constexpr (std::is_same<T, double>::value) {
          args.push_back(std::stod(arg));
        }
      }
      mgard_x::log::info(option_name + ": " + output_string, true);
    } catch (std::invalid_argument const &e) {
      throw std::runtime_error("Illegal argument for option: " + option_abbr +
                               " or " + option_full);
    }
  }
  return args;
}

enum mgard_x::data_type get_data_type(int argc, char *argv[]) {
  enum mgard_x::data_type dtype;
  std::string dt =
      get_arg<std::string>(argc, argv, "Data type", "-dt", "--data-type");
  if (dt == "s" || dt == "single") {
    dtype = mgard_x::data_type::Float;
  } else if (dt == "d" || dt == "double") {
    dtype = mgard_x::data_type::Double;
  } else {
    throw std::runtime_error("Wrong data type.");
  }
  return dtype;
}

enum mgard_x::error_bound_type get_error_bound_mode(int argc, char *argv[]) {
  enum mgard_x::error_bound_type mode; // REL or ABS
  std::string em = get_arg<std::string>(argc, argv, "Error mode", "-em",
                                        "--error-bound-mode");
  if (em.compare("rel") == 0) {
    mode = mgard_x::error_bound_type::REL;
  } else if (em.compare("abs") == 0) {
    mode = mgard_x::error_bound_type::ABS;
  } else {
    throw std::runtime_error("Wrong bound mode.");
  }
  return mode;
}

enum mgard_x::device_type get_device_type(int argc, char *argv[]) {
  enum mgard_x::device_type dev_type = mgard_x::device_type::AUTO;
  std::string dev =
      get_arg<std::string>(argc, argv, "Device", "-d", "--device");
  if (dev.compare("auto") == 0) {
    dev_type = mgard_x::device_type::AUTO;
  } else if (dev.compare("serial") == 0) {
    dev_type = mgard_x::device_type::SERIAL;
  } else if (dev.compare("openmp") == 0) {
    dev_type = mgard_x::device_type::OPENMP;
  } else if (dev.compare("cuda") == 0) {
    dev_type = mgard_x::device_type::CUDA;
  } else if (dev.compare("hip") == 0) {
    dev_type = mgard_x::device_type::HIP;
  } else if (dev.compare("sycl") == 0) {
    dev_type = mgard_x::device_type::HIP;
  } else {
    throw std::runtime_error("wrong device type.");
  }
  return dev_type;
}