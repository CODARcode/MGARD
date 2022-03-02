#ifndef _MDR_BITPLANE_ENCODER_INTERFACE_HPP
#define _MDR_BITPLANE_ENCODER_INTERFACE_HPP

#include <cassert>
namespace MDR {
namespace concepts {
#define UINT8_BITS 8
// concept of encoder which encodes T type data into bitstreams
template <typename T> class BitplaneEncoderInterface {
public:
  virtual ~BitplaneEncoderInterface() = default;

  virtual std::vector<uint8_t *>
  encode(T const *data, uint32_t n, int32_t exp, uint8_t num_bitplanes,
         std::vector<uint32_t> &streams_sizes) const = 0;

  virtual T *decode(const std::vector<uint8_t const *> &streams, uint32_t n,
                    int exp, uint8_t num_bitplanes) = 0;

  virtual T *progressive_decode(const std::vector<uint8_t const *> &streams,
                                uint32_t n, int exp, uint8_t starting_bitplane,
                                uint8_t num_bitplanes, int level) = 0;

  virtual void print() const = 0;
};
} // namespace concepts
} // namespace MDR

namespace mgard_x {
namespace MDR {
namespace concepts {
// concept of encoder which encodes T type data into bitstreams
template <typename T_data, typename T_bitplane, typename T_error>
class BitplaneEncoderInterface {
public:
  virtual ~BitplaneEncoderInterface() = default;

  virtual Array<2, T_bitplane, CUDA>
  encode(SIZE n, SIZE num_bitplanes, int32_t exp,
         SubArray<1, T_data, CUDA> v,
         SubArray<1, T_error, CUDA> level_errors,
         std::vector<SIZE> &streams_sizes, int queue_idx) const = 0;

  virtual Array<1, T_data, CUDA>
  decode(SIZE n, SIZE num_bitplanes, int32_t exp,
         SubArray<2, T_bitplane, CUDA> encoded_bitplanes,
         int level, int queue_idx) = 0;

  virtual Array<1, T_data, CUDA> progressive_decode(
      SIZE n, SIZE starting_bitplanes,
      SIZE num_bitplanes, int32_t exp,
      SubArray<2, T_bitplane, CUDA> encoded_bitplanes,
      int level, int queue_idx) = 0;

  virtual SIZE buffer_size(SIZE n) const = 0;

  virtual void print() const = 0;
};
} // namespace concepts
} // namespace MDR
} // namespace mgard_x
#endif
